"""Stage orchestration.

Each stage is independently runnable and reads its inputs from disk, so any of
them can be re-run on an existing run directory without repeating the ones
before it. When several stages run together the trained model is loaded once
and shared, rather than being deserialised again for each.
"""

from __future__ import annotations

from collections.abc import Callable

from naviernet.config.schema import STAGES, resolved_datasets
from naviernet.data.preprocess import preprocess
from naviernet.physics.groups import save_groups
from naviernet.utils.logging import format_mapping, get_logger
from naviernet.utils.paths import RunPaths

log = get_logger(__name__)


class Pipeline:
    """Runs stages against one config and run directory."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.paths = RunPaths.from_config(cfg).ensure()
        self._model = None
        self._data = None

    # -- stages ------------------------------------------------------------
    def preprocess(self) -> dict:
        """Raw TIFFs -> calibrated tensors, plus the dimensionless groups."""
        groups = save_groups(self.cfg, self.paths.groups_json)
        log.info("dimensionless groups: %s", format_mapping(groups))
        meta = preprocess(self.cfg, self.paths)
        # The tensors changed underneath us; drop anything cached from them.
        self._model = self._data = None
        return meta

    def train(
        self,
        steps: int | None = None,
        on_log: Callable[[dict], None] | None = None,
    ):
        from naviernet import training

        model, data, state = training.train(self.cfg, self.paths, steps=steps, on_log=on_log)
        self._model, self._data = model, data
        return state

    def evaluate(self) -> dict:
        # A joint (multi-dataset) run scores every dataset it spans, each with its
        # own conditioning; a single-dataset run keeps the original report.
        if len(resolved_datasets(self.cfg)) > 1:
            from naviernet import training
            from naviernet.evaluation import evaluate_joint

            model, contexts, heldout = training.load_joint(self.cfg, self.paths)
            return evaluate_joint(
                self.cfg, model, contexts, self.paths, heldout_datasets=heldout
            )

        from naviernet.evaluation import evaluate

        model, data = self._load()
        return evaluate(self.cfg, model, data, self.paths)

    def figures(self) -> None:
        from naviernet.viz import render_all_figures

        if len(resolved_datasets(self.cfg)) > 1:
            for cfg_ds, bound, context, paths in self._joint_viz_scenes():
                log.info("rendering figures for joint dataset %s", context.name)
                render_all_figures(cfg_ds, bound, context.data, paths)
            return

        model, data = self._load()
        render_all_figures(self.cfg, model, data, self.paths)

    def video(self, n_t: int | None = None):
        import shutil

        from naviernet.viz import render_video

        if len(resolved_datasets(self.cfg)) > 1:
            if shutil.which("ffmpeg") is None:
                # A missing renderer must not fail the whole joint run's report;
                # single-dataset runs keep the hard error (their video IS the stage).
                log.warning("ffmpeg not found; skipping joint-run videos")
                return None
            rendered = None
            for cfg_ds, bound, context, paths in self._joint_viz_scenes():
                log.info("rendering video for joint dataset %s", context.name)
                rendered = render_video(cfg_ds, bound, context.data, paths, n_t=n_t)
            return rendered

        model, data = self._load()
        return render_video(self.cfg, model, data, self.paths, n_t=n_t)

    def _joint_viz_scenes(self):
        """Per-dataset (cfg, bound model, context, paths) for joint-run viz.

        Each dataset renders with its own conditioning row bound into the model,
        its own reference scales (from its tensors), and its own figures/video
        subdirectory — the single-dataset viz path runs unchanged on top.
        """
        from omegaconf import OmegaConf

        from naviernet import training
        from naviernet.utils.paths import DatasetVizPaths

        model, contexts, _ = training.load_joint(self.cfg, self.paths)
        for context in contexts:
            cfg_ds = OmegaConf.merge(self.cfg)  # a per-dataset copy
            OmegaConf.set_readonly(cfg_ds, False)  # merge keeps the source flag
            meta = context.data.meta
            if meta.get("L_ref_um") is not None:
                cfg_ds.scales.L_ref_um = float(meta["L_ref_um"])
            if meta.get("U_ref") is not None:
                cfg_ds.scales.U_ref = float(meta["U_ref"])
            OmegaConf.set_readonly(cfg_ds, True)
            data_paths = self.paths.for_dataset(context.name)
            paths = DatasetVizPaths(
                raw_dir=data_paths.raw_dir,
                processed_dir=data_paths.processed_dir,
                output_dir=self.paths.output_dir,
                viz_dataset=context.name,
            ).ensure()
            yield cfg_ds, model.bound(context.c), context, paths

    # -- dispatch ----------------------------------------------------------
    def run(self, stage: str) -> None:
        """Run one stage, or every stage in order when ``stage`` is ``all``."""
        if stage != "all" and stage not in STAGES:
            raise ValueError(
                f"unknown stage {stage!r}; expected one of {', '.join(STAGES)}, or all"
            )
        for name in STAGES if stage == "all" else (stage,):
            log.info("=== stage: %s ===", name)
            getattr(self, name)()

    def _load(self):
        """Trained model and dataset, deserialised at most once per process."""
        if self._model is None or self._data is None:
            from naviernet import training

            self._model, self._data, _ = training.load_model(self.cfg, self.paths)
        return self._model, self._data

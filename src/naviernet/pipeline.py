"""Stage orchestration.

Each stage is independently runnable and reads its inputs from disk, so any of
them can be re-run on an existing run directory without repeating the ones
before it. When several stages run together the trained model is loaded once
and shared, rather than being deserialised again for each.
"""

from __future__ import annotations

from naviernet.config.schema import STAGES
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

    def train(self, steps: int | None = None, on_log=None):
        from naviernet import training

        model, data, state = training.train(self.cfg, self.paths, steps=steps, on_log=on_log)
        self._model, self._data = model, data
        return state

    def evaluate(self) -> dict:
        from naviernet.evaluation import evaluate

        model, data = self._load()
        return evaluate(self.cfg, model, data, self.paths)

    def figures(self) -> None:
        from naviernet.viz import render_all_figures

        model, data = self._load()
        render_all_figures(self.cfg, model, data, self.paths)

    def video(self, n_t: int | None = None):
        from naviernet.viz import render_video

        model, data = self._load()
        return render_video(self.cfg, model, data, self.paths, n_t=n_t)

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

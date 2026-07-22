"""Command-line entry point.

Hydra composes the config from ``configs/``, so anything in the schema is
overridable on the command line and the fully-resolved config for every run is
snapshotted next to its results::

    naviernet stage=preprocess
    naviernet stage=train training.steps=500
    naviernet stage=all run_name=sharper model.alpha_eps=0.03
    naviernet --multirun training.seed=0,1,2

Run ``naviernet --help`` for the generated help, or ``naviernet --cfg job`` to
print the composed config without running anything.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from naviernet.config import config_dir, register_configs
from naviernet.pipeline import Pipeline
from naviernet.utils.logging import get_logger

log = get_logger(__name__)

register_configs()


@hydra.main(version_base="1.3", config_path=str(config_dir()), config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("run_name=%s dataset=%s stage=%s", cfg.run_name, cfg.dataset, cfg.stage)
    log.debug("composed config:\n%s", OmegaConf.to_yaml(cfg))
    Pipeline(cfg).run(cfg.stage)


if __name__ == "__main__":
    main()

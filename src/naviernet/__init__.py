"""naviernet — physics-informed neural networks for multiphase microchannel flow.

Reconstructs the continuous velocity and interface fields of a growing vapour
bubble in a heated microchannel from a handful of high-speed camera frames, by
constraining a neural network with the governing conservation laws.

Typical use from Python::

    from hydra import compose, initialize_config_dir
    from naviernet.pipeline import Pipeline

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        cfg = compose(config_name="config", overrides=["training.steps=100"])
    Pipeline(cfg).run("train")

From the shell, use the ``naviernet`` command (see :mod:`naviernet.cli`).
"""

__version__ = "0.1.0"

__all__ = ["__version__"]

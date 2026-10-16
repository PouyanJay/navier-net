#!/usr/bin/env python
"""Write the field table's JSON copy, and the trusted recipe's stamp, for the web app.

The web renders its forms from ``apps/web/src/lib/fieldTable.json``, a copy of
``naviernet.config.fields.FIELD_TABLE``, and stamps its "Trusted fields" recipe
from ``apps/web/src/lib/trustedRecipe.json``, the launch-form values that
``model=trusted training=trusted`` compose to. ``tests/test_field_table.py``
fails when either copy is stale, naming this script. Run it after editing the
table or the trusted recipes::

    .venv/bin/python scripts/export_field_table.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_LIB = REPO_ROOT / "apps" / "web" / "src" / "lib"
TABLE_COPY = WEB_LIB / "fieldTable.json"
RECIPE_COPY = WEB_LIB / "trustedRecipe.json"

# The CLI recipe the web's "Trusted fields" card stamps.
TRUSTED_OVERRIDES = ["model=trusted", "training=trusted"]
# Launch fields the recipe leaves to the form: the seed is the user's, and the
# server fixes the device.
NOT_STAMPED = {"seed", "device"}


ENCODING = "utf-8"  # the table carries µ, ° and ²


def _current(path: Path, text: str) -> bool:
    return path.is_file() and path.read_text(encoding=ENCODING) == text


def _dumps(payload: object) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def render_table() -> str:
    """The table as the JSON text the web copy holds."""
    from naviernet.config.fields import export

    return _dumps(export())


def render_recipe() -> str:
    """The trusted recipe as launch-form values: every form-owned key the API's
    override list builds (the launch request's own fields), read off the
    composed CLI config, plus the Stage-A loss weights the form owns."""
    from hydra import compose, initialize_config_dir
    from naviernet_api.models import RunLaunchRequest
    from naviernet_api.services.overrides import FORM_OWNED_KEYS
    from omegaconf import OmegaConf

    from naviernet.config import config_dir, register_configs

    register_configs()
    with initialize_config_dir(config_dir=str(config_dir()), version_base="1.3"):
        cfg = compose(config_name="config", overrides=TRUSTED_OVERRIDES)
    stamp: dict[str, object] = {}
    for key in sorted(FORM_OWNED_KEYS):
        name = key.rsplit(".", 1)[1]
        if name in RunLaunchRequest.model_fields and name not in NOT_STAMPED:
            stamp[name] = OmegaConf.select(cfg, key)
    stamp["weights"] = {
        term: cfg.training.weights[term] for term in ("data", "vof", "div", "src", "bc")
    }
    return _dumps(stamp)


def main(argv: list[str] | None = None) -> int:
    """Write both copies, or with ``--check`` only report whether they are current."""
    parser = argparse.ArgumentParser(
        description="export the field table and the trusted recipe"
    )
    parser.add_argument("--check", action="store_true", help="report, do not write")
    args = parser.parse_args(argv)
    outputs = {TABLE_COPY: render_table(), RECIPE_COPY: render_recipe()}
    stale = [path for path, text in outputs.items() if not _current(path, text)]
    if args.check:
        for path in outputs:
            print(f"{path.relative_to(REPO_ROOT)}: {'STALE' if path in stale else 'current'}")
        return 1 if stale else 0
    for path, text in outputs.items():
        path.write_text(text, encoding=ENCODING)
        print(f"wrote {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

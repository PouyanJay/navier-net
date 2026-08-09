"""RunLaunchRequest defaults vs configs/training/stage_a.yaml: one configuration.

The launch request's docstring promises its numeric defaults map 1:1 onto
``cfg.training`` as the CLI would compose it, and the web form's defaults chain
off the same file -- but nothing ever checked the promise, so the copies could
drift silently. This is that check, over every field the two actually share.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from naviernet_api.models import RunLaunchRequest

REPO_ROOT = Path(__file__).parents[3]

# The launch fields whose defaults claim to mirror the CLI's Stage-A training
# config. Fields the platform deliberately owns differently (val_fraction is a
# platform axis the YAML does not carry; device is forced to cpu server-side)
# are not shared and are not listed.
SHARED_FIELDS = (
    "steps",
    "lr",
    "lr_halflife",
    "n_data",
    "n_coll",
    "n_bc",
    "holdout_frame",
    "rebalance_every",
    "log_every",
    "seed",
)


def test_launch_defaults_match_the_cli_training_config():
    config = yaml.safe_load((REPO_ROOT / "configs" / "training" / "stage_a.yaml").read_text())
    request = RunLaunchRequest(dataset="any")

    # Bare == on the floats is deliberate: both sides are hand-written literals
    # parsed independently, and the test exists to catch drift between exactly
    # those literals -- approx would let one side be rounded without notice.
    for field in SHARED_FIELDS:
        assert getattr(request, field) == config[field], (
            f"RunLaunchRequest.{field} default {getattr(request, field)!r} has "
            f"drifted from configs/training/stage_a.yaml's {config[field]!r}"
        )
    for term in ("data", "vof", "div", "src", "bc"):
        assert getattr(request.weights, term) == config["weights"][term], (
            f"weights.{term} default has drifted from stage_a.yaml"
        )

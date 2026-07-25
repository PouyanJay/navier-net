"""The fluids catalogue endpoint.

Lists every characterised working fluid with its saturated properties and its
(atmospheric) saturation temperature, so the new-series form can offer the choice
and show what physics each fluid commits to.
"""

from __future__ import annotations


def test_lists_available_fluids(client):
    r = client.get("/api/fluids")
    assert r.status_code == 200
    fluids = r.json()
    assert isinstance(fluids, list) and fluids

    by_id = {f["id"]: f for f in fluids}
    assert "fc72" in by_id
    fc72 = by_id["fc72"]
    assert fc72["name"] == "FC-72"
    assert fc72["T_sat_C"] == 56.6
    # The full saturated-property set the pipeline needs is present.
    for key in ("rho_l", "rho_v", "mu_l", "k_l", "cp_l", "sigma", "h_lv"):
        assert isinstance(fc72[key], (int, float))


def test_base_fluid_template_is_not_listed(client):
    """The abstract ``base_fluid`` schema node is not a selectable fluid."""
    ids = {f["id"] for f in client.get("/api/fluids").json()}
    assert "base_fluid" not in ids

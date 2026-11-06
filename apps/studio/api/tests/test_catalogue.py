"""The quantity catalogue: complete, self-consistent, and honest about what a
run can produce."""

from __future__ import annotations

from fastapi.testclient import TestClient
from naviernet_studio import catalogue
from naviernet_studio.scales import Scales

from conftest import RUN_ID


def test_every_entry_explains_itself():
    for q in catalogue.ALL:
        assert q.label and q.plain and q.how, q.id
        assert q.kind in {
            catalogue.KIND_VOLUME,
            catalogue.KIND_DERIVED,
            catalogue.KIND_RESIDUAL,
            catalogue.KIND_TERM,
            catalogue.KIND_EDGE,
        }, q.id


def test_ids_are_unique_and_scales_exist():
    ids = [q.id for q in catalogue.ALL]
    assert len(ids) == len(set(ids))
    fields = set(Scales.__dataclass_fields__)
    for q in catalogue.ALL:
        assert q.scale == "" or q.scale in fields, (q.id, q.scale)


def test_components_point_at_catalogued_terms():
    for q in catalogue.ALL:
        for part in q.components:
            assert part in catalogue.BY_ID, (q.id, part)


def test_the_served_catalogue_marks_what_the_tiny_run_lacks(client: TestClient):
    body = client.get("/api/catalogue", params={"run": RUN_ID}).json()

    by_id = {q["id"]: q for q in body}
    assert by_id["u"]["available"] is True
    assert by_id["u"]["factor"] == 200.0
    # A Stage-A default run has no explicit front, so edge quantities are off.
    assert by_id["yl_jump"]["available"] is False
    assert client.get("/api/catalogue").status_code == 200

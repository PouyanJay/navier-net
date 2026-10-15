"""The field table over HTTP: every user-facing input, as declared once."""

from __future__ import annotations

from fastapi.testclient import TestClient

from naviernet.config.fields import FIELD_TABLE


def test_the_field_table_is_served_in_table_order(client: TestClient):
    r = client.get("/api/config/fields")
    assert r.status_code == 200
    body = r.json()

    assert [entry["key"] for entry in body] == [spec.key for spec in FIELD_TABLE]
    assert len({entry["name"] for entry in body}) == len(body)


def test_an_entry_carries_what_a_form_needs_to_render_it(client: TestClient):
    entries = {entry["name"]: entry for entry in client.get("/api/config/fields").json()}

    walls = entries["walls_at"]
    assert walls["kind"] == "enum"
    assert walls["options"] == ["crop", "physical"]
    assert walls["default"] == "crop"
    assert walls["owner"] == "form" and walls["section"] == "physics"

    inlet = entries["T_inlet_C"]
    assert inlet["kind"] == "float" and inlet["optional"] is True
    assert inlet["unit"] == "°C" and inlet["default"] is None
    assert inlet["lo"] < inlet["hi"]

    assert entries["dt_frame_ms"]["baked"] is True
    assert entries["mirror_max"]["key"] == "evaluation.acceptance.mirror_max"

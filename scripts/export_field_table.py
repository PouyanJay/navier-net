#!/usr/bin/env python
"""Write the field table's JSON copy for the web app.

The web renders its forms from ``apps/web/src/lib/fieldTable.json``, a copy of
``naviernet.config.fields.FIELD_TABLE``; ``tests/test_field_table.py`` fails
when the copy is stale, naming this script. Run it after editing the table::

    .venv/bin/python scripts/export_field_table.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_COPY = REPO_ROOT / "apps" / "web" / "src" / "lib" / "fieldTable.json"


def render() -> str:
    """The table as the JSON text the web copy holds."""
    from naviernet.config.fields import export

    return json.dumps(export(), indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str]) -> int:
    """Write the copy, or with ``--check`` only report whether it is current."""
    text = render()
    current = WEB_COPY.is_file() and WEB_COPY.read_text() == text
    if "--check" in argv:
        print(f"{WEB_COPY.relative_to(REPO_ROOT)}: {'current' if current else 'STALE'}")
        return 0 if current else 1
    WEB_COPY.write_text(text)
    print(f"wrote {WEB_COPY.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

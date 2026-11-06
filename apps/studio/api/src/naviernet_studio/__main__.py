"""Run the studio service: ``naviernet-studio`` or ``python -m naviernet_studio``."""

from __future__ import annotations

import os

DEFAULT_PORT = 8010


def main() -> None:
    import uvicorn

    uvicorn.run(
        "naviernet_studio.app:app",
        host=os.environ.get("NAVIERNET_STUDIO_HOST", "127.0.0.1"),
        port=int(os.environ.get("NAVIERNET_STUDIO_PORT", str(DEFAULT_PORT))),
        reload=bool(os.environ.get("NAVIERNET_STUDIO_RELOAD")),
    )


if __name__ == "__main__":
    main()

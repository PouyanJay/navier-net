"""Run the API with uvicorn: `naviernet-api` or `python -m naviernet_api`."""

from __future__ import annotations

import os


def main() -> None:
    import uvicorn

    uvicorn.run(
        "naviernet_api.main:app",
        host=os.environ.get("NAVIERNET_API_HOST", "127.0.0.1"),
        port=int(os.environ.get("NAVIERNET_API_PORT", "8000")),
        reload=bool(os.environ.get("NAVIERNET_API_RELOAD")),
    )


if __name__ == "__main__":
    main()

"""FastAPI application factory for the naviernet platform API.

A thin HTTP layer over the naviernet pipeline. Routes are grouped by resource;
each reads from the on-disk artifacts the pipeline produces. No database.

When the web app has been built (`npm run build` → `apps/web/dist`), the same
server also serves it, so one process on one port is the whole deployed
platform. Without a build it stays an API-only server (dev mode: Vite serves
the UI and proxies /api here).
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from naviernet_api.routes import datasets, model, projects, runs, sweeps
from naviernet_api.settings import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    # The pipeline's INFO transcript is part of the product here: the solver
    # console streams it live, so it must clear the (WARNING) root default.
    logging.getLogger("naviernet").setLevel(logging.INFO)
    app = FastAPI(
        title="naviernet API",
        version="0.1.0",
        summary="HTTP interface to the naviernet PINN solver platform.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/healthz", tags=["meta"])
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(projects.router)
    app.include_router(runs.router)
    app.include_router(sweeps.router)
    app.include_router(datasets.router)
    app.include_router(model.router)
    _mount_web_app(app, settings.web_dist_dir)
    return app


def _mount_web_app(app: FastAPI, dist: Path) -> None:
    """Serve the built single-page app, if a build exists.

    Registered after every API router, so real routes always win; unknown
    /api/* paths still 404 instead of leaking index.html to API clients.
    """
    if not (dist / "index.html").is_file():
        return

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        if path.startswith("api/"):
            raise HTTPException(status_code=404, detail=f"no route /{path}")
        candidate = (dist / path).resolve()
        if path and candidate.is_relative_to(dist.resolve()) and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(dist / "index.html")  # SPA fallback: client-side views


app = create_app()

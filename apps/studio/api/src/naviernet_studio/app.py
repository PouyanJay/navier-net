"""The FastAPI application: routers under ``/api`` and, when a build exists,
the studio's static bundle on the same port."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from naviernet_studio.routes import budgets, catalogue, edge, fields, geometry, probes, runs
from naviernet_studio.settings import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="naviernet studio",
        version="0.1.0",
        summary="Evaluates a trained naviernet run on demand for the post-processing studio.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/healthz", tags=["meta"])
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    for module in (runs, catalogue, fields, probes, edge, geometry, budgets):
        app.include_router(module.router)
    _mount_web(app, settings.web_dist_dir)
    return app


def _mount_web(app: FastAPI, dist: Path) -> None:
    """Serve the built studio, if there is one; unknown /api paths still 404."""
    if not (dist / "index.html").is_file():
        return

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        if path.startswith("api/"):
            raise HTTPException(status_code=404, detail=f"no route /{path}")
        candidate = (dist / path).resolve()
        if path and candidate.is_relative_to(dist.resolve()) and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(dist / "index.html")


app = create_app()

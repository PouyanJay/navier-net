"""FastAPI application factory for the naviernet platform API.

A thin HTTP layer over the naviernet pipeline. Routes are grouped by resource;
each reads from the on-disk artifacts the pipeline produces. No database.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from naviernet_api.routes import datasets, model, runs, sweeps
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

    app.include_router(runs.router)
    app.include_router(sweeps.router)
    app.include_router(datasets.router)
    app.include_router(model.router)
    return app


app = create_app()

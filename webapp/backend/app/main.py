"""FastAPI entrypoint.

Run in dev with::

    uvicorn webapp.backend.app.main:app --reload --port 8000

Environment variables are documented in :mod:`webapp.backend.app.settings`.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from webapp.backend.app.routes import schemas
from webapp.backend.app.routes.series import router as series_router
from webapp.backend.app.services.state import AppState
from webapp.backend.app.settings import get_settings

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "This software is for research and education only. It is not a medical device "
    "and must not be used for clinical diagnosis, treatment, or any decision "
    "affecting patient care."
)


def create_app() -> FastAPI:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    settings = get_settings()
    AppState.init(settings)

    app = FastAPI(
        title="RSNA Aneurysm Webapp API",
        description=DISCLAIMER,
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health", response_model=schemas.HealthResponse, tags=["system"])
    def health() -> schemas.HealthResponse:
        return schemas.HealthResponse(
            status="ok",
            checkpoint_loaded=state.checkpoint_path is not None,
            device=str(state.predictor.device),
        )

    @app.get("/api/disclaimer", tags=["system"])
    def disclaimer() -> dict[str, str]:
        return {"disclaimer": DISCLAIMER}

    app.include_router(series_router)

    return app


app = create_app()

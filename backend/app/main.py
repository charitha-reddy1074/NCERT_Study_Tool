from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import get_settings
from app.core.container import BackendContainer
from app.core.logging import configure_logging
from app.services.catalog import CatalogService
from app.services.ingest import IngestService
from app.services.rag import RagService
from app.services.vectorstore import VectorStoreService


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)

    vectorstore = VectorStoreService(settings)
    catalog = CatalogService(settings)
    ingest = IngestService(settings, vectorstore)
    rag = RagService(settings, vectorstore)

    app.state.container = BackendContainer(
        settings=settings,
        vectorstore=vectorstore,
        catalog=catalog,
        ingest=ingest,
        rag=rag,
    )
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        # Dev-friendly CORS to prevent browser "Failed to fetch" across local ports.
        allow_origins=["*"],
        allow_origin_regex=settings.backend_cors_origin_regex,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router, prefix=settings.api_v1_prefix)
    return app


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)

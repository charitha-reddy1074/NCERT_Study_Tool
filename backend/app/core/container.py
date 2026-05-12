from dataclasses import dataclass

from app.core.config import Settings
from app.services.catalog import CatalogService
from app.services.ingest import IngestService
from app.services.rag import RagService
from app.services.vectorstore import VectorStoreService


@dataclass(slots=True)
class BackendContainer:
    settings: Settings
    vectorstore: VectorStoreService
    catalog: CatalogService
    ingest: IngestService
    rag: RagService

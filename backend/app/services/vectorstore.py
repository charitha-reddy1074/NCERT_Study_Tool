from __future__ import annotations

from uuid import uuid4
from threading import RLock

import chromadb
from langchain_community.vectorstores import Chroma

from app.core.config import Settings
from app.core.utils import ensure_path, normalize_key
from app.services.llm import build_embeddings


class VectorStoreService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = RLock()
        self._embeddings = build_embeddings(settings)
        self._persist_dir = ensure_path(settings.chroma_persist_dir)
        self._client = self._build_client()
        self._collection_name = f"ncert_class6_{normalize_key(self._active_embedding_name()) or 'default'}"
        self._vectorstore = self._build_store()

    def _active_embedding_name(self) -> str:
        return self.settings.google_embedding_model if self.settings.google_api_key else self.settings.embedding_model_name

    def _use_cloud(self) -> bool:
        return bool(self.settings.chroma_tenant and self.settings.chroma_database and self.settings.chroma_api_key)

    def _build_client(self):
        if not self._use_cloud():
            return None

        return chromadb.CloudClient(
            tenant=self.settings.chroma_tenant,
            database=self.settings.chroma_database,
            api_key=self.settings.chroma_api_key,
            cloud_host=self.settings.chroma_cloud_host,
            cloud_port=self.settings.chroma_cloud_port,
            enable_ssl=self.settings.chroma_cloud_enable_ssl,
        )

    def _build_store(self) -> Chroma:
        if self._client is not None:
            return Chroma(
                collection_name=self._collection_name,
                embedding_function=self._embeddings,
                client=self._client,
            )

        return Chroma(
            collection_name=self._collection_name,
            embedding_function=self._embeddings,
            persist_directory=str(self._persist_dir),
        )

    def reset(self) -> None:
        with self._lock:
            self._collection_name = f"ncert_class6_{uuid4().hex}"
            self._vectorstore = self._build_store()

    def add_documents(self, documents: list, ids: list[str] | None = None) -> None:
        with self._lock:
            if ids is None:
                self._vectorstore.add_documents(documents)
            else:
                self._vectorstore.add_documents(documents, ids=ids)

    def similarity_search_with_scores(self, query: str, *, k: int, filter: dict | None = None):
        chroma_filter = self._to_chroma_filter(filter)
        return self._vectorstore.similarity_search_with_relevance_scores(query, k=k, filter=chroma_filter)

    def _to_chroma_filter(self, raw_filter: dict | None) -> dict | None:
        if not raw_filter:
            return None

        # Chroma expects either a single-field where clause or a logical operator like $and.
        entries = [{key: value} for key, value in raw_filter.items()]
        if len(entries) == 1:
            return entries[0]
        return {"$and": entries}

    def count_documents(self) -> int:
        try:
            return int(self._vectorstore._collection.count())
        except Exception:
            return 0

    def has_documents(self, *, filter: dict | None = None) -> bool:
        """Check whether at least one document exists for a given metadata filter."""
        try:
            chroma_filter = self._to_chroma_filter(filter)
            result = self._vectorstore._collection.get(where=chroma_filter, limit=1)
            ids = result.get("ids", []) if isinstance(result, dict) else []
            return len(ids) > 0
        except Exception:
            return False

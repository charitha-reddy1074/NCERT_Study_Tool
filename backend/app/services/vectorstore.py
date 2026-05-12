from __future__ import annotations

from uuid import uuid4
from threading import RLock

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
        self._collection_name = f"ncert_class6_{normalize_key(settings.embedding_model_name) or 'default'}"
        self._vectorstore = self._build_store()

    def _build_store(self) -> Chroma:
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

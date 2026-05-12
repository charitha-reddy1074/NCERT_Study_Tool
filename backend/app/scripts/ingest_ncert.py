from __future__ import annotations

import argparse

from app.core.config import get_settings
from app.core.container import BackendContainer
from app.core.logging import configure_logging
from app.schemas import IngestRequest
from app.services.catalog import CatalogService
from app.services.ingest import IngestService
from app.services.rag import RagService
from app.services.vectorstore import VectorStoreService


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest official NCERT textbook files into Chroma")
    parser.add_argument("--source-dir", default=None, help="Directory containing official NCERT files")
    parser.add_argument("--class-num", type=int, default=6, help="Class number")
    parser.add_argument("--subject", default=None, help="Optional subject override")
    parser.add_argument("--chapter", default=None, help="Optional chapter override")
    parser.add_argument("--clear-existing", action="store_true", help="Reset the persisted vector store before ingesting")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)

    vectorstore = VectorStoreService(settings)
    container = BackendContainer(
        settings=settings,
        vectorstore=vectorstore,
        catalog=CatalogService(settings),
        ingest=IngestService(settings, vectorstore),
        rag=RagService(settings, vectorstore),
    )

    request = IngestRequest(
        source_dir=args.source_dir,
        class_num=args.class_num,
        subject=args.subject,
        chapter=args.chapter,
        clear_existing=args.clear_existing,
    )
    response = container.ingest.ingest_directory(request)
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

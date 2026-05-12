from fastapi import APIRouter, Request

from app.core.deps import get_container
from app.schemas import HealthResponse

router = APIRouter(prefix="/health")


@router.get("", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    container = get_container(request)
    return HealthResponse(
        status="ok",
        app_name=container.settings.app_name,
        class_num=container.settings.default_class_num,
        documents_indexed=container.vectorstore.count_documents(),
        ollama_base_url=container.settings.ollama_base_url,
        ollama_model=container.settings.ollama_model,
        generation_model=container.settings.gemini_model,
        embedding_model_name=container.settings.google_embedding_model if container.settings.google_api_key else container.settings.embedding_model_name,
    )

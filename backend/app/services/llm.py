from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.core.config import Settings


def build_embeddings(settings: Settings) -> OllamaEmbeddings:
    return OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model=settings.embedding_model_name,
    )


def build_chat_model(settings: Settings, model_name: str | None = None, *, num_predict: int = 512) -> ChatOllama:
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=model_name or settings.ollama_model,
        temperature=0.2,
        num_predict=num_predict,
    )

try:
    from langchain_core.embeddings import Embeddings
except Exception:  # pragma: no cover - optional dependency
    class Embeddings:  # type: ignore[no-redef]
        def embed_documents(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - interface fallback
            raise NotImplementedError

        def embed_query(self, text: str) -> list[float]:  # pragma: no cover - interface fallback
            raise NotImplementedError

try:
    from langchain_ollama import ChatOllama, OllamaEmbeddings
except Exception:  # pragma: no cover - optional dependency
    class OllamaEmbeddings:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("langchain-ollama is not installed")

    class ChatOllama:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("langchain-ollama is not installed")

from app.core.config import Settings

try:
    from google import genai
except Exception:  # pragma: no cover - optional dependency
    genai = None


class GoogleGenAIEmbeddings(Embeddings):
    def __init__(self, *, api_key: str, model: str) -> None:
        if genai is None:
            raise RuntimeError("google-genai is not installed, so Google embeddings cannot be used")
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self._client.models.embed_content(model=self._model, contents=texts)
            vectors = self._extract_vectors(response)
            if len(vectors) == len(texts):
                return vectors
        except Exception:
            pass

        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        response = self._client.models.embed_content(model=self._model, contents=text)
        vectors = self._extract_vectors(response)
        if not vectors:
            raise ValueError("Google GenAI embedding response did not include any vectors")
        return vectors[0]

    def _extract_vectors(self, response: object) -> list[list[float]]:
        embeddings = getattr(response, "embeddings", None)
        if not embeddings:
            single_embedding = getattr(response, "embedding", None)
            if single_embedding is None:
                return []
            embeddings = [single_embedding]

        vectors: list[list[float]] = []
        for embedding in embeddings:
            values = getattr(embedding, "values", None)
            if values is None and isinstance(embedding, dict):
                values = embedding.get("values") or embedding.get("embedding")
            if values is None:
                values = getattr(embedding, "embedding", None)
            if values is None:
                raise ValueError("Google GenAI embedding response format was not recognized")
            vectors.append([float(value) for value in values])
        return vectors


def build_embeddings(settings: Settings) -> Embeddings:
    if settings.google_api_key:
        return GoogleGenAIEmbeddings(api_key=settings.google_api_key, model=settings.google_embedding_model)

    return OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model=settings.embedding_model_name,
    )


def build_chat_model(settings: Settings, model_name: str | None = None, *, num_predict: int = 512):
    """Build Gemini model when API key is present, else fall back to Ollama.

    This keeps local development usable while still making Gemini the primary
    generation path in configured environments.
    """
    if settings.google_api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name or settings.gemini_model,
            temperature=0.2,
            max_output_tokens=num_predict,
            google_api_key=settings.google_api_key,
        )

    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=model_name or settings.ollama_model,
        temperature=0.2,
        num_predict=num_predict,
    )

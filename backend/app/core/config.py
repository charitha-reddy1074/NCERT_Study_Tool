from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "NCERT Class 6 Intelligent Study Assistant"
    api_v1_prefix: str = "/api/v1"
    log_level: str = "INFO"
    backend_cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3001",
            "http://localhost:3002",
            "http://127.0.0.1:3002",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    )
    backend_cors_origin_regex: str = r"https?://(localhost|127\.0\.0\.1)(:\d+)?$"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:1.5b"
    quiz_ollama_model: str = "qwen2.5:1.5b"
    math_ollama_model: str = "deepseek-r1:8b"

    embedding_model_name: str = "nomic-embed-text"
    chroma_persist_dir: Path = BASE_DIR / "storage/chroma"
    ncert_data_dir: Path = BASE_DIR / "data/ncert"

    default_class_num: int = 6
    max_retrieval_docs: int = 6
    answer_relevance_threshold: float = 0.35
    # Token-based chunking (preferred). If tokenization lib isn't available,
    # fall back to character-based `chunk_size` / `chunk_overlap` values below.
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 100

    # Character-based fallback (approximation, ~4 chars per token)
    chunk_size: int = 2048
    chunk_overlap: int = 400


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

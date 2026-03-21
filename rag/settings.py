from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    google_api_key: str = ""
    gemini_model: str = "llama3:latest"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_dir: str = "vector_storage/chroma_db"
    collection_name: str = "pdf_rag_chunks"
    top_k: int = 3
    chunk_size: int = 3000
    chunk_overlap: int = 300
    parse_strategy: str = "fast"
    use_title_chunking: bool = False

    @classmethod
    def from_env(cls, project_root: Path) -> "Settings":
        load_dotenv(project_root / ".env", override=False)
        key = (os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")).strip()

        return cls(
            google_api_key=key,
            gemini_model=os.getenv(
                "OLLAMA_MODEL",
                os.getenv("LLM_MODEL", os.getenv("GEMINI_MODEL", "llama3:latest")),
            ),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            chroma_dir=os.getenv("CHROMA_PERSIST_DIR", "vector_storage/chroma_db"),
            collection_name=os.getenv("CHROMA_COLLECTION_NAME", "pdf_rag_chunks"),
            top_k=int(os.getenv("RETRIEVAL_TOP_K", "3")),
            chunk_size=int(os.getenv("CHUNK_MAX_CHARACTERS", "3000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "300")),
            parse_strategy=os.getenv("PDF_PARSE_STRATEGY", "fast"),
            use_title_chunking=os.getenv("USE_TITLE_CHUNKING", "false").strip().lower()
            in {"1", "true", "yes", "on"},
        )

    def chroma_path(self, project_root: Path) -> Path:
        path = project_root / self.chroma_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

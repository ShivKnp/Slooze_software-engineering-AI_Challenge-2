from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class SystemSettings:
    """Minimal configuration for the PDF RAG pipeline."""

    google_api_key: str
    gemini_model: str = "models/gemini-2.5-flash"
    embedding_model: str = "models/gemini-embedding-001"
    chroma_persist_dir: str = "vector_storage/chroma_db"
    chroma_collection_name: str = "pdf_rag_chunks"
    retrieval_top_k: int = 3
    chunk_max_characters: int = 3000
    chunk_overlap: int = 300

    @classmethod
    def from_env(cls) -> "SystemSettings":
        # Load .env from current working directory when available.
        load_dotenv(override=False)

        google_api_key = os.getenv("GOOGLE_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()

        if not google_api_key:
            raise ValueError("Missing GOOGLE_API_KEY (or GEMINI_API_KEY) environment variable.")

        return cls(
            google_api_key=google_api_key,
            gemini_model=os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001"),
            chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "vector_storage/chroma_db"),
            chroma_collection_name=os.getenv("CHROMA_COLLECTION_NAME", "pdf_rag_chunks"),
            retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "3")),
            chunk_max_characters=int(os.getenv("CHUNK_MAX_CHARACTERS", "3000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "300")),
        )

    def ensure_storage_path(self, project_root: Path) -> Path:
        persist_path = project_root / self.chroma_persist_dir
        persist_path.mkdir(parents=True, exist_ok=True)
        return persist_path

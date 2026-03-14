from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from unstructured.chunking.title import chunk_by_title

from config.system_settings import SystemSettings
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class DocumentChunker:
    """Title-based chunking stage for interview-friendly RAG flow."""

    def __init__(self, settings: SystemSettings) -> None:
        self.settings = settings

    def chunk(self, raw_elements: list[Any], source_pdf: Path) -> list[dict[str, Any]]:
        logger.info("Chunking parsed elements with chunk_by_title")
        chunked_elements = chunk_by_title(
            raw_elements,
            max_characters=self.settings.chunk_max_characters,
            new_after_n_chars=max(1, self.settings.chunk_max_characters - self.settings.chunk_overlap),
            combine_text_under_n_chars=500,
        )

        chunks: list[dict[str, Any]] = []
        for element in chunked_elements:
            metadata = dict(getattr(getattr(element, "metadata", None), "to_dict", lambda: {})())
            text = str(getattr(element, "text", "") or "").strip()
            if not text:
                continue

            page_number = metadata.get("page_number")
            chunk = {
                "id": str(uuid.uuid4()),
                "source": str(source_pdf),
                "text": text,
                "title": metadata.get("section") or "Untitled",
                "page": page_number if isinstance(page_number, int) else None,
                "metadata": metadata,
                "tables": [],
                "images": [],
                "embedding_text": text,
            }
            chunks.append(chunk)

        if not chunks:
            # fallback: if title chunking returned nothing, build simple rolling text chunks
            full_text = "\n\n".join(str(getattr(e, "text", "") or "") for e in raw_elements).strip()
            if full_text:
                step = max(1, self.settings.chunk_max_characters - self.settings.chunk_overlap)
                for i in range(0, len(full_text), step):
                    part = full_text[i : i + self.settings.chunk_max_characters].strip()
                    if not part:
                        continue
                    chunks.append(
                        {
                            "id": str(uuid.uuid4()),
                            "source": str(source_pdf),
                            "text": part,
                            "title": "Chunk",
                            "page": None,
                            "metadata": {},
                            "tables": [],
                            "images": [],
                            "embedding_text": part,
                        }
                    )

        logger.info("Created %s chunks", len(chunks))
        return chunks

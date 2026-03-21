from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from unstructured.chunking.title import chunk_by_title
from unstructured.partition.pdf import partition_pdf
from unstructured_pytesseract.pytesseract import TesseractNotFoundError


class PDFPipeline:
    """Step 1-3: input PDF -> extract text -> split into chunks."""

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        parse_strategy: str = "fast",
        use_title_chunking: bool = False,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.parse_strategy = parse_strategy
        self.use_title_chunking = use_title_chunking

    def parse_pdf(self, pdf_path: Path) -> list[Any]:
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        def has_text(elements: list[Any]) -> bool:
            return any(str(getattr(e, "text", "") or "").strip() for e in elements)

        def run_partition(strategy: str) -> list[Any]:
            if strategy == "hi_res":
                return partition_pdf(
                    filename=str(pdf_path),
                    strategy="hi_res",
                    infer_table_structure=True,
                    extract_image_block_types=["Image"],
                    extract_image_block_to_payload=True,
                )
            return partition_pdf(filename=str(pdf_path), strategy=strategy, infer_table_structure=False)

        if self.parse_strategy == "fast":
            strategy_order = ["fast", "hi_res", "ocr_only"]
        elif self.parse_strategy == "hi_res":
            strategy_order = ["hi_res", "fast", "ocr_only"]
        elif self.parse_strategy == "ocr_only":
            strategy_order = ["ocr_only", "hi_res", "fast"]
        else:
            strategy_order = [self.parse_strategy, "fast", "hi_res", "ocr_only"]

        for strategy in strategy_order:
            try:
                elements = run_partition(strategy)
            except TesseractNotFoundError:
                continue
            except Exception:
                continue

            if has_text(elements):
                return elements

        return []

    def to_chunks(self, elements: list[Any], source_pdf: Path) -> list[dict]:
        if self.use_title_chunking:
            chunked = chunk_by_title(
                elements,
                max_characters=self.chunk_size,
                new_after_n_chars=max(1, self.chunk_size - self.chunk_overlap),
                combine_text_under_n_chars=500,
            )

            chunks: list[dict] = []
            for element in chunked:
                text = str(getattr(element, "text", "") or "").strip()
                if not text:
                    continue
                md = dict(getattr(getattr(element, "metadata", None), "to_dict", lambda: {})())
                page = md.get("page_number")
                chunks.append(
                    {
                        "id": str(uuid.uuid4()),
                        "source": str(source_pdf),
                        "title": md.get("section") or "Untitled",
                        "page": page if isinstance(page, int) else None,
                        "text": text,
                        "embedding_text": text,
                    }
                )
            return chunks

        # Faster default: simple rolling chunks from full extracted text.
        full_text = "\n\n".join(str(getattr(e, "text", "") or "").strip() for e in elements).strip()
        if not full_text:
            return []

        chunks: list[dict] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        for i in range(0, len(full_text), step):
            part = full_text[i : i + self.chunk_size].strip()
            if not part:
                continue
            chunks.append(
                {
                    "id": str(uuid.uuid4()),
                    "source": str(source_pdf),
                    "title": "Chunk",
                    "page": None,
                    "text": part,
                    "embedding_text": part,
                }
            )
        return chunks

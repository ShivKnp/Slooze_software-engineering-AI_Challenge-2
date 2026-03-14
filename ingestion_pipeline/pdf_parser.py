from __future__ import annotations

from pathlib import Path
from typing import Any

from unstructured.partition.pdf import partition_pdf
from unstructured_pytesseract.pytesseract import TesseractNotFoundError

from utils.logging_utils import get_logger

logger = get_logger(__name__)


class PDFParser:
    """PDF input -> Unstructured parsing -> extract plain text."""

    def parse(self, pdf_path: Path) -> list[Any]:
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info("Parsing PDF with Unstructured: %s", pdf_path)
        try:
            elements = partition_pdf(
                filename=str(pdf_path),
                strategy="hi_res",
                infer_table_structure=True,
                extract_image_block_types=["Image"],
                extract_image_block_to_payload=True,
            )
        except TesseractNotFoundError:
            logger.warning(
                "Tesseract is not installed. Falling back to fast parsing without OCR/image payload extraction."
            )
            elements = partition_pdf(
                filename=str(pdf_path),
                strategy="fast",
                infer_table_structure=False,
            )
        logger.info("Parsed %s elements", len(elements))
        return elements

    @staticmethod
    def extract_text(elements: list[Any]) -> str:
        parts: list[str] = []
        for element in elements:
            text = str(getattr(element, "text", "") or "").strip()
            if text:
                parts.append(text)
        return "\n\n".join(parts)

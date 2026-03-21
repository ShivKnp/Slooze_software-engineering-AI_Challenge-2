from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag.gemini import GeminiEmbedding, GeminiLLM
from rag.pipeline import PDFPipeline
from rag.settings import Settings
from rag.vector_store import ChromaStore


class RAGService:
    """Single orchestrator for the full basic RAG flow."""

    def __init__(self, settings: Settings, project_root: Path) -> None:
        self.settings = settings
        self.project_root = project_root

        self.pipeline = PDFPipeline(
            settings.chunk_size,
            settings.chunk_overlap,
            parse_strategy=settings.parse_strategy,
            use_title_chunking=settings.use_title_chunking,
        )
        self.embedding = GeminiEmbedding(settings.google_api_key, settings.embedding_model)
        self.llm = GeminiLLM(settings.google_api_key, settings.gemini_model)
        self.store = ChromaStore(settings.chroma_path(project_root), settings.collection_name)
        self.active_source: str | None = None

    def ingest(self, pdf_path: Path) -> int:
        source = str(pdf_path)
        self.active_source = source
        existing = self.store.all_chunks()
        if any(c.get("source") == source for c in existing):
            return -1

        elements = self.pipeline.parse_pdf(pdf_path)
        chunks = self.pipeline.to_chunks(elements, pdf_path)
        if not chunks:
            return 0
        embeddings = self.embedding.embed_documents([c["embedding_text"] for c in chunks])
        if not embeddings:
            return 0
        self.store.store_chunks(chunks, embeddings)
        return len(chunks)

    def ask(self, question: str) -> tuple[str, list[dict[str, Any]]]:
        qvec = self.embedding.embed_query(question)
        overfetch_k = max(self.settings.top_k * 8, self.settings.top_k)
        rows = self.store.retrieve(qvec, overfetch_k)
        if self.active_source:
            rows = [r for r in rows if r.get("chunk", {}).get("source") == self.active_source]
        rows = rows[: self.settings.top_k]
        if not rows:
            return "The document does not contain enough information to answer this question.", []
        context = self._context(rows)
        return self.llm.answer(question, context), rows

    def summarize_document(self) -> str:
        chunks = self.store.all_chunks()
        if self.active_source:
            chunks = [c for c in chunks if c.get("source") == self.active_source]
        if not chunks:
            return "No indexed content found for the active PDF."
        context = "\n\n".join(c.get("text", "")[:1200] for c in chunks[:8])
        return self.llm.summarize(context)

    @staticmethod
    def _context(rows: list[dict[str, Any]]) -> str:
        blocks: list[str] = []
        for i, row in enumerate(rows, start=1):
            c = row.get("chunk", {})
            blocks.append(
                f"Chunk {i} | score={row.get('score', 0.0):.4f} | page={c.get('page')} | title={c.get('title')}\n"
                f"{c.get('text', '')}"
            )
        return "\n\n---\n\n".join(blocks)

    def export_retrieved(self, rows: list[dict[str, Any]], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

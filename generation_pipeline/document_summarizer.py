from __future__ import annotations

from llm_providers.gemini_client import GeminiLLMClient


class DocumentSummarizer:
    """Creates document-level summary from indexed chunks."""

    def __init__(self, llm_client: GeminiLLMClient) -> None:
        self.llm_client = llm_client

    def summarize_document(self, chunks: list[dict]) -> str:
        if not chunks:
            return "No chunks are indexed yet."

        context_parts: list[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            page = chunk.get("page")
            pages = str(page) if page is not None else "unknown"
            context_parts.append(
                f"Chunk {idx} | Page: {pages} | Title: {chunk.get('title', 'Untitled')}\n"
                f"Text: {chunk.get('text', '')[:1200]}"
            )

        prompt = (
            "Summarize the document in this exact structure:\n"
            "1) Summary\n"
            "2) Methodology\n"
            "3) Key Findings\n\n"
            f"Context:\n{chr(10).join(context_parts[:8])}"
        )
        response = self.llm_client.generate_text(prompt)
        if response:
            return response

        preview = "\n".join(context_parts[:6])
        return (
            "1) Summary\n"
            "Document content extracted from indexed chunks.\n\n"
            "2) Methodology\n"
            "Could not generate model summary; extractive fallback shown below.\n\n"
            "3) Key Findings\n"
            f"{preview}\n\n"
        )

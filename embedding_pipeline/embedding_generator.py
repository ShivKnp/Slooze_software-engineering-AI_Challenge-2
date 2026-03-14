from __future__ import annotations

from llm_providers.gemini_client import GeminiEmbeddingClient


class EmbeddingGenerator:
    """Chunk -> embedding vectors using Gemini embedding API."""

    def __init__(self, client: GeminiEmbeddingClient) -> None:
        self.client = client

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.client.embed_documents(texts)

    def embed_query(self, query: str) -> list[float]:
        return self.client.embed_query(query)

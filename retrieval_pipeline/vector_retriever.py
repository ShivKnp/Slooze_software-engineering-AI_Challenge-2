from __future__ import annotations

from embedding_pipeline.vector_database_manager import VectorDatabaseManager


class VectorRetriever:
    """Vector similarity retrieval stage."""

    def __init__(self, vector_db: VectorDatabaseManager, top_k: int) -> None:
        self.vector_db = vector_db
        self.top_k = top_k

    def retrieve(self, query_embedding: list[float]) -> list[dict]:
        return self.vector_db.retrieve(query_embedding=query_embedding, top_k=self.top_k)

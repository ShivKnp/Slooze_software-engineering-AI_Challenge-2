from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.errors import InvalidArgumentError

from utils.json_exporter import export_json
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class VectorDatabaseManager:
    """Stores and retrieves chunk embeddings in Chroma."""

    def __init__(self, persist_dir: str, collection_name: str) -> None:
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.readable_export_path = self.persist_dir / "readable_collection.json"

    def _recreate_collection(self) -> None:
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: list[dict[str, Any]], embeddings: list[list[float]]) -> None:
        if not chunks or not embeddings:
            return

        ids = [c["id"] for c in chunks]
        docs = [c["embedding_text"] for c in chunks]
        metadatas = [{"chunk_json": json.dumps(c, ensure_ascii=False)} for c in chunks]

        try:
            self.collection.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metadatas)
        except InvalidArgumentError as exc:
            if "dimension" not in str(exc).lower():
                raise
            logger.warning("Embedding dimension changed. Recreating collection and retrying upsert.")
            self._recreate_collection()
            self.collection.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metadatas)

        self.export_readable_collection()

    def retrieve(self, query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "distances"],
        )

        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        rows: list[dict[str, Any]] = []
        for metadata, distance in zip(metadatas, distances):
            chunk_json = metadata.get("chunk_json", "{}")
            chunk = json.loads(chunk_json)
            rows.append({"score": float(distance), "chunk": chunk})
        return rows

    def get_all_chunks(self) -> list[dict[str, Any]]:
        data = self.collection.get(include=["metadatas"])
        chunks: list[dict[str, Any]] = []
        for metadata in data.get("metadatas", []):
            chunks.append(json.loads(metadata.get("chunk_json", "{}")))
        return chunks

    def export_readable_collection(self) -> None:
        data = self.collection.get(include=["documents", "metadatas", "embeddings"])

        ids = data.get("ids", [])
        docs = data.get("documents", [])
        metadatas = data.get("metadatas", [])
        embeddings = data.get("embeddings", [])

        records: list[dict[str, Any]] = []
        for idx, item_id in enumerate(ids):
            emb = embeddings[idx] if idx < len(embeddings) else []
            emb_list = list(emb) if emb is not None else []
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            chunk_json = metadata.get("chunk_json") if isinstance(metadata, dict) else None
            chunk = json.loads(chunk_json) if isinstance(chunk_json, str) else None

            records.append(
                {
                    "id": item_id,
                    "document": docs[idx] if idx < len(docs) else "",
                    "embedding_dimensions": len(emb_list),
                    "embedding_preview_first_12": emb_list[:12],
                    "chunk": chunk,
                }
            )

        export_json(
            {
                "exported_at": datetime.utcnow().isoformat() + "Z",
                "collection_name": self.collection_name,
                "records_count": len(records),
                "records": records,
            },
            self.readable_export_path,
        )

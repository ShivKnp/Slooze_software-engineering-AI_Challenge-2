from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.errors import InvalidArgumentError


class ChromaStore:
    """Step 5-6: store embeddings and retrieve relevant chunks."""

    def __init__(self, persist_dir: Path, collection_name: str) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.readable_path = persist_dir / "readable_collection.json"

    def store_chunks(self, chunks: list[dict[str, Any]], embeddings: list[list[float]]) -> None:
        if not chunks or not embeddings:
            return
        ids = [c["id"] for c in chunks]
        docs = [c["embedding_text"] for c in chunks]
        md = [{"chunk_json": json.dumps(c, ensure_ascii=False)} for c in chunks]
        try:
            self.collection.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=md)
        except InvalidArgumentError as exc:
            # If an older collection was created with a different embedding size,
            # recreate it so the current embedding model can write vectors.
            message = str(exc).lower()
            if "dimension" not in message:
                raise
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self.collection.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=md)
        self.export_readable()

    def retrieve(self, query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "distances"],
        )
        rows: list[dict[str, Any]] = []
        for metadata, distance in zip(result.get("metadatas", [[]])[0], result.get("distances", [[]])[0]):
            chunk = json.loads(metadata.get("chunk_json", "{}"))
            rows.append({"score": float(distance), "chunk": chunk})
        return rows

    def all_chunks(self) -> list[dict[str, Any]]:
        data = self.collection.get(include=["metadatas"])
        return [json.loads(m.get("chunk_json", "{}")) for m in data.get("metadatas", [])]

    def export_readable(self) -> None:
        data = self.collection.get(include=["documents", "metadatas", "embeddings"])
        records: list[dict[str, Any]] = []
        ids = data.get("ids", [])
        docs = data.get("documents", [])
        mds = data.get("metadatas", [])
        embs = data.get("embeddings", [])

        for i, item_id in enumerate(ids):
            emb = list(embs[i]) if i < len(embs) and embs[i] is not None else []
            chunk_json = mds[i].get("chunk_json") if i < len(mds) and isinstance(mds[i], dict) else None
            chunk = json.loads(chunk_json) if isinstance(chunk_json, str) else None
            records.append(
                {
                    "id": item_id,
                    "document": docs[i] if i < len(docs) else "",
                    "embedding_dimensions": len(emb),
                    "embedding_preview": emb[:8],
                    "chunk": chunk,
                }
            )

        payload = {
            "exported_at": datetime.utcnow().isoformat() + "Z",
            "collection_name": self.collection_name,
            "records_count": len(records),
            "records": records,
        }
        self.readable_path.parent.mkdir(parents=True, exist_ok=True)
        self.readable_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

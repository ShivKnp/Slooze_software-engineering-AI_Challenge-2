from __future__ import annotations

import math
import re
from dataclasses import dataclass

import google.generativeai as genai

from utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class GeminiEmbeddingClient:
    api_key: str
    model: str

    def __post_init__(self) -> None:
        genai.configure(api_key=self.api_key)

    def embed_query(self, text: str) -> list[float]:
        try:
            response = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_query",
            )
            return list(response["embedding"])
        except Exception as exc:
            logger.warning("Gemini query embedding failed, using local fallback: %s", exc)
            return self._deterministic_embedding(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        try:
            response = genai.embed_content(
                model=self.model,
                content=texts,
                task_type="retrieval_document",
            )

            embedding_field = response.get("embedding")
            if isinstance(embedding_field, list) and embedding_field:
                if isinstance(embedding_field[0], (float, int)):
                    return [list(embedding_field)]
                return [list(item) for item in embedding_field]

            embeddings_field = response.get("embeddings")
            if isinstance(embeddings_field, list):
                parsed: list[list[float]] = []
                for item in embeddings_field:
                    values = item.get("values") if isinstance(item, dict) else None
                    if isinstance(values, list):
                        parsed.append([float(v) for v in values])
                if parsed:
                    return parsed

            raise ValueError("Unexpected embed_content response format")
        except Exception as exc:
            logger.warning("Gemini batch embedding failed, using local fallback: %s", exc)
            return [self._deterministic_embedding(text) for text in texts]

    @staticmethod
    def _deterministic_embedding(text: str, dims: int = 256) -> list[float]:
        tokens = re.findall(r"[A-Za-z0-9_]+", (text or "").lower())
        vector = [0.0] * dims
        if not tokens:
            return vector

        for token in tokens:
            idx = hash(token) % dims
            vector[idx] += 1.0

        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


@dataclass
class GeminiLLMClient:
    api_key: str
    model_name: str

    def __post_init__(self) -> None:
        genai.configure(api_key=self.api_key)
        self._model = genai.GenerativeModel(self.model_name)

    def generate_text(self, prompt: str) -> str:
        try:
            response = self._model.generate_content(prompt)
            return (response.text or "").strip()
        except Exception as exc:
            logger.warning("Gemini generation failed: %s", exc)
            return ""

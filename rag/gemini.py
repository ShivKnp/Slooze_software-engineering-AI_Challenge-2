from __future__ import annotations

import json
import math
import os
import re
from urllib import error, request
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer


@dataclass
class OpenSourceEmbedding:
    api_key: str
    model: str

    def __post_init__(self) -> None:
        # Keep init resilient so retrieval still works via fallback vectors.
        self._encoder: SentenceTransformer | None = None
        try:
            self._encoder = SentenceTransformer(self.model)
        except Exception:
            self._encoder = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            if self._encoder is None:
                return [self._fallback_vector(t) for t in texts]
            arr = self._encoder.encode(texts, normalize_embeddings=True)
            return [[float(x) for x in v] for v in arr]
        except Exception:
            return [self._fallback_vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        try:
            if self._encoder is None:
                return self._fallback_vector(text)
            arr = self._encoder.encode([text], normalize_embeddings=True)
            return [float(x) for x in arr[0]]
        except Exception:
            return self._fallback_vector(text)

    @staticmethod
    def _fallback_vector(text: str, dims: int = 256) -> list[float]:
        tokens = re.findall(r"[A-Za-z0-9_]+", (text or "").lower())
        vec = [0.0] * dims
        for token in tokens:
            vec[hash(token) % dims] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


@dataclass
class OpenSourceLLM:
    api_key: str
    model: str
    max_new_tokens: int = 220
    ollama_host: str = "http://127.0.0.1:11434"

    def __post_init__(self) -> None:
        # Respect env override while keeping a safe local default.
        self.ollama_host = (os.getenv("OLLAMA_HOST", self.ollama_host) or self.ollama_host).rstrip("/")

    def answer(self, question: str, context: str) -> str:
        prompt = (
            "Answer the user's question using only the provided context. "
            "Be concise (2-5 sentences). Do not add sections, bullets, or extra commentary. "
            "If the answer is not in context, reply exactly: Not found in provided context.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{context}"
        )
        try:
            text = self._ollama_generate(prompt)
            return text or "The document does not contain enough information to answer this question."
        except Exception:
            return self._extractive_answer_fallback(question, context)

    def summarize(self, context: str) -> str:
        prompt = (
            "Create a concise summary from the provided context only. "
            "Return plain text only, one short paragraph (4-7 sentences). "
            "Do not add headings, bullet points, or extra sections.\n\n"
            f"Context:\n{context}"
        )
        try:
            text = self._ollama_generate(prompt)
            return text or "No summary available."
        except Exception:
            return self._extractive_summary_fallback(context)

    def _ollama_generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": self.max_new_tokens,
            },
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self.ollama_host}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=120) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                data = json.loads(raw)
                return str(data.get("response", "")).strip()
        except error.URLError as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

    @staticmethod
    def _extractive_summary_fallback(context: str, max_sentences: int = 4) -> str:
        pieces = [p.strip() for p in re.split(r"(?<=[.!?])\s+", context) if p.strip()]
        if not pieces:
            return "No summary available."
        return " ".join(pieces[:max_sentences])

    @staticmethod
    def _extractive_answer_fallback(question: str, context: str) -> str:
        q_tokens = set(re.findall(r"[A-Za-z0-9_]+", question.lower()))
        lines = [line.strip() for line in context.splitlines() if line.strip()]
        best_line = ""
        best_score = 0
        for line in lines:
            l_tokens = set(re.findall(r"[A-Za-z0-9_]+", line.lower()))
            score = len(q_tokens.intersection(l_tokens))
            if score > best_score:
                best_score = score
                best_line = line
        if best_score == 0:
            return "The document does not contain enough information to answer this question."
        return best_line


# Backward-compatible aliases for existing imports.
GeminiEmbedding = OpenSourceEmbedding
GeminiLLM = OpenSourceLLM

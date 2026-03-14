from __future__ import annotations

from llm_providers.gemini_client import GeminiLLMClient


class AnswerGenerator:
    """Context -> grounded answer generation."""

    def __init__(self, llm_client: GeminiLLMClient) -> None:
        self.llm_client = llm_client

    def generate_answer(self, question: str, context: str) -> str:
        prompt = (
            "You are a grounded document assistant.\n"
            "Answer ONLY from the provided context.\n"
            "If context is insufficient, reply exactly: "
            "The document does not contain enough information to answer this question.\n\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{context}\n"
        )
        answer = self.llm_client.generate_text(prompt=prompt)
        if answer:
            return answer
        return "The document does not contain enough information to answer this question."

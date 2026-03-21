from __future__ import annotations

import logging
from pathlib import Path

from rag.service import RAGService
from rag.settings import Settings


def _normalize_pdf_input(raw: str) -> Path:
    text = raw.strip()
    if text.startswith("&"):
        text = text[1:].strip()
    if (text.startswith("\"") and text.endswith("\"")) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1]
    return Path(text)


def run_cli() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    project_root = Path(__file__).resolve().parent

    try:
        settings = Settings.from_env(project_root)
    except Exception as exc:
        print(f"Configuration error: {exc}")
        return

    app = RAGService(settings=settings, project_root=project_root)

    print("Multimodal PDF RAG CLI")
    print("Type exit to quit.")

    try:
        pdf_path_input = input("Enter PDF path: ")
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye.")
        return

    pdf_path = _normalize_pdf_input(pdf_path_input)

    try:
        count = app.ingest(pdf_path)
        if count == -1:
            print("This PDF is already indexed. Reusing existing chunks.")
        elif count == 0:
            print("Ingestion finished, but no extractable content was indexed.")
            return
        else:
            print(f"Ingestion complete. Indexed chunks: {count}")
    except Exception as exc:
        logging.exception("Failed to ingest PDF")
        print(f"Ingestion failed: {exc}")
        return

    while True:
        try:
            query = input("\nAsk a question, 'summarize', or 'change pdf': ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        try:
            if query.lower() in {"change pdf", "change", "switch pdf", "switch"}:
                new_pdf_input = input("Enter new PDF path: ")
                new_pdf = _normalize_pdf_input(new_pdf_input)
                count = app.ingest(new_pdf)
                if count == -1:
                    print("This PDF is already indexed. Switched to that PDF.")
                elif count == 0:
                    print("Ingestion finished, but no extractable content was indexed.")
                else:
                    print(f"Ingestion complete. Indexed chunks: {count}")
                continue

            if query.lower() in {"summarize the document", "summarize", "summary"}:
                summary = app.summarize_document()
                print("\nDocument Summary:\n")
                print(summary)
                continue

            answer, retrieved = app.ask(query)
            print("\nAnswer:\n")
            print(answer)

            debug_path = project_root / "documents" / "retrieved_chunks_debug.json"
            app.export_retrieved(retrieved, debug_path)
            print(f"\nDebug retrieval export: {debug_path}")
        except Exception as exc:
            logging.exception("Query failed")
            print(f"Query failed: {exc}")


if __name__ == "__main__":
    run_cli()

# PDF RAG System 

Simple, professional basic RAG project with a compact structure.

## End-to-End Steps

1. Accept a PDF file as input
2. Extract text from the PDF (Unstructured->fast(text),hi_res(tables,headings,etc.)->ocr(image))
3. Split text into chunks (fast rolling chunks by default)
4. Generate embeddings for chunks (SentenceTransformers - Ollama)
5. Store embeddings in Chroma
6. Retrieve top relevant chunks for a user question
7. Send retrieved context to local Ollama model and generate grounded answer (active PDF only)

## Clean Structure

multimodal_pdf_rag_system/
  main.py
  rag/
    settings.py
    pipeline.py
    gemini.py
    vector_store.py
    service.py
  documents/
  vector_storage/chroma_db/
  requirements.txt
  README.md

## Run

1. Install dependencies
   pip install -r requirements.txt

2. Add `.env` in project root
   OLLAMA_MODEL=llama3:latest
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   RETRIEVAL_TOP_K=3
   PDF_PARSE_STRATEGY=fast
   USE_TITLE_CHUNKING=false

   # Optional backward compatibility aliases
   # LLM_MODEL=llama3:latest
   # GEMINI_MODEL=llama3:latest

3. Start CLI
   python main.py

## Notes

- Entering PDF path supports plain path, quoted path, and PowerShell-style input like:
   & 'c:\Users\name\Downloads\file.pdf'
- Use `change pdf` / `switch pdf` in the CLI to change the active PDF mid-conversation.
- Retrieval and summary are scoped to the active PDF, which prevents unrelated cross-document answers.
- Parser has automatic fallback (`fast`, `hi_res`, `ocr_only`) if initial extraction returns no text.
- `summarize`, `summary`, or `summarize the document` runs document summary.
- Retrieved chunk debug export is written to `documents/retrieved_chunks_debug.json`.
- Readable Chroma mirror is written to `vector_storage/chroma_db/readable_collection.json`.

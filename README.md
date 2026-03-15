# Multimodal PDF RAG System 

This project keeps the full RAG flow but with a minimal, interview-friendly structure.

## Architecture

PDF
-> Unstructured Parsing
-> Title-Based Chunking
-> Embeddings
-> Chroma Vector Store
-> Query Retrieval
-> Context Assembly
-> LLM Answer

## Minimal File Structure

multimodal_pdf_rag_system/
  main.py
  app.py
  config/system_settings.py
  ingestion_pipeline/
    pdf_parser.py
    document_chunker.py
  embedding_pipeline/
    embedding_generator.py
    vector_database_manager.py
  retrieval_pipeline/
    vector_retriever.py
  generation_pipeline/
    answer_generator.py
    document_summarizer.py
  llm_providers/
    gemini_client.py
  utils/
    logging_utils.py
    json_exporter.py
  vector_storage/chroma_db/
  documents/
  requirements.txt

## Features

- PDF text extraction via Unstructured
- Title-based chunking
- Embedding generation with Gemini embedding model
- Chroma vector storage and similarity retrieval
- Grounded Q&A with retrieved context
- Document summarization command
- Readable Chroma mirror at vector_storage/chroma_db/readable_collection.json

## Setup

1. Install dependencies:

   pip install -r requirements.txt

2. Create .env in project root:

   GOOGLE_API_KEY=your_key
   GEMINI_MODEL=models/gemini-2.5-flash
   EMBEDDING_MODEL=models/gemini-embedding-001
   RETRIEVAL_TOP_K=3

3. Run:

   python main.py

## CLI Usage

1. Enter PDF path.
2. Ask questions, for example:
   - Summarize the document
   - What methodology was used in the study?
3. Type exit to close.

## Example Output Style

Answer:
Summary: The document discusses the impact of AI-driven automation in enterprise workflows and evaluates productivity improvements.
Methodology: The study used case studies and experimental evaluations across multiple environments.

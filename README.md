# PDF Summarization and Question Answering Agent

This project implements a Node.js PDF RAG workflow that:

- accepts a PDF as input,
- extracts text from the document,
- splits the text into chunks,
- generates embeddings with Gemini,
- stores vectors in a local on-disk vector store,
- retrieves relevant chunks for a user question,
- produces grounded answers and document summaries.

## Project Structure

```text
agent/
  cli.js
  config.js
   local-vector-store.js
  prompts.js
  rag-agent.js
.env.example
package.json
README.md
```

## Submission Checklist

This repository includes all requested deliverables:

- `README.md` with setup, usage, architecture, dependencies, and design notes
- `agent/` source folder with the PDF RAG implementation
- `package.json` with the Node.js runtime dependencies and scripts
- Clear local run instructions for the local vector store mode

## Architecture Overview

1. **Document ingestion**  
   The CLI accepts a PDF path and loads the file with `pdf-parse`.

2. **Text extraction**  
   The agent extracts text from the PDF into a normalized plain-text representation.

3. **Chunking**  
   A chunking utility splits the extracted text into overlapping segments for retrieval.

4. **Embedding generation**  
   `@google/genai` generates Gemini embeddings for each chunk.

5. **Vector storage**  
   The Node app stores embeddings in a local JSON vector store for indexing and retrieval.

6. **Retrieval-Augmented Generation**  
   For questions, the app embeds the query, runs vector similarity search over the local vector store, and injects the top matching chunks into Gemini.

7. **Grounded generation**  
   Gemini generates either a direct answer or a hierarchical summary constrained by the document context.

## Design Decisions and Trade-offs

- **Node.js CLI-first setup**: keeps the implementation lightweight and aligned with the requirement to use Node instead of Python.
- **Local-only vector store**: removes infrastructure requirements so the app runs locally without Docker or a separate database.
- **Direct Gemini SDK usage**: avoids heavy framework abstraction and keeps the RAG flow explicit.
- **Character-based chunking**: simple and reliable for PDFs, though less semantically aware than sentence or token-based splitters.
- **Hierarchical summarization**: improves whole-document coverage compared with summarizing only the top retrieved chunks.

## Setup Instructions

### 1. Install dependencies

```powershell
npm install
```

### 2. Configure environment variables

Copy `.env.example` to `.env` and update it with your Gemini credentials:

```powershell
Copy-Item .env.example .env
```

Required variables:

- `GEMINI_API_KEY`
- `GEMINI_MODEL` (default: `gemini-2.5-flash`)

Optional tuning variables:

- `GEMINI_EMBEDDING_MODEL`
- `COLLECTION_PREFIX`
- `LOCAL_VECTOR_STORE_DIR`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `TOP_K`
- `SUMMARY_GROUP_SIZE`
- `EMBEDDING_DIMENSIONS`

### 3. Verify the setup

```powershell
npm run check
```

This validates the Node.js source files before you run the app.

## How to Run the Project

If PowerShell gives you trouble with `npm run ... -- ...`, use the direct Node command form instead:

```powershell
node agent/cli.js summarize --pdf "C:\path\to\document.pdf"
```

### Index a PDF

```powershell
npm run index -- --pdf "C:\path\to\document.pdf"
```

Or:

```powershell
npm start -- index --pdf "C:\path\to\document.pdf"
```

### Summarize a PDF

```powershell
npm run summarize -- --pdf "C:\path\to\document.pdf"
```

Or:

```powershell
npm start -- summarize --pdf "C:\path\to\document.pdf"
```

### Ask a question about the PDF

```powershell
npm run ask -- --pdf "C:\path\to\document.pdf" --question "What methodology was used in the study?"
```

Or:

```powershell
npm start -- ask --pdf "C:\path\to\document.pdf" --question "What methodology was used in the study?"
```

### Force re-indexing

```powershell
npm run ask -- --pdf "C:\path\to\document.pdf" --question "Summarize the document" --rebuild
```

## Example Output

```text
Answer:
Summary: The document discusses the impact of AI-driven automation in enterprise workflows and evaluates productivity improvements across multiple organizations.

Methodology: The study used case studies combined with experimental evaluations across three enterprise environments.
```

## Dependencies Used

- `@google/genai`
- `commander`
- `dotenv`
- `pdf-parse`

## Local Run Summary

For a reviewer running the project locally:

1. Run `npm install`
2. Copy `.env.example` to `.env` and add `GEMINI_API_KEY`
3. Run either:
   - `npm run summarize -- --pdf "C:\path\to\document.pdf"`
   - `npm run ask -- --pdf "C:\path\to\document.pdf" --question "Your question here"`

## Notes

- The Node app uses a local on-disk vector store under `storage/local-vectors`.
- Gemini rate limits are applied per project and per model; if the selected model is rate-limited, the CLI returns the provider error directly.
- Each indexed PDF maps to a stable collection name derived from the file path and file metadata.
- If the document changes, run the command again with `--rebuild` to refresh embeddings.
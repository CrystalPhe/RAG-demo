# Qdrant PDF RAG Demo

Streamlit demo for grounded question-answering over PDFs using local embeddings, Qdrant, an optional cross-encoder reranker, and OpenRouter for reasoning.

## Setup

1. Create and activate a Python environment:

```powershell
python -m venv .venv

.venv\Scripts\Activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and fill in your Qdrant URL, Qdrant API key, and OpenRouter API key.

## Run

```bash
streamlit run search.py
```

## Workflow

1. **Upload & Parse**: Upload a PDF via the sidebar. Text is extracted per page with `PyPDF2` and normalized to remove common header/footer artifacts.

2. **Chunk & Embed**: Text is split into chunks using sentence-aware and optional semantic chunking (configurable). Tiny or header/footer-like fragments are filtered and small chunks are merged into neighbors to reduce noise. Chunks are embedded with a SentenceTransformer (configured in `config.py`) and stored in Qdrant.

3. **Store**: Vectors and chunk metadata (page number, chunk index, source file) are stored in Qdrant for fast similarity search.

4. **Query & Retrieve**: Enter a question. The query is embedded and searched in Qdrant. The app supports an optional cross-encoder reranker: the reranker re-scores a selected set of retrieved chunks and the reranked results are used as grounding for the LLM.

5. **Reason**: The reranked (or vector-ranked) top chunks are passed to OpenRouter (if configured) for a grounded answer. Errors from OpenRouter fall back to displaying the retrieved context.

6. **Display**: The app shows the AI answer, the chunks used as grounding (reranked), and—below that—an informational list of the original vector-retrieved chunks for inspection.

## How it works (Technical)

**Text Extraction & Normalization**: `PyPDF2` extracts text per page and `pdf_utils.normalize_pdf_text()` removes page markers, short all-caps running headers, and normalizes paragraphs to reduce noisy fragments.

**Chunking**: The project provides sentence-aware chunking and optional semantic chunking (uses embeddings of sentences to detect topic shifts). Very small chunks are automatically merged into neighbors to avoid one-line fragments.

**Embeddings**: Uses a `SentenceTransformer` model (set by `EMBEDDING_MODEL_NAME` in `config.py`) to create dense vectors for passages and queries.

**Reranking**: An optional cross-encoder (`RERANKER_MODEL_NAME`) rescoring step improves precision by reordering a subset of retrieved chunks before grounding the LLM.

**Vector DB**: Qdrant stores vectors and payloads. `qdrant_utils.search_chunks()` returns multiple raw hits; the app separately shows reranked chunks and the original vector-ranked list for inspection.

**UI**: Streamlit UI exposes sliders and checkboxes to tune `chunk_size`, `chunk_overlap`, semantic settings, and `top_k` used for answers.

## Tech Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| Web UI | Streamlit | Interactive web app interface |
| PDF Reading | PyPDF2 | Extract text from PDF files |
| Text Splitting | LangChain (RecursiveCharacterTextSplitter) | Intelligent chunk splitting with hierarchy |
| Vectorization | sentence-transformers (SentenceTransformer) | Local dense embeddings for passages and queries |
| Vector DB | Qdrant | Store and search vectors by similarity |
| Env Config | python-dotenv | Load Qdrant credentials from `.env` |

## Future Enhancements

- Add hybrid search (keyword + semantic).
- Support multiple file formats (DOCX, TXT, etc.).
- Add a chunk preview UI so you can tune chunking before indexing.
- Caching for faster re-retrieval on the same document.

## Tuning tips

- If you see noisy one-line chunks or headers in results:
	- Increase `chunk_size` (try 1000–2000).
	- Increase `chunk_overlap` (200–500 helps keep sentences intact).
	- Lower `semantic_similarity_threshold` (e.g., 0.50–0.62) or increase `semantic_min_chunk_chars` to avoid aggressive semantic splits.
	- Disable semantic chunking to use deterministic sentence-based splitting.

## Config

- Edit `config.py` to set defaults:
	- `DEFAULT_TOP_K` — how many chunks are used to form the final context (and sent to the reranker if enabled).
	- `EMBEDDING_MODEL_NAME` — which SentenceTransformer to load locally.
	- `RERANKER_MODEL_NAME` — cross-encoder model used for reranking.

## Files of interest

- `search.py` — Streamlit app and UI logic.
- `pdf_utils.py` — text normalization and chunking logic (where the chunking heuristics live).
- `embeddings.py` — loads and runs the embedding model.
- `reranker.py` — cross-encoder reranker wrapper.
- `qdrant_utils.py` — Qdrant client helpers and storage/search wrappers.
- `openrouter.py` — wrapper that calls OpenRouter for grounded answers.

This is a basic RAG app: retrieval happens in Qdrant, then OpenRouter reasons over the retrieved chunks.
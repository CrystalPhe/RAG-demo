# Qdrant PDF RAG Demo

Simple Streamlit app for uploading a PDF, extracting text with PyPDF2, storing the chunks in Qdrant, and sending the retrieved chunks to OpenRouter for grounded AI reasoning.

## Setup

1. Create and activate a Python environment:

```bash
python -m venv .venv

.venv\Scripts\activate.bat
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

1. **Upload & Parse**: Upload a PDF file via the sidebar. The file is extracted using PyPDF2 with structured text preservation (paragraphs and line breaks are kept).

2. **Chunk & Embed**: Text is split into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`. The splitter respects document structure (paragraphs → sentences → words) before splitting. Each chunk is vectorized using a scikit-learn `HashingVectorizer`.

3. **Store**: Vectors and chunk metadata (page number, chunk index, source file) are stored in Qdrant for fast similarity search.

4. **Query & Retrieve**: Enter a question in the app. The query is vectorized the same way and searched against Qdrant. The top matches are returned with relevance scores.

5. **Reason**: The top retrieved chunks are passed to OpenRouter as grounded context. The model answers the question using only those chunks.

6. **Display**: Retrieved chunks are shown in expandable cards with page and chunk references, and the AI answer is shown above them.

## How it works (Technical)

- **Text Extraction**: PyPDF2 reads each PDF page and preserves newlines to maintain paragraph structure.
- **Recursive Splitting**: RecursiveCharacterTextSplitter tries to split on `"\n\n"` (paragraphs) first, then `"\n"` (lines), then sentences, then spaces. This keeps related content together.
- **Vectorization**: A stateless `HashingVectorizer` from scikit-learn converts text to fixed-size vectors (default 384 dimensions) without storing a vocabulary. Good for demo/local use.
- **Vector DB**: Qdrant stores vectors and metadata. Queries return points ranked by cosine similarity.
- **LLM**: OpenRouter receives the retrieved chunks and generates the final grounded answer.
- **UI**: Streamlit provides a simple web interface for upload, sliders for chunk tuning, and expandable results.

## Tech Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| Web UI | Streamlit | Interactive web app interface |
| PDF Reading | PyPDF2 | Extract text from PDF files |
| Text Splitting | LangChain (RecursiveCharacterTextSplitter) | Intelligent chunk splitting with hierarchy |
| Vectorization | scikit-learn (HashingVectorizer) | Convert text to vectors locally |
| Vector DB | Qdrant | Store and search vectors by similarity |
| Env Config | python-dotenv | Load Qdrant credentials from `.env` |

## Future Enhancements

- Add hybrid search (keyword + semantic).
- Support multiple file formats (DOCX, TXT, etc.).
- Caching for faster re-retrieval on the same document.

This is now a basic RAG app: retrieval happens in Qdrant, then OpenRouter reasons over the retrieved chunks.
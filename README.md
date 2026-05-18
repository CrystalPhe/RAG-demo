# Qdrant PDF RAG Demo

Simple Streamlit app for uploading a PDF, extracting text with PyPDF2, storing the chunks in Qdrant, and showing the most relevant chunks for a question.

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

3. Copy `.env.example` to `.env` and fill in your Qdrant URL and API key.

## Run

```bash
streamlit run search.py
```

## How it works

1. Upload a PDF in the sidebar.
2. The file is saved locally in `uploads/`.
3. Text is extracted with PyPDF2 and split into overlapping chunks.
4. Ask a question and the app retrieves the most related chunks from Qdrant.

This is a simple retrieval demo of RAG. It does not call an LLM yet.
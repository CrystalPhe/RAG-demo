from __future__ import annotations

import json
from typing import Iterable
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

from config import (
    APP_TITLE,
    DEFAULT_TOP_K,
    CONTEXT_MAX_CHUNKS,
    CONTEXT_MIN_SCORE,
    QDRANT_URL,
    QDRANT_API_KEY,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
)

from pdf_utils import (
    Chunk,
    ensure_upload_dir,
    save_uploaded_pdf,
    get_document_id,
    extract_pdf_text,
    build_chunks,
)

from embeddings import load_embedding_model, vectorize_text
from qdrant_utils import get_qdrant_client, ensure_collection, upsert_chunks, search_chunks
from openrouter import ask_openrouter


def init_app() -> None:
    load_embedding_model()


# Chuẩn bị context từ các chunk đã lấy được.
def format_retrieved_context(retrieved, max_chunks: int = CONTEXT_MAX_CHUNKS, min_score: float = CONTEXT_MIN_SCORE) -> str:
    contexts: list[str] = []
    ranked_points = sorted(retrieved, key=lambda point: getattr(point, "score", 0.0), reverse=True)

    for point in ranked_points:
        score = float(getattr(point, "score", 0.0) or 0.0)

        payload = point.payload or {}
        page_number = payload.get("page_number", "?")
        chunk_index = payload.get("chunk_index", "?")
        text = str(payload.get("text", "")).strip()
        if text:
            contexts.append(f"Page {page_number} | chunk {chunk_index} | score {score:.3f}\n{text}")
        if len(contexts) >= max_chunks:
            break

    return "\n\n".join(contexts)


# Gọi OpenRouter để tạo câu trả lời dựa trên context từ Qdrant.
# The OpenRouter request implementation was moved to openrouter.py (ask_openrouter)


# Xóa trạng thái file cũ để người dùng có thể upload và index file mới.
def reset_index() -> None:
    st.session_state.pop("source_name", None)
    st.session_state.pop("page_count", None)
    st.session_state.pop("index_signature", None)
    st.session_state.pop("document_id", None)
    st.session_state.pop("indexed_points", None)


# Streamlit main func
def main() -> None:
    init_app()
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.write("Upload a PDF, split it into chunks, store them in Qdrant, and ask a question to retrieve related text.")

    with st.sidebar:
        st.header("Connection")
        if QDRANT_URL and QDRANT_API_KEY:
            st.success("Qdrant config loaded from .env")
        else:
            st.error("Set QDRANT_URL and QDRANT_API_KEY in .env")

        if OPENROUTER_API_KEY:
            st.success(f"OpenRouter loaded: {OPENROUTER_MODEL}")
        else:
            st.warning("Set OPENROUTER_API_KEY to enable AI reasoning")

        st.header("PDF Settings")
        uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
        chunk_size = st.slider("Chunk size (characters)", min_value=200, max_value=4000, value=1000, step=100)
        chunk_overlap = st.slider("Chunk overlap (characters)", min_value=0, max_value=1000, value=200, step=50)
        top_k = st.slider("Top K retrieved chunks", min_value=1, max_value=10, value=DEFAULT_TOP_K, step=1)

        if st.button("Clear current document"):
            reset_index()
            st.rerun()

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        document_id = get_document_id(file_bytes)
        current_signature = (document_id, chunk_size, chunk_overlap)
        cached_signature = st.session_state.get("index_signature")

        if cached_signature != current_signature:
            try:
                saved_path = save_uploaded_pdf(uploaded_file.name, file_bytes)
                pages = extract_pdf_text(saved_path)
                chunks = build_chunks(pages, chunk_size, chunk_overlap, saved_path.name, document_id)

                client = get_qdrant_client()
                ensure_collection(client)
                upsert_chunks(client, chunks, vectorize_text)

                st.session_state["source_name"] = saved_path.name
                st.session_state["page_count"] = len(pages)
                st.session_state["index_signature"] = current_signature
                st.session_state["document_id"] = document_id
                st.session_state["indexed_points"] = len(chunks)
            except Exception as exc:
                st.error(f"Failed to index the PDF in Qdrant: {exc}")

    document_id = st.session_state.get("document_id")

    if document_id:
        st.success(
            f"Indexed {st.session_state.get('indexed_points', 0)} chunks from {st.session_state.get('source_name')} across {st.session_state.get('page_count', 0)} text pages."
        )

        question = st.text_input("Ask a question about the PDF")
        if question:
            try:
                client = get_qdrant_client()
                retrieved = search_chunks(client, vectorize_text(question), document_id, top_k)
            except Exception as exc:
                st.error(f"Qdrant search failed: {exc}")
                retrieved = []

            st.subheader("AI answer")
            try:
                st.write(ask_openrouter(question, retrieved))
            except Exception as exc:
                st.error(f"OpenRouter request failed: {exc}")
                st.write("Here is the retrieved context instead:")
                st.write(format_retrieved_context(retrieved) or "No context retrieved.")

            st.subheader("Retrieved context")
            if retrieved:
                for rank, point in enumerate(retrieved, start=1):
                    payload = point.payload or {}
                    page_number = payload.get("page_number", "?")
                    chunk_index = payload.get("chunk_index", "?")
                    score = getattr(point, "score", 0.0)
                    with st.expander(f"Match {rank} | page {page_number} | chunk {chunk_index} | score {score:.3f}"):
                        st.write(payload.get("text", ""))
            else:
                st.info("No relevant chunk found. Try changing the question or increasing chunk size.")
    else:
        st.info("Upload a PDF to build the Qdrant index.")

    with st.expander("How this demo maps to RAG"):
        st.markdown(
            """
            1. The PDF is uploaded and saved locally.
            2. Text is extracted with PyPDF2.
            3. The text is chunked into small overlapping pieces.
            4. Each chunk is embedded locally with a hashing vectorizer.
            5. The vectors are stored in Qdrant with document metadata.
            6. A question is embedded and searched against the same Qdrant collection.
            7. The top chunks are sent to OpenRouter for grounded reasoning.
            8. The assistant answer is shown alongside the retrieved context.
            """
        )


if __name__ == "__main__":
    main()
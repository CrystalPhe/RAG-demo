from __future__ import annotations

import hashlib
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from sklearn.feature_extraction.text import HashingVectorizer


load_dotenv()


APP_TITLE = "Qdrant PDF RAG Demo"
UPLOAD_DIR = Path("uploads")
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 3
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "384"))
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "pdf_demo_chunks")

HASHING_VECTORIZER = HashingVectorizer(
    n_features=VECTOR_SIZE,
    alternate_sign=False,
    norm="l2",
    lowercase=True,
    stop_words="english",
)


@dataclass(frozen=True)
class Chunk:
    text: str
    page_number: int
    chunk_index: int
    source: str
    document_id: str


def ensure_upload_dir() -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOAD_DIR


def save_uploaded_pdf(file_name: str, file_bytes: bytes) -> Path:
    upload_dir = ensure_upload_dir()
    safe_name = Path(file_name).name
    file_path = upload_dir / safe_name
    file_path.write_bytes(file_bytes)
    return file_path


def get_document_id(file_bytes: bytes) -> str:
    return hashlib.sha1(file_bytes).hexdigest()


def extract_pdf_text(file_path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(file_path))
    pages: list[tuple[int, str]] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"[ \t\f\v]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        normalized = text
        if normalized:
            pages.append((index, normalized))
    return pages


def split_text_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
    )
    return splitter.split_text(text)


def build_chunks(pages: Iterable[tuple[int, str]], chunk_size: int, overlap: int, source_name: str, document_id: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page_number, text in pages:
        for chunk_index, chunk_text in enumerate(split_text_into_chunks(text, chunk_size, overlap), start=1):
            chunks.append(
                Chunk(
                    text=chunk_text,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    source=source_name,
                    document_id=document_id,
                )
            )
    return chunks


def get_qdrant_client() -> QdrantClient:
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("Set QDRANT_URL and QDRANT_API_KEY in .env")
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def ensure_collection(client: QdrantClient) -> None:
    if not client.collection_exists(QDRANT_COLLECTION):
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )

    # Create payload index for filtering
    client.create_payload_index(
        collection_name=QDRANT_COLLECTION,
        field_name="document_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )


def vectorize_text(text: str) -> list[float]:
    vector = HASHING_VECTORIZER.transform([text]).toarray()[0]
    return vector.astype(float).tolist()


def upsert_chunks(client: QdrantClient, chunks: list[Chunk]) -> None:
    if not chunks:
        return

    document_id = chunks[0].document_id
    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=models.Filter(
            must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
        ),
    )

    points = []
    for chunk in chunks:
        point_id = uuid.uuid5(uuid.NAMESPACE_URL, f"{chunk.document_id}:{chunk.page_number}:{chunk.chunk_index}")
        points.append(
            models.PointStruct(
                id=point_id,
                vector=vectorize_text(chunk.text),
                payload={
                    "text": chunk.text,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "source": chunk.source,
                    "document_id": chunk.document_id,
                },
            )
        )

    client.upsert(collection_name=QDRANT_COLLECTION, points=points)


def search_chunks(client: QdrantClient, query: str, document_id: str, top_k: int):
    return client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=vectorize_text(query),
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id)
                )
            ]
        ),
        limit=top_k,
        with_payload=True,
    ).points


def make_demo_answer(query: str, retrieved) -> str:
    if not retrieved:
        return "I could not find a strong match in the uploaded PDF. Try a different wording or upload a longer document."

    contexts: list[str] = []
    for rank, point in enumerate(retrieved[:3], start=1):
        payload = point.payload or {}
        page_number = payload.get("page_number", "?")
        chunk_index = payload.get("chunk_index", "?")
        text = str(payload.get("text", "")).strip()
        if text:
            contexts.append(f"Match {rank} | page {page_number} | chunk {chunk_index}\n{text}")

    answer = "\n\n".join(contexts)
    return (
        "This is a Qdrant-backed retrieval demo. The most relevant context I found is:\n\n"
        f"{answer}\n\n"
        f"Question: {query}"
    )


def reset_index() -> None:
    st.session_state.pop("source_name", None)
    st.session_state.pop("page_count", None)
    st.session_state.pop("index_signature", None)
    st.session_state.pop("document_id", None)
    st.session_state.pop("indexed_points", None)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.write("Upload a PDF, split it into chunks, store them in Qdrant, and ask a question to retrieve related text.")

    with st.sidebar:
        st.header("Connection")
        if QDRANT_URL and QDRANT_API_KEY:
            st.success("Qdrant config loaded from .env")
        else:
            st.error("Set QDRANT_URL and QDRANT_API_KEY in .env")

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
                upsert_chunks(client, chunks)

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
                retrieved = search_chunks(client, question, document_id, top_k)
            except Exception as exc:
                st.error(f"Qdrant search failed: {exc}")
                retrieved = []

            st.subheader("Demo answer")
            st.write(make_demo_answer(question, retrieved))

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
            7. The top chunks are shown as retrieved context.

            In a full RAG app, those chunks would then be sent to an LLM to generate a grounded answer.
            """
        )


if __name__ == "__main__":
    main()
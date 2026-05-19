from __future__ import annotations

# Python libraries
import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# env, UI, PDF parsing
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader

# Chunking, lấy vector Qdrant.
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models

# Semantic embeddings for RAG retrieval
from sentence_transformers import SentenceTransformer


load_dotenv()


APP_TITLE = "Qdrant PDF RAG Demo"
UPLOAD_DIR = Path("uploads")
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 3
DEFAULT_CONTEXT_MAX_CHUNKS = 5
DEFAULT_CONTEXT_MIN_SCORE = 0.0
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 output dimension
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "pdf_demo_chunks")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "")
OPENROUTER_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE", APP_TITLE)
CONTEXT_MAX_CHUNKS = int(os.getenv("CONTEXT_MAX_CHUNKS", str(DEFAULT_CONTEXT_MAX_CHUNKS)))
CONTEXT_MIN_SCORE = float(os.getenv("CONTEXT_MIN_SCORE", str(DEFAULT_CONTEXT_MIN_SCORE)))

# Load semantic embedding model once at startup
@st.cache_resource
def load_embedding_model() -> SentenceTransformer:
    with st.spinner("Loading embedding model (first time only, ~2 min)..."):
        return SentenceTransformer("all-MiniLM-L6-v2")


# Pre-load model at app startup
def init_app() -> None:
    load_embedding_model()


@dataclass(frozen=True)
class Chunk:
    text: str
    page_number: int
    chunk_index: int
    source: str
    document_id: str


# Tạo folder upload
def ensure_upload_dir() -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOAD_DIR


# Save file PDF upload vào máy
def save_uploaded_pdf(file_name: str, file_bytes: bytes) -> Path:
    upload_dir = ensure_upload_dir()
    safe_name = Path(file_name).name
    file_path = upload_dir / safe_name
    file_path.write_bytes(file_bytes)
    return file_path


# Tạo mã định danh cho tài liệu dựa trên nội dung của nó.
def get_document_id(file_bytes: bytes) -> str:
    return hashlib.sha1(file_bytes).hexdigest()


# Lấy text từng trang từ PDF, giữ xuống dòng và khoảng trắng.
def extract_pdf_text(file_path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(file_path))
    pages: list[tuple[int, str]] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        # Giữ lại xuống dòng để splitter tách theo đoạn và câu tốt hơn.
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"[ \t\f\v]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        normalized = text
        if normalized:
            pages.append((index, normalized))
    return pages


# Tách text theo cấu trúc từ lớn đến nhỏ
def split_text_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    # Tách theo thứ tự: đoạn văn -> dòng -> câu -> từ.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
    )
    return splitter.split_text(text)


# Tạo danh sách các chunk với metadata để lưu vào Qdrant.
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


# Client kết nối Qdrant, dùng để tạo collection, upsert và tìm kiếm.
def get_qdrant_client() -> QdrantClient:
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("Set QDRANT_URL and QDRANT_API_KEY in .env")
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


# Đảm bảo collection tồn tại trong Qdrant, nếu chưa có thì tạo mới với cấu hình vector.
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


# Chuyển text thành vector sử dụng semantic embedding model
def vectorize_text(text: str) -> list[float]:
    model = load_embedding_model()
    vector = model.encode(text, convert_to_numpy=True)
    return vector.astype(float).tolist()


# Ghi chunk vào Qdrant, xóa chunk cũ nếu trùng lặp.
def upsert_chunks(client: QdrantClient, chunks: list[Chunk]) -> None:
    if not chunks:
        return

    document_id = chunks[0].document_id
    # Xóa chunk cũ của cùng tài liệu trước khi ghi lại.
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


# Tìm chunk gần nhất theo user query
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
def ask_openrouter(question: str, retrieved) -> str:
    if not OPENROUTER_API_KEY:
        return "Set OPENROUTER_API_KEY to generate an AI answer from the retrieved Qdrant context."

    context = format_retrieved_context(retrieved)
    if not context:
        return "I could not find a strong match in the uploaded PDF, so there is not enough context to reason over."

    messages = [
        {
            "role": "system",
            "content": (
                "You are a grounded PDF question-answering assistant. "
                "Answer only from the context provided from Qdrant. "
                "Do not use outside knowledge. "
                "If the context is insufficient, say exactly what is missing and that you cannot confirm the answer. "
                "Be concise and cite page/chunk numbers inline when you use them. "
                "Reply in the same language as the user question."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Retrieved context from Qdrant:\n{context}"
            ),
        },
    ]

    payload = json.dumps(
        {
            "model": OPENROUTER_MODEL,
            "messages": messages,
            "temperature": 0.2,
        }
    ).encode("utf-8")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    if OPENROUTER_HTTP_REFERER:
        headers["HTTP-Referer"] = OPENROUTER_HTTP_REFERER
    if OPENROUTER_APP_TITLE:
        headers["X-Title"] = OPENROUTER_APP_TITLE

    request = Request(OPENROUTER_BASE_URL, data=payload, headers=headers, method="POST")

    try:
        with urlopen(request, timeout=60) as response:
            response_text = response.read().decode("utf-8")
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
        raise RuntimeError(f"OpenRouter request failed: {exc.code} {details}") from exc
    except URLError as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc.reason}") from exc

    data = json.loads(response_text)
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("OpenRouter returned no choices.")

    message = choices[0].get("message") or {}
    content = str(message.get("content", "")).strip()
    if not content:
        raise RuntimeError("OpenRouter returned an empty response.")
    return content


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
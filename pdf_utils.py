from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import UPLOAD_DIR


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


def normalize_pdf_text(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"-\n(?=\w)", "", text)
    text = re.sub(r"[ \t\f\v]+", " ", text)

    paragraphs: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        lines = [re.sub(r"\s+", " ", line).strip() for line in paragraph.split("\n")]
        lines = [line for line in lines if line]
        if lines:
            paragraphs.append(" ".join(lines))

    if paragraphs:
        short_paragraphs = sum(1 for paragraph in paragraphs if len(paragraph.split()) <= 3)
        if short_paragraphs / len(paragraphs) >= 0.6:
            return " ".join(paragraphs).strip()

    return "\n\n".join(paragraphs).strip()


def extract_pdf_text(file_path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(file_path))
    pages: list[tuple[int, str]] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        normalized = normalize_pdf_text(text)
        if normalized:
            pages.append((index, normalized))
    return pages


def split_text_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    sentence_splitter = re.compile(r"(?<=[.!?])\s+")
    word_fallback = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["; ", ", ", " ", ""],
    )

    sentences = [segment.strip() for segment in sentence_splitter.split(text) if segment.strip()]
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush_current() -> None:
        nonlocal current, current_len
        if not current:
            return

        chunk_text = " ".join(current).strip()
        if chunk_text:
            chunks.append(chunk_text)

        if overlap <= 0:
            current = []
            current_len = 0
            return

        overlap_sentences: list[str] = []
        overlap_len = 0
        for sentence in reversed(current):
            addition = len(sentence) + (1 if overlap_sentences else 0)
            if overlap_len + addition > overlap:
                break
            overlap_sentences.insert(0, sentence)
            overlap_len += addition

        current = overlap_sentences
        current_len = len(" ".join(current)) if current else 0

    for sentence in sentences:
        if len(sentence) > chunk_size:
            flush_current()
            chunks.extend(word_fallback.split_text(sentence))
            continue

        addition = len(sentence) + (1 if current else 0)
        if current and current_len + addition > chunk_size:
            flush_current()

        current.append(sentence)
        current_len = current_len + addition if current_len else len(sentence)

    flush_current()
    return chunks


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def split_text_semantic(
    text: str,
    chunk_size: int,
    overlap: int,
    embed_many_fn,
    similarity_threshold: float = 0.62,
    min_chunk_chars: int = 250,
) -> list[str]:
    sentence_splitter = re.compile(r"(?<=[.!?])\s+")
    sentences = [segment.strip() for segment in sentence_splitter.split(text) if segment.strip()]
    if not sentences:
        return []

    if len(sentences) < 3 or embed_many_fn is None:
        return split_text_into_chunks(text, chunk_size, overlap)

    vectors = embed_many_fn(sentences)
    if len(vectors) != len(sentences):
        return split_text_into_chunks(text, chunk_size, overlap)

    word_fallback = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["; ", ", ", " ", ""],
    )

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush_current() -> None:
        nonlocal current, current_len
        if not current:
            return

        chunk_text = " ".join(current).strip()
        if chunk_text:
            chunks.append(chunk_text)

        if overlap <= 0:
            current = []
            current_len = 0
            return

        overlap_sentences: list[str] = []
        overlap_len = 0
        for sentence in reversed(current):
            addition = len(sentence) + (1 if overlap_sentences else 0)
            if overlap_len + addition > overlap:
                break
            overlap_sentences.insert(0, sentence)
            overlap_len += addition

        current = overlap_sentences
        current_len = len(" ".join(current)) if current else 0

    for idx, sentence in enumerate(sentences):
        sentence_len = len(sentence)
        if sentence_len > chunk_size:
            flush_current()
            chunks.extend(word_fallback.split_text(sentence))
            continue

        addition = sentence_len + (1 if current else 0)
        would_overflow = current and (current_len + addition > chunk_size)

        topic_shift = False
        if idx > 0 and current_len >= min_chunk_chars:
            similarity = _cosine_similarity(vectors[idx - 1], vectors[idx])
            topic_shift = similarity < similarity_threshold

        if would_overflow or topic_shift:
            flush_current()

        current.append(sentence)
        current_len = current_len + addition if current_len else sentence_len

    flush_current()
    return chunks


def build_chunks(
    pages: Iterable[tuple[int, str]],
    chunk_size: int,
    overlap: int,
    source_name: str,
    document_id: str,
    use_semantic_chunking: bool = False,
    semantic_similarity_threshold: float = 0.62,
    semantic_min_chunk_chars: int = 250,
    embed_many_fn=None,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page_number, text in pages:
        if use_semantic_chunking:
            page_chunks = split_text_semantic(
                text,
                chunk_size,
                overlap,
                embed_many_fn=embed_many_fn,
                similarity_threshold=semantic_similarity_threshold,
                min_chunk_chars=semantic_min_chunk_chars,
            )
        else:
            page_chunks = split_text_into_chunks(text, chunk_size, overlap)

        for chunk_index, chunk_text in enumerate(page_chunks, start=1):
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

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

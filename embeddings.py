from sentence_transformers import SentenceTransformer
import streamlit as st

from config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_NORMALIZE,
    EMBEDDING_PASSAGE_PREFIX,
    EMBEDDING_QUERY_PREFIX,
)


@st.cache_resource
def load_embedding_model() -> SentenceTransformer:
    with st.spinner("\nLoading embedding model (first time only, ~2 min)..."):
        return SentenceTransformer(EMBEDDING_MODEL_NAME)


def vectorize_text(text: str) -> list[float]:
    return vectorize_passage(text)


def vectorize_query(text: str) -> list[float]:
    model = load_embedding_model()
    vector = model.encode(
        f"{EMBEDDING_QUERY_PREFIX}{text}",
        convert_to_numpy=True,
        normalize_embeddings=EMBEDDING_NORMALIZE,
    )
    return vector.astype(float).tolist()


def vectorize_passage(text: str) -> list[float]:
    model = load_embedding_model()
    vector = model.encode(
        f"{EMBEDDING_PASSAGE_PREFIX}{text}",
        convert_to_numpy=True,
        normalize_embeddings=EMBEDDING_NORMALIZE,
    )
    return vector.astype(float).tolist()


def vectorize_passages(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    model = load_embedding_model()
    prefixed = [f"{EMBEDDING_PASSAGE_PREFIX}{text}" for text in texts]
    vectors = model.encode(
        prefixed,
        convert_to_numpy=True,
        normalize_embeddings=EMBEDDING_NORMALIZE,
    )
    return vectors.astype(float).tolist()

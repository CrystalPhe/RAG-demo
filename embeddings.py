from sentence_transformers import SentenceTransformer
import streamlit as st


@st.cache_resource
def load_embedding_model() -> SentenceTransformer:
    with st.spinner("Loading embedding model (first time only, ~2 min)..."):
        return SentenceTransformer("all-MiniLM-L6-v2")


def vectorize_text(text: str) -> list[float]:
    model = load_embedding_model()
    vector = model.encode(text, convert_to_numpy=True)
    return vector.astype(float).tolist()

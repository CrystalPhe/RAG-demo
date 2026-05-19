from dataclasses import dataclass

import streamlit as st
from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL_NAME


@dataclass(frozen=True)
class RankedPoint:
    payload: dict
    score: float
    base_score: float


@st.cache_resource
def load_reranker() -> CrossEncoder:
    with st.spinner("Loading reranker model (first time only)..."):
        return CrossEncoder(RERANKER_MODEL_NAME)


def rerank_retrieved(question: str, retrieved, top_k: int):
    if not retrieved:
        return []

    model = load_reranker()
    candidates: list[tuple[int, str]] = []
    for idx, point in enumerate(retrieved):
        payload = point.payload or {}
        text = str(payload.get("text", "")).strip()
        if text:
            candidates.append((idx, text))

    if not candidates:
        return list(retrieved)[:top_k]

    pairs = [[question, text] for _, text in candidates]
    scores = model.predict(pairs)

    ranked = []
    for (idx, _), rerank_score in zip(candidates, scores):
        point = retrieved[idx]
        payload = point.payload or {}
        base_score = float(getattr(point, "score", 0.0) or 0.0)
        ranked.append(
            RankedPoint(
                payload=payload,
                score=float(rerank_score),
                base_score=base_score,
            )
        )

    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked[:top_k]

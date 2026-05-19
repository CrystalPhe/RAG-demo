from typing import Iterable
from uuid import uuid5, NAMESPACE_URL

from qdrant_client import QdrantClient, models

from config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION, VECTOR_SIZE
from pdf_utils import Chunk


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

    client.create_payload_index(
        collection_name=QDRANT_COLLECTION,
        field_name="document_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )


def upsert_chunks(client: QdrantClient, chunks: list[Chunk], vectorize_fn) -> None:
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
        point_id = uuid5(NAMESPACE_URL, f"{chunk.document_id}:{chunk.page_number}:{chunk.chunk_index}")
        points.append(
            models.PointStruct(
                id=point_id,
                vector=vectorize_fn(chunk.text),
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


def search_chunks(client: QdrantClient, query_vector, document_id: str, top_k: int):
    return client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id)
                )
            ]
        ),
        limit=max(top_k * 4, top_k),
        with_payload=True,
    ).points

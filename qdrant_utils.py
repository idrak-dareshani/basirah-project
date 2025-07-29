import os
import uuid
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, VectorParams

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "tafsir"

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

def setup_collection(vector_size=512):
    collections = client.get_collections().collections
    if not any(c.name == COLLECTION_NAME for c in collections):
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

def add_tafsir_doc(embedding, payload: dict):
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[PointStruct(id=str(uuid.uuid4()), vector=embedding, payload=payload)]
    )

def search_tafsir(query_embedding, top_k=3, author=None, surah=None):
    filter_ = {}
    if author:
        filter_["author"] = author
    if surah:
        filter_["surah"] = surah

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k,
        query_filter={"must": [{"key": k, "match": {"value": v}} for k, v in filter_.items()]} if filter_ else None
    )
    return results

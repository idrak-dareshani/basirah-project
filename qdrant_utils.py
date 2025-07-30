import os
import uuid
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
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
    filter_conditions = []
    if author:
        filter_conditions.append({"key": "author", "match": {"value": author}})
    if surah:
        filter_conditions.append({"key": "surah", "match": {"value": surah}})

    search_params = {
        "collection_name": COLLECTION_NAME,
        "query_vector": query_embedding,
        "limit": top_k
    }
    
    # Try query_filter instead of filter
    if filter_conditions:
        search_params["query_filter"] = {"must": filter_conditions}
    
    results = client.search(**search_params)
    return results

def search_text(author, surah, ayah):
    filter_conditions = []

    filter_conditions.append({"key": "author", "match": {"value": author}})
    filter_conditions.append({"key": "surah", "match": {"value": surah}})
    filter_conditions.append({"key": "ayah_range", "range": {"gte": ayah, "lte": ayah}})
    search_params = {
        "collection_name": COLLECTION_NAME,
        "scroll_filter": {"must": filter_conditions}
    }
    
    #results, next_offset = client.scroll(**search_params)
    results, next_offset = client.scroll(collection_name=COLLECTION_NAME,
                                         scroll_filter=models.Filter(
                                             must=[
                                                 models.FieldCondition(
                                                     key="author",
                                                     match=models.MatchValue(value=author)
                                                 ),
                                                 models.FieldCondition(
                                                     key="surah",
                                                     match=models.MatchValue(value=surah)
                                                 ),
                                                 models.FieldCondition(
                                                     key="ayah_range",
                                                     range=models.Range(gte=ayah, lte=ayah)
                                                 )
                                             ]
                                         ))
    
    tafsir_text = None
    for result in results:
        if 'tafsir_text' in result.payload:
            tafsir_text = result.payload["tafsir_text"]
    return tafsir_text

#search_text('qurtubi', '4', '5')
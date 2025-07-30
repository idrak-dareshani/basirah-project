from sentence_transformers import SentenceTransformer

model = SentenceTransformer("distiluse-base-multilingual-cased-v2")

def get_embedding(text: str):
    return model.encode(text).tolist()

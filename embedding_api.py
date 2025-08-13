import os
#from dotenv import load_dotenv
from huggingface_hub import InferenceClient

#load_dotenv()

def get_embedding(text: str):
    client = InferenceClient(
        provider="hf-inference",
        api_key=os.getenv("HF_API_KEY"),
    )

    result = client.feature_extraction(
        text=text,
        model="sentence-transformers/distiluse-base-multilingual-cased-v2",
    )
    return result

#get_embedding('fasting')
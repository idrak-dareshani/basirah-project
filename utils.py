import os
import json
import dotenv
from openai import OpenAI

dotenv.load_dotenv()

TRANSLATION_DIR = "translation"
REFLECTION_DIR = "reflection"

def load_cached_translation(author, surah, ayah, lang_code):
    path = os.path.join(TRANSLATION_DIR, author, f"{surah}_{ayah}_{lang_code}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("translated_text")
    return None

def save_translation_to_cache(author, surah, ayah, lang_code, translated_text):
    os.makedirs(os.path.join(TRANSLATION_DIR, author), exist_ok=True)
    path = os.path.join(TRANSLATION_DIR, author, f"{surah}_{ayah}_{lang_code}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"translated_text": translated_text}, f, ensure_ascii=False, indent=2)

def load_cached_reflection(author, surah, from_ayah, to_ayah, lang):
    fname = f"{surah}_{from_ayah}_{to_ayah}_{lang}.json"
    fpath = os.path.join(REFLECTION_DIR, author, fname)
    if os.path.exists(fpath):
        with open(fpath, "r", encoding="utf-8") as f:
            return json.load(f).get("reflection")
    return None

def save_reflection_to_cache(author, surah, from_ayah, to_ayah, lang, reflection):
    os.makedirs(os.path.join(REFLECTION_DIR, author), exist_ok=True)
    fname = f"{surah}_{from_ayah}_{to_ayah}_{lang}.json"
    fpath = os.path.join(REFLECTION_DIR, author, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump({"reflection": reflection}, f, ensure_ascii=False, indent=2)

def generate_reflection_gpt(tafsir_text: str, lang: str = "en") -> str:
    client = OpenAI()

    prompt = (
        f"Based on the following Arabic Islamic tafsir text, write a spiritual reflection "
        f"in {lang} that helps the reader draw practical wisdom and guidance. Focus on character, morality, or life purpose.\n\n"
        f"{tafsir_text}"
    )

    response = client.responses.create(
        model="gpt-4o",
        input=prompt,
        temperature=0.6
    )
    return response.output_text.strip()

def get_embedding(text):
    api_key = os.getenv("HF_API_KEY")
    url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/distiluse-base-multilingual-cased-v1"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": text}
    
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        raise RuntimeError(f"HuggingFace API error: {response.status_code} - {response.text}")

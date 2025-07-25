import os
import json

CACHE_DIR = "cache"

def load_cached_translation(author, surah, ayah, lang_code):
    path = os.path.join(CACHE_DIR, author, f"{surah}_{ayah}_{lang_code}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("translated_text")
    return None

def save_translation_to_cache(author, surah, ayah, lang_code, translated_text):
    os.makedirs(os.path.join(CACHE_DIR, author), exist_ok=True)
    path = os.path.join(CACHE_DIR, author, f"{surah}_{ayah}_{lang_code}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"translated_text": translated_text}, f, ensure_ascii=False, indent=2)

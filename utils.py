import os
import json

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




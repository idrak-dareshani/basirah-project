from fastapi import FastAPI, HTTPException, Query
from typing import Optional
import json
import os
from translate import TafsirTranslator

app = FastAPI()
translator = TafsirTranslator()

DATA_ROOT = "output"
CACHE_DIR = "cache"

language_codes = {
    "en": "english",
    "ur": "urdu",
    "fr": "french",
    "de": "german"
}

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

@app.get("/tafsir/{author}/{surah}/{ayah}", summary="Get Tafsir for a specific Ayah")
def get_tafsir(author: str, surah: int, ayah: int, lang: Optional[str] = Query("ar")):
    
    surah_file = os.path.join(DATA_ROOT, author, f"{surah}.json")

    if not os.path.exists(surah_file):
        raise HTTPException(status_code=404, detail="Tafsir file not found for the given author and surah")

    with open(surah_file, "r", encoding="utf-8") as f:
        tafsir_entries = json.load(f)

    for entry in tafsir_entries:
        start, end = entry["ayah_range"]
        if start <= ayah <= end:
            tafsir_text = entry["tafsir_text"]

            if lang and lang != "ar":
                lang_code = language_codes.get(lang)
                if not lang_code:
                    raise HTTPException(status_code=400, detail=f"Unsupported language code: {lang}")
                
                cached = load_cached_translation(author, surah, ayah, lang)
                if cached:
                    tafsir_text = cached
                else:
                    try:
                        result = translator.translate_tafsir(tafsir_text, target_language=lang)
                        tafsir_text = result['translated_text']
                        save_translation_to_cache(author, surah, ayah, lang, tafsir_text)
                    except Exception as e:
                        print(f"[Translation error] {e}")
                        raise HTTPException(status_code=500, detail="Translation failed")
                
            return {
                "author": entry["author"],
                "surah_number": entry["surah_number"],
                "ayah_range": entry["ayah_range"],
                "tafsir_text": tafsir_text,
                "language": lang,
                "surah_name_english": entry["surah_name_english"],
                "surah_name_arabic": entry["surah_name_arabic"],
                "source_urls": entry.get("source_urls", [])
            }

    raise HTTPException(status_code=404, detail="Ayah not found in the given surah's tafsir")

import os
import json
from fastapi import FastAPI, HTTPException, Query
from typing import Optional
from translate import TafsirTranslator
from search_index import TafsirSearchEngine
from utils import load_cached_translation, save_translation_to_cache

app = FastAPI()
translator = TafsirTranslator()
search_engine = TafsirSearchEngine()

DATA_ROOT = "output"

language_codes = {
    "en": "english",
    "ur": "urdu",
    "fr": "french",
    "de": "german"
}

@app.get("/tafsir/topic")
def topic_search(q: str = Query(..., description="Topic query (e.g. zakat, fasting)"),
                 author: str = Query(None, description="Optional author filter"),
                 surah: int = Query(None, description="Optional surah filter"),
                 top_k: int = Query(3, ge=1, le=10),
                 lang: str = Query("ar", description="Language Code (eg. en, ur, fr)")):

    results = search_engine.search(q, top_k=top_k, author=author, surah=surah)
    final_results = []

    for res in results:
        translated_text = None
        if lang != "ar":
            lang_code = language_codes.get(lang)
            if not lang_code:
                continue

            # Use ayah range key for filename
            ayah_key = f"{res['ayah_range'][0]}_{res['ayah_range'][1]}"
            cached = load_cached_translation(res["author"], res["surah"], ayah_key, lang)
            if cached:
                translated_text = cached
            else:
                try:
                    result = translator.translate_tafsir(res["text"], "ar", lang_code)
                    translated_text = result["translated_text"]
                    save_translation_to_cache(res["author"], res["surah"], ayah_key, lang, translated_text)
                except Exception as e:
                    print(f"[Translation error] {e}")

        final_results.append({
            **res,
            "language": lang,
            "translated_text": translated_text if lang != "ar" else None
        })

    return {
        "query": q,
        "author_filter": author,
        "language": lang,
        "results": final_results
    }

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

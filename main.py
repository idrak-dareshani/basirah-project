import os
import json
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Query
from typing import Optional
from translate import TafsirTranslator
from embedding import get_embedding
from reflection import generate_reflection
from utils import (
    save_translation_to_cache,
    load_cached_translation, 
    save_reflection_to_cache,
    load_cached_reflection)
from qdrant_utils import search_tafsir

app = FastAPI()

translator = TafsirTranslator()

class TafsirPayload(BaseModel):
    author: str
    surah: str
    ayah_range: str
    tafsir_text: str

DATA_ROOT = "output"

language_codes = {
    "en": "english",
    "ur": "urdu",
    "fr": "french",
    "de": "german"
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

@app.get("/search")
def search_topic(q: str, author: str, surah: str, top_k: int = 3, lang: str = "ar"):
    embedding = get_embedding(q)
    results = search_tafsir(embedding, top_k=top_k, author=author, surah=surah)

    response = []
    for r in results:
        tafsir_text = r.payload["tafsir_text"]
        translated_text = translator.translate_tafsir(tafsir_text, "ar", lang)["translated_text"] if lang != "ar" else tafsir_text
        response.append({
            "score": r.score,
            "author": r.payload.get("author"),
            "surah": r.payload.get("surah"),
            "ayah": r.payload.get("ayah"),
            "text": tafsir_text,
            "translated_text": translated_text
        })
    return {"results": response}

@app.get("/reflect")
def reflect(author: str, surah: int, from_ayah: int, to_ayah: int,
            lang: str = Query("en")):
    surah_path = os.path.join(DATA_ROOT, author, f"{surah}.json")
    if not os.path.exists(surah_path):
        raise HTTPException(status_code=404, detail="Tafsir file not found")

    with open(surah_path, "r", encoding="utf-8") as f:
        tafsir_entries = json.load(f)

    relevant_texts = []
    for entry in tafsir_entries:
        start, end = entry["ayah_range"]
        if end < from_ayah or start > to_ayah:
            continue
        relevant_texts.append(entry["tafsir_text"])

    if not relevant_texts:
        raise HTTPException(status_code=404, detail="No tafsir text found in specified range")

    combined_text = "\n\n".join(relevant_texts)

    # Cache lookup
    cached = load_cached_reflection(author, surah, from_ayah, to_ayah, lang)
    if cached:
        reflection = cached
    else:
        reflection = generate_reflection(combined_text, lang)
        save_reflection_to_cache(author, surah, from_ayah, to_ayah, lang, reflection)

    return {
        "author": author,
        "surah": surah,
        "from_ayah": from_ayah,
        "to_ayah": to_ayah,
        "language": lang,
        "reflection": reflection
    }
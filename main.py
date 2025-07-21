from fastapi import FastAPI, HTTPException, Query
from typing import Optional
import json
import os
from translate import TafsirTranslator

app = FastAPI()

DATA_ROOT = "output"  # Adjust to match your output path

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

            translator = TafsirTranslator()
            if lang and lang != "ar":
                result = translator.translate_tafsir(tafsir_text, target_language=lang)
                tafsir_text = result['translated_text']
                
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

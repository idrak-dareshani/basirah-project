from fastapi import FastAPI, HTTPException, Query
from typing import Optional
import json
import os
from openai import OpenAI

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DATA_ROOT = "output"  # Adjust to match your output path

def translate_with_openai(text: str, target_lang: str) -> str:
    prompt = f"Translate the following Arabic Islamic tafsir text to {target_lang} in formal, clear prose:\n\n{text}"
    try:
        response = client.responses.create(
            model="gpt-4o",
            input=prompt,
            temperature=0.4,
        )
        return response.output_text.strip()
    except Exception as e:
        print(f"[Translation error] {e}")
        raise HTTPException(status_code=500, detail="Translation failed using OpenAI")
    
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
                tafsir_text = translate_with_openai(tafsir_text, lang)
                
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

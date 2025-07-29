import os
import json
from tqdm import tqdm
from qdrant_utils import setup_collection, add_tafsir_doc
from embedding import get_embedding

# Path to root directory containing author folders
DATA_DIR = "output"  # e.g. output/ibn-katheer/1.json

# Initialize collection
setup_collection(vector_size=512)

def process_tafsir_data():
    authors = [author for author in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, author))]

    for author in authors:
        author_path = os.path.join(DATA_DIR, author)
        surah_files = [f for f in os.listdir(author_path) if f.endswith(".json")]

        for surah_file in tqdm(surah_files, desc=f"Ingesting {author}"):
            surah_number = surah_file.replace(".json", "")
            file_path = os.path.join(author_path, surah_file)

            try:
                with open(file_path, encoding="utf-8") as f:
                    entries = json.load(f)
            except Exception as e:
                print(f"⚠️ Failed to load {file_path}: {e}")
                continue

            for entry in entries:
                try:
                    ayah_range = entry.get("ayah_range")
                    tafsir_text = entry.get("tafsir_text", "").strip()

                    if not tafsir_text:
                        continue

                    embedding = get_embedding(tafsir_text)

                    doc = {
                        "author": author,
                        "surah": surah_number,
                        "ayah_range": ayah_range,
                        "tafsir_text": tafsir_text
                    }

                    add_tafsir_doc(embedding, doc)
                except Exception as e:
                    print(f"⚠️ Failed to process entry in {file_path}: {e}")

if __name__ == "__main__":
    process_tafsir_data()
    print("✅ Ingestion complete.")

import os
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from translate import translate_text

class TafsirSearchEngine:
    def __init__(self, data_root: str = "output", cache_dir: str = "index"):
        self.data_root = data_root
        self.cache_dir = cache_dir
        self.model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
        self.entries = []
        self.embeddings = None

        if self._load_index_from_cache():
            print("[Index] Loaded from disk.")
        else:
            print("[Index] Rebuilding index...")
            self._build_index()
            self._save_index_to_cache()

    def _build_index(self):
        for author in os.listdir(self.data_root):
            author_path = os.path.join(self.data_root, author)
            if not os.path.isdir(author_path):
                continue
            for file in os.listdir(author_path):
                if not file.endswith(".json"):
                    continue
                surah_number = int(file.replace(".json", ""))
                with open(os.path.join(author_path, file), "r", encoding="utf-8") as f:
                    tafsir_list = json.load(f)
                for entry in tafsir_list:
                    self.entries.append({
                        "text": entry["tafsir_text"],
                        "author": entry["author"],
                        "surah": entry["surah_number"],
                        "ayah_range": entry["ayah_range"],
                        "surah_name_arabic": entry.get("surah_name_arabic", ""),
                        "surah_name_english": entry.get("surah_name_english", ""),
                        "source_urls": entry.get("source_urls", [])
                    })
        texts = [e["text"] for e in self.entries]
        self.embeddings = self.model.encode(texts, show_progress_bar=True)

    def _save_index_to_cache(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(os.path.join(self.cache_dir, "entries.pkl"), "wb") as f:
            pickle.dump(self.entries, f)
        np.save(os.path.join(self.cache_dir, "embeddings.npy"), self.embeddings)

    def _load_index_from_cache(self) -> bool:
        try:
            entries_path = os.path.join(self.cache_dir, "entries.pkl")
            embeddings_path = os.path.join(self.cache_dir, "embeddings.npy")
            if not os.path.exists(entries_path) or not os.path.exists(embeddings_path):
                return False
            with open(entries_path, "rb") as f:
                self.entries = pickle.load(f)
            self.embeddings = np.load(embeddings_path)
            return True
        except Exception as e:
            print(f"[Index Load Error] {e}")
            return False

    def search(self, query: str, top_k: int = 5, author: str = None, surah: int = None):
        #first change the query to Arabic then send it to the model
        query_ar = translate_text(query)

        query_embedding = self.model.encode([query_ar])
        sims = cosine_similarity(query_embedding, self.embeddings)[0]

        indexed_scores = list(enumerate(sims))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in indexed_scores:
            entry = self.entries[idx]
            if author and entry["author"].lower() != author.lower():
                continue
            if surah and entry["surah"] != surah:
                continue
            results.append({**entry, "score": float(score)})
            if len(results) >= top_k:
                break

        return results

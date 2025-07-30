# Basirah Project

Basirah is a multi-lingual, AI-powered Quranic Tafsir (exegesis) search and reflection engine. It enables users to search, translate, and reflect on classical tafsir texts using semantic search, translation, and generative AI.

## Features

- **Semantic Search:** Find relevant tafsir passages using vector embeddings and Qdrant vector database.
- **Multi-Language Support:** Translate tafsir into English, Urdu, French, German, and more using Deep Translator and Google Translate.
- **Spiritual Reflection:** Generate spiritual reflections on tafsir passages using OpenAI GPT models.
- **FastAPI API:** RESTful endpoints for tafsir retrieval, search, translation, and reflection.
- **Bulk Ingestion:** Scripts to process and ingest large tafsir datasets into the vector database.

## Project Structure

- `main.py` — FastAPI app exposing endpoints for tafsir, search, and reflection.
- `bulk_ingest.py` — Script to ingest tafsir data into Qdrant.
- `data_ingestion.py` — Processes raw tafsir files and consolidates them into JSON.
- `embedding.py` / `embedding_api.py` — Embedding generation using Sentence Transformers or HuggingFace API.
- `qdrant_utils.py` — Qdrant vector DB utilities.
- `reflection.py` — Generates spiritual reflections using OpenAI.
- `translate.py` — Multi-language translation utilities.
- `utils.py` — Caching for translations and reflections.
- `data/` — Raw tafsir data (per author, per surah, per ayah).
- `output/` — Processed tafsir data (per author, per surah, JSON format).
- `requirements.txt` — Python dependencies.

## Installation

1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd basirah-project
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Set up environment variables:**
   Create a `.env` file with the following (replace with your keys):
   ```env
   QDRANT_HOST=your_qdrant_url
   QDRANT_API_KEY=your_qdrant_api_key
   HF_API_KEY=your_huggingface_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

### 1. Data Preparation
- Place raw tafsir data in the `data/` directory (see structure in repo).
- Run `data_ingestion.py` to process and consolidate ayah-level tafsir into surah-level JSON files in `output/`.

   ```sh
   python data_ingestion.py
   ```

### 2. Bulk Ingestion
- Ingest processed tafsir data into Qdrant vector DB:

   ```sh
   python bulk_ingest.py
   ```

### 3. Run the API Server
- Start the FastAPI server:

   ```sh
   uvicorn main:app --reload
   ```
- Access the API docs at [http://localhost:8000/docs](http://localhost:8000/docs)

## API Endpoints

- `GET /tafsir/{author}/{surah}/{ayah}` — Retrieve tafsir for a specific ayah (with optional translation).
- `GET /search` — Semantic search for tafsir passages.
- `GET /reflect` — Generate a spiritual reflection for a range of ayahs.

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies.

## Notes
- Requires Qdrant, HuggingFace, and OpenAI API keys.
- For translation, Deep Translator (Google Translate) is used.
- Data and output folders must be structured as described above.

## License
MIT License

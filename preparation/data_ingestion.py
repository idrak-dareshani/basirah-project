import os
import json
import requests
from collections import defaultdict
from bs4 import BeautifulSoup

def fetch_tafsir_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")
        content_div = soup.find("div", id="preloaded-text")

        if not content_div:
            print(f"[Missing div] Could not find tafsir content in: {url}")
            return None

        # Extract text with newlines
        text = content_div.get_text(separator="\n")

        # Optional: remove excessive blank lines
        lines = [line.strip() for line in text.splitlines()]
        clean_text = "\n".join([line for line in lines if line])

        return clean_text.strip()

    except Exception as e:
        print(f"[Error] Failed to fetch from {url}: {e}")
        return None
    
def process_author_folder(author_name, base_dir, output_dir):
    for file in os.listdir(base_dir):
        if file.endswith(".json"):
            json_path = os.path.join(base_dir, file)
            surah_number = int(file.replace(".json", ""))

            # Load metadata
            with open(json_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            tafsir_groups = defaultdict(list)

            for entry in metadata:
                ayah = entry["ayah_number"]
                file_name = f"{surah_number}_{ayah}.txt"
                file_path = os.path.join(base_dir, file_name)

                tafsir_text = None
                if os.path.exists(file_path):
                    with open(file_path, "r", encoding="utf-8") as f:
                        tafsir_text = f.read().strip()
                else:
                    print(f"[Missing] {file_path} - attempting to fetch from {entry.get('url', '')}")
                    tafsir_text = fetch_tafsir_text_from_url(entry.get("url", ""))
                    if tafsir_text:
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(tafsir_text)
            
                if tafsir_text:
                        tafsir_groups[tafsir_text].append(entry)

            # Consolidate identical tafsir texts into ranges
            consolidated = []
            for tafsir_text, entries in tafsir_groups.items():
                ayah_numbers = sorted(e["ayah_number"] for e in entries)

                # Group consecutive ayahs
                ranges = []
                start = end = ayah_numbers[0]
                for n in ayah_numbers[1:]:
                    if n == end + 1:
                        end = n
                    else:
                        ranges.append((start, end))
                        start = end = n
                ranges.append((start, end))

                for r_start, r_end in ranges:
                    representative = entries[0]
                    consolidated.append({
                        "author": author_name,
                        "surah_number": surah_number,
                        "ayah_range": [r_start, r_end],
                        "tafsir_text": tafsir_text,
                        "surah_name_english": representative.get("surah_name_english", ""),
                        "surah_name_arabic": representative.get("surah_name_arabic", ""),
                        "source_urls": [e.get("url") for e in entries if "url" in e]
                    })

            # # Delete original ayah .txt files
            # for entry_list in tafsir_groups.values():
            #     for entry in entry_list:
            #         file_path = os.path.join(base_dir, f"{surah_number}_{entry['ayah_number']}.txt")
            #         if os.path.exists(file_path):
            #             os.remove(file_path)
            #             print(f"[Deleted] {file_path}")

            # Save the consolidated JSON
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{surah_number}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(consolidated, f, ensure_ascii=False, indent=2)

            print(f"[Saved] {output_file}")

# # Example usage:
# author = "Ibn Kathir"
# source_folder = "data/ibn-katheer"
# output_folder = "output/ibn-katheer"

if __name__ == "__main__":
    data_dir = "../data"
    output_dir = "../output"
    for author in os.listdir(data_dir):
        source_folder = os.path.join(data_dir, author)
        output_folder = os.path.join(output_dir, author)

        process_author_folder(author, source_folder, output_folder)

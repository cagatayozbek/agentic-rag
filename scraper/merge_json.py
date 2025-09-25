import os
import json

CHUNKED_DIR = "./scraped_docs/chunked_json"
OUTPUT_FILE_JSON = "./scraped_docs/all_chunks.json"
OUTPUT_FILE_JSONL = "./scraped_docs/all_chunks.jsonl"

def merge_chunked_json():
    all_chunks = []
    count = 0
    global_id = 1  # benzersiz id için sayaç

    with open(OUTPUT_FILE_JSONL, "w", encoding="utf-8") as fout_jsonl:
        for root, _, files in os.walk(CHUNKED_DIR):
            for file in files:
                if not file.endswith(".json"):
                    continue
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        if isinstance(data, list):
                            for chunk in data:
                                # metadata’ya global_chunk_id ekle
                                if "metadata" in chunk:
                                    chunk["metadata"]["global_chunk_id"] = global_id
                                else:
                                    chunk["metadata"] = {"global_chunk_id": global_id}
                                all_chunks.append(chunk)

                                # JSONL olarak satır bazlı yaz
                                fout_jsonl.write(json.dumps(chunk, ensure_ascii=False) + "\n")

                                global_id += 1
                            count += len(data)
                    except Exception as e:
                        print(f"[HATA] {path}: {e}")

    # Tek JSON kaydet
    with open(OUTPUT_FILE_JSON, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"[BİTTİ] {count} chunk birleşti → {OUTPUT_FILE_JSON} ve {OUTPUT_FILE_JSONL}")

if __name__ == "__main__":
    merge_chunked_json()
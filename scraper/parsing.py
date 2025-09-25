import os
import json
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.base import Language

RAW_MD_DIR = "./scraped_docs/raw_md"
RAW_JSON_DIR = "./scraped_docs/raw_json"
CHUNKED_DIR = "./scraped_docs/chunked_json"
os.makedirs(CHUNKED_DIR, exist_ok=True)

# Chunk ayarları
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

def get_project_from_path(md_path: str) -> str:
    """path içinden proje adını çıkar (örn: langsmith, langgraph, misc)"""
    parts = os.path.normpath(md_path).split(os.sep)
    try:
        idx = parts.index("raw_md")
        return parts[idx + 1] if idx + 1 < len(parts) else "misc"
    except ValueError:
        return "misc"

def load_raw_meta(project: str, filename_md: str):
    json_path = os.path.join(RAW_JSON_DIR, project, filename_md.replace(".md", ".json"))
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        url = data.get("url")
        title = data.get("title")
        return {"url": url, "title": title}
    except Exception:
        return None

def extract_titles(md_text: str):
    """Markdown içinden başlıkları çıkar"""
    title = None
    sections = []

    for line in md_text.splitlines():
        if line.startswith("# "):  # H1
            if not title:
                title = line.lstrip("#").strip()
        elif line.startswith("## "):  # H2
            sections.append(line.lstrip("#").strip())
        elif line.startswith("### "):  # H3
            sections.append(line.lstrip("#").strip())

    return title, sections

def chunk_markdown_file(md_path: str):
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n# ", "\n", " ", ""]
    )

    docs = splitter.create_documents([md_text])

    project = get_project_from_path(md_path)
    out_dir = os.path.join(CHUNKED_DIR, project)
    os.makedirs(out_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(md_path))[0]
    out_path = os.path.join(out_dir, base_name + ".json")

    raw_meta = load_raw_meta(project, base_name + ".md")

    # Başlıkları çıkar
    page_title, sections = extract_titles(md_text)

    out_data = []
    current_section = None
    for i, doc in enumerate(docs, 1):
        # hangi section’a düştüğünü tahmin et (en son görülen başlık)
        section = None
        for sec in sections:
            if sec in doc.page_content:
                section = sec
                current_section = sec
                break
        if not section:
            section = current_section

        out_data.append({
            "content": doc.page_content,
            "metadata": {
                "source": raw_meta.get("url") if raw_meta else None,
                "project": project,
                "title": raw_meta.get("title") if raw_meta and raw_meta.get("title") else page_title,
                "section": section,
                "chunk_id": i,
            }
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    print(f"[OK] {md_path} → {out_path} ({len(out_data)} chunk)")

def process_all(limit=None):
    count = 0
    for root, _, files in os.walk(RAW_MD_DIR):
        for file in files:
            if not file.endswith(".md"):
                continue
            md_path = os.path.join(root, file)
            chunk_markdown_file(md_path)
            count += 1
            if limit and count >= limit:
                print(f"[BİTTİ] {limit} dosya işlendi (test modu).")
                return
    print(f"[BİTTİ] Toplam {count} dosya işlendi.")

if __name__ == "__main__":
    # test için önce 2 dosya çalıştır
    # process_all(limit=2)

    # tamamına koşmak için:
    process_all()
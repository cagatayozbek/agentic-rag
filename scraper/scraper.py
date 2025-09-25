import os
import re
import json
import time
import requests
from urllib.parse import urlparse

# --- Ayarlar ---
TXT_URL = "https://docs.langchain.com/llms.txt"
OUTPUT_DIR = "./scraped_docs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_txt(url):
    """llms.txt içeriğini getir"""
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.text.splitlines()
    except Exception as e:
        print(f"[HATA] {url}: {e}")
    return []


def parse_llms(lines):
    """llms.txt satırlarını title, url, summary olarak ayır"""
    entries = []
    for line in lines:
        match = re.match(r"- \[(.+?)\]\((https://[^\s)]+\.md)\)(?:: (.*))?", line.strip())
        if match:
            title, url, summary = match.groups()
            entries.append({
                "title": title.strip(),
                "url": url.strip(),
                "summary": summary.strip() if summary else None
            })
    return entries


def get_project_name(url):
    """URL içinden proje ismini çıkar (örn: langsmith, langgraph, deep-agents)"""
    parts = urlparse(url).path.split("/")
    for p in parts:
        if p in ["langsmith", "langgraph", "langchain", "integrations", "deep-agents", "langgraph-platform", "swe","oap"]:
            return p
    return "misc"


def save_markdown(entry, content):
    """Markdown ve JSON kaydet"""
    project = get_project_name(entry["url"])
    filename = os.path.basename(entry["url"])  # orijinal .md dosya adı

    # Markdown kaydet
    md_dir = os.path.join(OUTPUT_DIR, "raw_md", project)
    os.makedirs(md_dir, exist_ok=True)
    md_path = os.path.join(md_dir, filename)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content)

    # JSON kaydet
    json_dir = os.path.join(OUTPUT_DIR, "raw_json", project)
    os.makedirs(json_dir, exist_ok=True)
    data = {
        "url": entry["url"],
        "title": entry["title"],
        "markdown": content,
        "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    json_path = os.path.join(json_dir, filename.replace(".md", ".json"))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def download_markdowns(txt_url, limit=None):
    print(f"[BAŞLIYOR] {txt_url}")
    lines = fetch_txt(txt_url)
    entries = parse_llms(lines)

    if limit:
        entries = entries[:limit]

    for i, entry in enumerate(entries, 1):
        try:
            resp = requests.get(entry["url"], timeout=10)
            if resp.status_code == 200:
                save_markdown(entry, resp.text)
                print(f"[OK] {entry['title']} → {entry['url']}")
        except Exception as e:
            print(f"[HATA] {entry['url']}: {e}")
        time.sleep(0.5)


if __name__ == "__main__":
    download_markdowns(TXT_URL, limit=1000)  # önce test için 20 sayfa
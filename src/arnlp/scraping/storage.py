import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _monthly_file() -> Path:
    now = datetime.utcnow()
    return DATA_DIR / f"articles_{now.year}_{now.month:02d}.jsonl"


def append_article(article: dict):
    path = _monthly_file()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(article, ensure_ascii=False) + "\n")


def load_all() -> list[dict]:
    articles = []
    for path in sorted(DATA_DIR.glob("articles_*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        articles.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return articles


def count_articles() -> int:
    total = 0
    for path in DATA_DIR.glob("articles_*.jsonl"):
        with open(path, encoding="utf-8") as f:
            total += sum(1 for line in f if line.strip())
    return total

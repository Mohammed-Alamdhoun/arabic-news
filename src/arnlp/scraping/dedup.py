from pathlib import Path
from urllib.parse import urlparse, urlencode, parse_qs, urlunparse

DATA_DIR = Path(__file__).parent.parent / "data"
SEEN_FILE = DATA_DIR / "urls_seen.txt"
QUEUE_FILE = DATA_DIR / "urls_queue.txt"

_STRIP_PARAMS = {"traffic_source", "utm_source", "utm_medium", "utm_campaign", "utm_term"}

_SKIP_PATTERNS = [
    "/gallery/",
    "/video/scenarios/",
    "/tag/",
    "/where/",
    "/author/",
    "/search/",
]


def normalize(url: str) -> str:
    parsed = urlparse(url.strip())
    kept = {k: v for k, v in parse_qs(parsed.query).items() if k not in _STRIP_PARAMS}
    return urlunparse(parsed._replace(query=urlencode(kept, doseq=True)))


def is_article_url(url: str) -> bool:
    return not any(p in url for p in _SKIP_PATTERNS)


class URLManager:
    def __init__(self):
        self.seen: set[str] = set()
        self.queue: list[str] = []
        self._load()

    def _load(self):
        if SEEN_FILE.exists():
            self.seen = set(line.strip() for line in SEEN_FILE.read_text().splitlines() if line.strip())
        if QUEUE_FILE.exists():
            self.queue = [line.strip() for line in QUEUE_FILE.read_text().splitlines() if line.strip()]

    def add(self, url: str) -> bool:
        norm = normalize(url)
        if not norm or norm in self.seen:
            return False
        self.seen.add(norm)
        self.queue.append(norm)
        with open(SEEN_FILE, "a", encoding="utf-8") as f:
            f.write(norm + "\n")
        with open(QUEUE_FILE, "a", encoding="utf-8") as f:
            f.write(norm + "\n")
        return True

    def add_many(self, urls: list[str]) -> int:
        return sum(self.add(u) for u in urls)

    def pop_batch(self, n: int = 50) -> list[str]:
        batch, self.queue = self.queue[:n], self.queue[n:]
        QUEUE_FILE.write_text("\n".join(self.queue) + ("\n" if self.queue else ""), encoding="utf-8")
        return batch

    def queue_size(self) -> int:
        return len(self.queue)

    def seen_count(self) -> int:
        return len(self.seen)

import re
import time
import random
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
from pathlib import Path

from .storage import append_article

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
FAILED_FILE = Path(__file__).parent.parent / "data" / "urls_failed.txt"

CATEGORY_MAP = {
    "/news/":      "أخبار",
    "/sport/":     "رياضة",
    "/culture/":   "ثقافة",
    "/ebusiness/": "اقتصاد",
    "/tech/":      "تقنية",
    "/lifestyle/": "أسلوب حياة",
    "/programs/":  "برامج",
    "/opinions/":  "آراء",
}

# Candidate selectors for article body, tried in order
_BODY_SELECTORS = [
    "div.article-body p",
    "div.wysiwyg p",
    "div.article__body p",
    "div.post-content p",
    "section.article-content p",
    "article p",
]

# Patterns that appear in non-body <p> tags to exclude
_NOISE_PATTERNS = re.compile(
    r"(جميع الحقوق محفوظة|إعلان|advertisement|cookie|تابعنا على)",
    re.IGNORECASE,
)


def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://",  HTTPAdapter(max_retries=retry))
    session.headers.update(HEADERS)
    return session


def _extract_category(url: str) -> str:
    for path, label in CATEGORY_MAP.items():
        if path in url:
            return label
    return "غير محدد"


def _extract_body(soup: BeautifulSoup) -> str:
    # Try known selectors first
    for selector in _BODY_SELECTORS:
        paragraphs = soup.select(selector)
        if len(paragraphs) >= 2:
            texts = [p.get_text(strip=True) for p in paragraphs
                     if not _NOISE_PATTERNS.search(p.get_text())]
            body = " ".join(texts)
            if len(body) > 150:
                return body

    # Fallback: find the div containing the most <p> tags
    divs = soup.find_all("div")
    if not divs:
        return ""
    best_div = max(divs, key=lambda d: len(d.find_all("p")))
    paragraphs = best_div.find_all("p")
    texts = [p.get_text(strip=True) for p in paragraphs
             if not _NOISE_PATTERNS.search(p.get_text()) and len(p.get_text(strip=True)) > 20]
    return " ".join(texts)


def _extract_date(soup: BeautifulSoup, html_text: str) -> str:
    # Try <time> tag first (future-proof)
    time_tag = soup.find("time")
    if time_tag:
        return time_tag.get("datetime", time_tag.get_text(strip=True))

    # Try common date meta tags
    for attr in [("property", "article:published_time"),
                 ("name", "publish-date"),
                 ("itemprop", "datePublished")]:
        meta = soup.find("meta", {attr[0]: attr[1]})
        if meta and meta.get("content"):
            return meta["content"]

    # Regex fallback on raw HTML
    m = re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", html_text)
    if m:
        return m.group(0)
    m = re.search(r"\d{1,2}/\d{1,2}/\d{4}", html_text)
    if m:
        return m.group(0)

    return ""


def _extract_tags(soup: BeautifulSoup) -> list[str]:
    tags = []
    # Meta keywords
    meta_kw = soup.find("meta", {"name": "keywords"})
    if meta_kw and meta_kw.get("content"):
        tags = [t.strip() for t in meta_kw["content"].split(",") if t.strip()]
    # Breadcrumb / topic links
    if not tags:
        for a in soup.select("nav a, .breadcrumb a, .tags a"):
            text = a.get_text(strip=True)
            if text and len(text) < 40:
                tags.append(text)
    return tags[:10]


def scrape_article(url: str, session: requests.Session) -> dict | None:
    try:
        r = session.get(url, timeout=15)
        if r.status_code in (403, 404, 410):
            return None  # permanent block/gone — skip silently
        r.raise_for_status()
    except Exception as e:
        print(f"  [FAIL] {url} — {e}")
        return None

    soup = BeautifulSoup(r.text, "html.parser")

    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else ""

    body = _extract_body(soup)

    if not title or not body or len(body) < 100:
        return None

    source_el = soup.find(string=re.compile(r"المصدر\s*:"))
    source = source_el.strip() if source_el else ""

    return {
        "url": url,
        "title": title,
        "body": body,
        "body_word_count": len(body.split()),
        "category": _extract_category(url),
        "source_credit": source,
        "tags": _extract_tags(soup),
        "date_published": _extract_date(soup, r.text),
        "scraped_at": datetime.utcnow().isoformat() + "Z",
        "language": "ar",
    }


def run_scraper(url_manager, batch_size: int = 50, delay: tuple[float, float] = (1.5, 3.0)):
    session = _make_session()
    stats = {"scraped": 0, "failed": 0, "skipped": 0}
    total = url_manager.queue_size()

    print(f"[Scraper] Starting — {total} URLs in queue")

    with tqdm(total=total, desc="Articles", unit="art") as pbar:
        while True:
            batch = url_manager.pop_batch(batch_size)
            if not batch:
                break

            for url in batch:
                article = scrape_article(url, session)
                if article:
                    append_article(article)
                    stats["scraped"] += 1
                else:
                    with open(FAILED_FILE, "a", encoding="utf-8") as f:
                        f.write(url + "\n")
                    stats["failed"] += 1

                pbar.update(1)
                pbar.set_postfix(ok=stats["scraped"], fail=stats["failed"])
                time.sleep(random.uniform(*delay))

    print(f"[Scraper] Done — {stats['scraped']} saved | {stats['failed']} failed")
    return stats

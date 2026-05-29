import feedparser
import requests
from bs4 import BeautifulSoup

HOMEPAGE = "https://www.aljazeera.net"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}

# Fallback: known RSS feed patterns to try if autodiscovery fails
_FALLBACK_PATHS = [
    "/xml/rss2.0.xml",
    "/rss/",
    "/feed/",
]


def discover_rss_url() -> str | None:
    try:
        r = requests.get(HOMEPAGE, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
        tag = soup.find("link", rel="alternate", type="application/rss+xml")
        if tag and tag.get("href"):
            href = tag["href"]
            return href if href.startswith("http") else HOMEPAGE + href
    except Exception:
        pass

    for path in _FALLBACK_PATHS:
        try:
            url = HOMEPAGE + path
            r = requests.get(url, headers=HEADERS, timeout=10)
            if r.status_code == 200 and "<rss" in r.text[:500]:
                return url
        except Exception:
            continue

    return None


def collect_rss(url_manager, rss_url: str | None = None) -> dict:
    if rss_url is None:
        rss_url = discover_rss_url()
    if rss_url is None:
        print("[RSS] Could not discover RSS feed URL.")
        return {"found": 0, "added": 0, "skipped": 0, "rss_url": None}

    feed = feedparser.parse(rss_url)
    stats = {"found": len(feed.entries), "added": 0, "skipped": 0, "rss_url": rss_url}

    for entry in feed.entries:
        url = entry.get("link", "").strip()
        if not url:
            continue
        added = url_manager.add(url)
        if added:
            stats["added"] += 1
        else:
            stats["skipped"] += 1

    print(f"[RSS] {stats['found']} entries | +{stats['added']} new | {stats['skipped']} already seen")
    return stats

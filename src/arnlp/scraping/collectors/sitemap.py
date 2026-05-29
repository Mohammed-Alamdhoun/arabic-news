import re
import time
import requests
from datetime import date
from xml.etree import ElementTree as ET
from tqdm import tqdm

SITEMAP_INDEX = "https://www.aljazeera.net/sitemap.xml"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


def _parse_sitemap_date(loc: str) -> date | None:
    m = re.search(r"yyyy=(\d{4})&mm=(\d{1,2})&dd=(\d{1,2})", loc)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return None


def fetch_sitemap_index() -> list[tuple[str, date | None]]:
    r = requests.get(SITEMAP_INDEX, headers=HEADERS, timeout=15)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    result = []
    for sitemap in root.findall("sm:sitemap", NS):
        loc_el = sitemap.find("sm:loc", NS)
        if loc_el is not None and loc_el.text:
            loc = loc_el.text.strip()
            result.append((loc, _parse_sitemap_date(loc)))
    return result


def fetch_daily_sitemap(url: str) -> list[str]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        return [
            u.find("sm:loc", NS).text.strip()
            for u in root.findall("sm:url", NS)
            if u.find("sm:loc", NS) is not None
        ]
    except Exception as e:
        print(f"  [WARN] Failed to fetch {url}: {e}")
        return []


def collect_sitemap(
    url_manager,
    start: date | None = None,
    end: date | None = None,
    delay: float = 0.5,
) -> dict:
    print("[Sitemap] Fetching sitemap index...")
    all_sitemaps = fetch_sitemap_index()
    print(f"[Sitemap] Found {len(all_sitemaps)} daily sitemaps")

    # Filter by date range
    if start or end:
        filtered = []
        for loc, d in all_sitemaps:
            if d is None:
                continue
            if start and d < start:
                continue
            if end and d > end:
                continue
            filtered.append((loc, d))
        all_sitemaps = filtered
        print(f"[Sitemap] {len(all_sitemaps)} sitemaps in date range "
              f"[{start} → {end}]")

    stats = {"sitemaps_processed": 0, "urls_added": 0, "urls_skipped": 0}

    for loc, d in tqdm(all_sitemaps, desc="Sitemaps", unit="day"):
        urls = fetch_daily_sitemap(loc)
        added = url_manager.add_many(urls)
        skipped = len(urls) - added
        stats["sitemaps_processed"] += 1
        stats["urls_added"] += added
        stats["urls_skipped"] += skipped
        time.sleep(delay)

    print(f"[Sitemap] Done — {stats['urls_added']} URLs queued | "
          f"{stats['urls_skipped']} duplicates/filtered")
    return stats

"""Strong, readable topic names for the BERTopic clusters.

BERTopic's default names ("0_اسرائيل_في_num_token_الاسرائيلية") are noisy —
they keep stopwords and number placeholders and repeat morphological variants.
This builds clean Arabic labels from the top c-TF-IDF words by:

  • dropping number placeholders (NUM_TOKEN, YEAR_TOKEN, …) and stopwords,
    reusing the project's own ``arnlp.preprocessing`` resources,
  • collapsing near-duplicates (اسرائيل / الاسرائيلية → one),
  • keeping the few most distinctive content words.

Labels are cached to ``data/clusters/topic_labels.json`` and loaded by the app.
If an ``OPENROUTER_API_KEY`` is configured, ``llm_label`` can produce even
cleaner phrases (same provider as the Stage-1 word fixer); the heuristic is the
dependency-free default used here.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

_ARABIC = re.compile(r"[؀-ۿ]")

# Clustering ran on the un-joined text, so multi-word named entities arrive
# split across the c-TF-IDF words (e.g. "تل ابيب" → "تل" + "ابيب", and "تل"
# is then dropped as too short, leaving the meaningless fragment "ابيب").
# We rejoin them here using the same curated list applied in scripts/apply_mwe.py:
# whenever *all* parts of a phrase appear in a topic's representation, they are
# merged back into the full phrase before the label words are picked.
_CURATED_MWES: list[str] = [
    "حزب الله",
    "تل ابيب",
    "الولايات المتحدة",
    "مضيق هرمز",
    "اطلاق النار",
    "قطاع غزة",
    "الضفة الغربية",
    "الشرق الاوسط",
    "البيت الابيض",
    "الحرس الثوري",
    "مجلس الامن",
    "الاتحاد الاوروبي",
    "كوريا الشمالية",
    "الامم المتحدة",
    "بنيامين نتنياهو",
    "الاقمار الصناعية",
    "اليورانيوم المخصب",
    "الطائرات المسيرة",
    "الغاز الطبيعي",
    "بن غفير",
    "مجتبي خامنئي",
    "عباس عراقجي",
    "نهر الليطاني",
    "جزيرة خارك",
    "بنت جبيل",
    "صفارات الانذار",
    "وول ستريت",
    "ريال مدريد",
    "نيويورك تايمز",
    "اسلام اباد",
    "دوري ابطال",
]

_FALLBACK_STOPS = set(
    "في من الى علي عن مع التي الذي هذا هذه ذلك بين بعد قبل عند لدى او ام لا ما "
    "ان انه كان قد كل بعض غير حتى ثم اذ اذا منذ خلال نحو وقد وان".split()
)


def _filters() -> tuple[set[str], set[str]]:
    """(stopwords, placeholder-tokens-lowercased), reusing arnlp if available."""
    stops, placeholders = set(_FALLBACK_STOPS), set()
    try:
        from arnlp.preprocessing import PLACEHOLDER_TOKENS, build_stopwords

        stops |= {w for w in build_stopwords()}
        placeholders |= {str(p).lower() for p in PLACEHOLDER_TOKENS}
    except Exception:  # pragma: no cover - arnlp not importable
        pass
    placeholders |= {"num_token", "year_token", "pct_token", "money_token", "url_token"}
    return stops, placeholders


def _stem(w: str) -> str:
    return w[2:] if w.startswith("ال") and len(w) > 4 else w


def _near_dup(a: str, b: str, n: int = 4) -> bool:
    a, b = _stem(a), _stem(b)
    return len(a) >= n and len(b) >= n and a[:n] == b[:n]


def _pick_words(words: list[str], stops: set[str], placeholders: set[str], k: int = 3) -> list[str]:
    chosen: list[str] = []
    for raw in words:
        w = str(raw).strip()
        wl = w.lower()
        if not w or wl in placeholders:
            continue
        if not _ARABIC.search(w):          # drop pure latin / digits
            continue
        if w in stops or wl in stops:
            continue
        if len(w) <= 2:
            continue
        if any(_near_dup(w, c) for c in chosen):
            continue
        chosen.append(w)
        if len(chosen) >= k:
            break
    return chosen


def _merge_mwes(words: list[str], mwes: list[str] = _CURATED_MWES) -> list[str]:
    """Rejoin split named entities: if every part of an MWE is present in
    ``words``, emit the full phrase (at the first part's position) and drop
    the individual parts."""
    present = set(words)
    applicable = [p for p in mwes if all(part in present for part in p.split())]
    out: list[str] = []
    consumed: set[str] = set()
    used: set[str] = set()
    for w in words:
        if w in consumed:
            continue
        phrase = next(
            (p for p in applicable if p not in used and w in p.split()), None
        )
        if phrase is not None:
            out.append(phrase)
            used.add(phrase)
            consumed.update(phrase.split())
        else:
            out.append(w)
    return out


def label_from_words(words: list[str], k: int = 3) -> str:
    stops, placeholders = _filters()
    picked = _pick_words(_merge_mwes(words), stops, placeholders, k=k)
    return " · ".join(picked) if picked else "متفرّقات"


def _parse_representation(value) -> list[str]:
    try:
        out = ast.literal_eval(value)
        if isinstance(out, (list, tuple)):
            return [str(x) for x in out]
    except (ValueError, SyntaxError):
        pass
    return str(value).replace("_", " ").split()


def build_topic_labels(topics_csv: str | Path, out_json: str | Path | None = None) -> dict[int, str]:
    """Build {topic_id: clean_label} from a BERTopic ``topics_info.csv``."""
    import pandas as pd

    df = pd.read_csv(topics_csv)
    labels: dict[int, str] = {}
    for _, row in df.iterrows():
        topic = int(row["Topic"])
        if topic == -1:
            labels[topic] = "غير مُصنّف · ضجيج"
            continue
        labels[topic] = label_from_words(_parse_representation(row.get("Representation", "")))

    if out_json is not None:
        Path(out_json).write_text(
            json.dumps({str(k): v for k, v in labels.items()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return labels


def load_topic_labels(path: str | Path) -> dict[int, str]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items()}


def load_or_build_labels(topics_csv: str | Path, cache_json: str | Path) -> dict[int, str]:
    cache_json = Path(cache_json)
    if cache_json.exists():
        return load_topic_labels(cache_json)
    return build_topic_labels(topics_csv, out_json=cache_json)


if __name__ == "__main__":  # quick CLI: regenerate the cache
    import sys

    root = Path(__file__).resolve().parents[1]
    csv = root / "data" / "clusters" / "topics_info.csv"
    out = root / "data" / "clusters" / "topic_labels.json"
    sys.path.insert(0, str(root / "src"))
    labels = build_topic_labels(csv, out_json=out)
    for t in sorted(labels)[:15]:
        print(f"{t:>3}  {labels[t]}")
    print(f"... wrote {len(labels)} labels -> {out}")

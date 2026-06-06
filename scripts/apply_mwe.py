#!/usr/bin/env python3
"""
Join multi-word expressions (MWEs) in clustering_view with underscores so
BERTopic's c-TF-IDF treats them as single tokens (e.g. حزب_الله, تل_ابيب).

Two-pass approach:
  1. Auto-discover high-PMI bigrams from the corpus (freq >= 300, PMI >= 10).
  2. Supplement with a curated list of domain-specific named entities.
  3. Apply to clustering_view only and write preprocessed_mwe.jsonl.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

ROOT = next(
    (p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data").exists()), None
)
if ROOT is None:
    raise FileNotFoundError("Run from inside the repo (no parent dir has data/).")

IN_PATH  = ROOT / "data" / "processed" / "preprocessed.jsonl"
OUT_PATH = ROOT / "data" / "processed" / "preprocessed_mwe.jsonl"

# Always joined regardless of PMI score
CURATED: list[str] = [
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
    "بنيامين نتنياهو",
    "اسلام اباد",
    "دوري ابطال",
]

MIN_FREQ = 300
MIN_PMI  = 10.0


def discover_mwes(path: Path) -> list[str]:
    unigrams: Counter = Counter()
    bigrams:  Counter = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            tokens = json.loads(line).get("clustering_view", "").split()
            for t in tokens:
                unigrams[t] += 1
            for a, b in zip(tokens, tokens[1:]):
                bigrams[(a, b)] += 1

    total = sum(unigrams.values())
    mwes = []
    for (a, b), freq in bigrams.items():
        if freq < MIN_FREQ:
            continue
        pmi = math.log2(
            (freq / total)
            / ((unigrams[a] / total) * (unigrams[b] / total))
        )
        if pmi >= MIN_PMI:
            mwes.append(f"{a} {b}")
    return sorted(mwes, key=lambda x: -bigrams[tuple(x.split())])


def apply_mwes(text: str, mwes: list[str]) -> str:
    for phrase in mwes:
        text = text.replace(phrase, phrase.replace(" ", "_"))
    return text


def main() -> None:
    print("Discovering bigram MWEs …")
    auto_mwes = discover_mwes(IN_PATH)
    # curated first so their priority is preserved; then auto, deduped
    all_mwes = list(dict.fromkeys(CURATED + auto_mwes))
    print(f"  curated : {len(CURATED)}")
    print(f"  auto    : {len(auto_mwes)}")
    print(f"  total   : {len(all_mwes)}")
    print()
    print("Auto-discovered MWEs:")
    for m in auto_mwes:
        print(f"  {m}")

    print(f"\nApplying to {IN_PATH.name} …")
    n = 0
    with open(IN_PATH, encoding="utf-8") as fin, \
         open(OUT_PATH, "w", encoding="utf-8") as fout:
        for line in fin:
            record = json.loads(line)
            if cv := record.get("clustering_view"):
                record["clustering_view"] = apply_mwes(cv, all_mwes)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1

    print(f"wrote {n} records -> {OUT_PATH}")


if __name__ == "__main__":
    main()

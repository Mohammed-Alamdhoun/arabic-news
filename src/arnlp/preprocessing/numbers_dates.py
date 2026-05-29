"""Number / date extraction and masking for Arabic news text.

Design choices (vs spelling every digit out with ``num2words``):

* ``E5`` and ``mT5`` tokenize digits natively, so the embedding view
  keeps raw ASCII digits — they carry meaning for semantic search
  ("election 2024" ≠ "election 2025").
* ``BERTopic`` uses class-TF-IDF over tokens, so a separate clustering
  view masks numbers with placeholder tags (``YEAR_TOKEN``,
  ``PCT_TOKEN``, ``MONEY_TOKEN``, ``NUM_TOKEN``) to keep topic words
  meaningful.
* Dates are extracted into a structured list of :class:`DateHit`
  records before masking, so they remain queryable for NER / ambiguity
  / search filters.
"""

from __future__ import annotations

import re

from arnlp.preprocessing.types import DateHit

# ── Date extraction ────────────────────────────────────────────────────────

# Gregorian + Levantine month names as they appear in Al Jazeera's
# normalized-but-undiacritized form.
AR_MONTHS: dict[str, int] = {
    "يناير": 1, "كانون الثاني": 1,
    "فبراير": 2, "شباط": 2,
    "مارس": 3, "اذار": 3, "آذار": 3,
    "ابريل": 4, "أبريل": 4, "نيسان": 4,
    "مايو": 5, "ايار": 5, "أيار": 5,
    "يونيو": 6, "حزيران": 6,
    "يوليو": 7, "تموز": 7,
    "اغسطس": 8, "أغسطس": 8, "اب": 8, "آب": 8,
    "سبتمبر": 9, "ايلول": 9, "أيلول": 9,
    "اكتوبر": 10, "أكتوبر": 10, "تشرين الاول": 10, "تشرين الأول": 10,
    "نوفمبر": 11, "تشرين الثاني": 11,
    "ديسمبر": 12, "كانون الاول": 12, "كانون الأول": 12,
}
_MONTH_PATTERN = "|".join(sorted(AR_MONTHS.keys(), key=len, reverse=True))

_DATE_DMY = re.compile(
    rf"\b(\d{{1,2}})\s+(?:من\s+)?({_MONTH_PATTERN})\s+(\d{{4}})\b"
)
_DATE_MY = re.compile(rf"\b({_MONTH_PATTERN})\s+(\d{{4}})\b")
_DATE_NUMERIC = re.compile(r"\b(\d{1,4})[/\-.](\d{1,2})[/\-.](\d{1,4})\b")
_DATE_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")


def _iso(y: int, m: int | None = None, d: int | None = None) -> str:
    if m is None:
        return f"{y:04d}"
    if d is None:
        return f"{y:04d}-{m:02d}"
    return f"{y:04d}-{m:02d}-{d:02d}"


def extract_dates(text: str) -> list[DateHit]:
    """Find date mentions in digit-normalized text.

    Returns hits left-to-right with non-overlapping spans. Specificity
    order: full day-month-year first, then month-year, then numeric, then
    bare year, so a later (less specific) pattern never overrides an
    earlier one.
    """
    hits: list[DateHit] = []
    claimed: list[tuple[int, int]] = []

    def overlaps(span: tuple[int, int]) -> bool:
        return any(not (span[1] <= s or span[0] >= e) for s, e in claimed)

    for m in _DATE_DMY.finditer(text):
        d, mon, y = m.group(1), m.group(2), m.group(3)
        hits.append(DateHit(
            raw=m.group(0), iso=_iso(int(y), AR_MONTHS[mon], int(d)),
            kind="dmy", span=m.span(),
        ))
        claimed.append(m.span())

    for m in _DATE_MY.finditer(text):
        if overlaps(m.span()):
            continue
        mon, y = m.group(1), m.group(2)
        hits.append(DateHit(
            raw=m.group(0), iso=_iso(int(y), AR_MONTHS[mon]),
            kind="my", span=m.span(),
        ))
        claimed.append(m.span())

    for m in _DATE_NUMERIC.finditer(text):
        if overlaps(m.span()):
            continue
        a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if a > 31:
            y, mo, d = a, b, c
        elif c > 31:
            d, mo, y = a, b, c
        else:
            continue  # ambiguous — skip
        if not (1 <= mo <= 12 and 1 <= d <= 31):
            continue
        hits.append(DateHit(
            raw=m.group(0), iso=_iso(y, mo, d),
            kind="numeric", span=m.span(),
        ))
        claimed.append(m.span())

    for m in _DATE_YEAR.finditer(text):
        if overlaps(m.span()):
            continue
        hits.append(DateHit(
            raw=m.group(0), iso=_iso(int(m.group(1))),
            kind="year", span=m.span(),
        ))
        claimed.append(m.span())

    hits.sort(key=lambda h: h.span[0])
    return hits


# ── Number masking ─────────────────────────────────────────────────────────

_PCT = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:%|بالمئة|بالمائة|في\s+المئة|في\s+المائة)"
)
_CURRENCIES = r"دولار|دولارا|دولارات|يورو|درهم|دينار|ريال|جنيه|شيكل|ليرة"
_MONEY = re.compile(rf"\b\d+(?:[.,]\d+)?\s*(?:{_CURRENCIES})\b")
_YEAR_TOKEN = re.compile(r"\b(19\d{2}|20\d{2})\b")
_NUM_TOKEN  = re.compile(r"\b\d+(?:[.,]\d+)?\b")


def mask_numbers(text: str) -> str:
    """Replace numeric expressions with placeholder tags.

    Order matters — most specific first so they aren't gobbled by the
    generic ``NUM_TOKEN`` pattern.
    """
    text = _PCT.sub(" PCT_TOKEN ", text)
    text = _MONEY.sub(" MONEY_TOKEN ", text)
    text = _YEAR_TOKEN.sub(" YEAR_TOKEN ", text)
    text = _NUM_TOKEN.sub(" NUM_TOKEN ", text)
    return re.sub(r"\s+", " ", text).strip()

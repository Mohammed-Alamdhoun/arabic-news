"""Text normalizers for the two-view preprocessing strategy.

The pipeline produces two parallel text views of every article so each
downstream consumer sees the representation it actually wants:

  * ``normalize_for_embedding``  — digits, decimals, Latin letters all
    preserved. ``%`` is spelled out as " بالمئة ". Goes to E5 / mT5.
  * ``normalize_for_clustering`` — everything outside the Arabic block
    becomes whitespace (so "أبريل/نيسان" splits into two tokens), then
    placeholder masks (``YEAR_TOKEN``, ``NUM_TOKEN``, ...) are
    preserved as standalone tokens. Goes to BERTopic.
"""

from __future__ import annotations

import re

# Placeholders injected by the number masker — must survive both views
# when they appear (only the clustering view is expected to contain
# them, but we still want a robust definition).
PLACEHOLDER_TOKENS: frozenset[str] = frozenset(
    {"PCT_TOKEN", "MONEY_TOKEN", "YEAR_TOKEN", "NUM_TOKEN"}
)

# ── digit-only normalization (Arabic-Indic → ASCII) ────────────────────────

_AR_INDIC_DIGITS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)


def normalize_digits(text: str) -> str:
    """Convert Arabic-Indic and Eastern Arabic-Indic digits to ASCII."""
    return text.translate(_AR_INDIC_DIGITS)


# ── letter normalization (shared by both views) ────────────────────────────

_TASHKEEL = re.compile(r"[ً-ْٰ]")  # Arabic diacritics + dagger alif
_TATWEEL  = "ـ"
_HAMZA_ALIFS = re.compile(r"[إأآٱ]")


def _letter_normalize(text: str) -> str:
    text = _TASHKEEL.sub("", text)
    text = text.replace(_TATWEEL, "")
    text = _HAMZA_ALIFS.sub("ا", text)
    text = text.replace("ى", "ي")
    return text


# Used by the post-lemma filter to compare diacritized lemmas against
# the (bare) stopword set.
def strip_diacritics(text: str) -> str:
    """Drop tashkeel + unify alif/ya/taa-marbutah for stopword comparison."""
    text = _letter_normalize(text)
    return text.replace("ة", "ه")


# ── embedding view ─────────────────────────────────────────────────────────

# Embedding view: keep Arabic letters, ASCII digits, Latin letters,
# whitespace, and dot/comma. A second pass strips any dot/comma that
# isn't sandwiched between digits, so "1.7" survives but sentence-final
# punctuation does not.
_NON_ARABIC_EMBED   = re.compile(r"[^ء-ي0-9A-Za-z\s.,]")
_LOOSE_PUNCT        = re.compile(r"(?<!\d)[.,]|[.,](?!\d)")
_EXTRA_SPACES       = re.compile(r"\s+")


def normalize_for_embedding(text: str) -> str:
    """View consumed by E5, mT5, semantic search.

    Keeps digits, decimals (``1.7``), thousands separators (``1,500``),
    Latin letters, and ``%`` (spelled out as " بالمئة "). Multi-line
    whitespace is collapsed.
    """
    text = _letter_normalize(text)
    text = re.sub(r"%", " بالمئة ", text)
    text = _NON_ARABIC_EMBED.sub(" ", text)
    text = _LOOSE_PUNCT.sub(" ", text)
    return _EXTRA_SPACES.sub(" ", text).strip()


# ── clustering view ────────────────────────────────────────────────────────

# Replace any non-Arabic, non-placeholder, non-whitespace char with a
# space. We keep A-Z and underscore so placeholder tokens survive as
# single units; everything else (digits, punct, Latin lowercase) becomes
# whitespace and is dropped at the token-emit step.
_NON_ARABIC_CLUSTER = re.compile(r"[^ء-يA-Z_\s]")
_ARABIC_ONLY        = re.compile(r"[^ء-ي]")
_PLACEHOLDER_RE     = re.compile(r"^[A-Z_]+$")


def normalize_for_clustering(text: str) -> str:
    """View consumed by BERTopic.

    All non-Arabic characters become whitespace so ``أبريل/نيسان`` splits
    into two month tokens (each then filtered as a stopword).
    Placeholder tokens (``YEAR_TOKEN``, etc.) are preserved as units.
    """
    text = _letter_normalize(text)
    text = _NON_ARABIC_CLUSTER.sub(" ", text)
    cleaned: list[str] = []
    for tok in text.split():
        if tok in PLACEHOLDER_TOKENS:
            cleaned.append(tok)
            continue
        if _PLACEHOLDER_RE.match(tok):
            # Latin scrap that isn't a known placeholder — drop
            continue
        arab_only = _ARABIC_ONLY.sub("", tok)
        if arab_only:
            cleaned.append(arab_only)
    return " ".join(cleaned)

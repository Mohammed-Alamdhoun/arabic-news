"""Stage 2 — Arabic linguistic preprocessing (single-module edition).

This module bundles what used to be the ``arnlp.preprocessing`` package
(types, normalizers, stopwords, tokenization, numbers_dates,
lemmatization, pipeline) into one file. Sections are separated by banner
comments and ordered by dependency.

Public API:
    ArabicTextPipeline   — main entry point, composes the steps below
    PipelineConfig       — config dataclass
    PreprocessedDocument — return type of ``ArabicTextPipeline.process``
    DateHit              — single extracted date occurrence
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Iterator, Literal

__all__ = [
    "ArabicTextPipeline",
    "PipelineConfig",
    "PreprocessedDocument",
    "DateHit",
]


# ===========================================================================
# Shared dataclasses
# ===========================================================================

DateKind = Literal["dmy", "my", "numeric", "year"]


@dataclass(frozen=True)
class DateHit:
    """A single date mention extracted from an article."""
    raw: str          # original substring from the source
    iso: str          # best-effort ISO 8601: YYYY[-MM[-DD]]
    kind: DateKind
    span: tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class PreprocessedDocument:
    """All views produced for one article.

    The three text views serve three different downstream tasks:
      * ``embedding_view``  — for multilingual-E5 / mT5 / semantic search
      * ``clustering_view`` — for BERTopic class-TF-IDF
      * ``tokens`` / ``filtered_tokens`` / ``lemmas`` — derived from clustering view
    """
    embedding_view: str
    clustering_view: str
    tokens: list[str]
    filtered_tokens: list[str]
    lemmas: list[str]
    dates: list[DateHit] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "embedding_view":  self.embedding_view,
            "clustering_view": self.clustering_view,
            "tokens":          self.tokens,
            "filtered_tokens": self.filtered_tokens,
            "lemmas":          self.lemmas,
            "dates": [
                {"raw": d.raw, "iso": d.iso, "kind": d.kind}
                for d in self.dates
            ],
        }


# ===========================================================================
# Normalizers — two-view preprocessing strategy
# ===========================================================================
#
# The pipeline produces two parallel text views of every article so each
# downstream consumer sees the representation it actually wants:
#
#   * ``normalize_for_embedding``  — digits, decimals, Latin letters all
#     preserved. ``%`` is spelled out as " بالمئة ". Goes to E5 / mT5.
#   * ``normalize_for_clustering`` — everything outside the Arabic block
#     becomes whitespace (so "أبريل/نيسان" splits into two tokens), then
#     placeholder masks (``YEAR_TOKEN``, ``NUM_TOKEN``, ...) are
#     preserved as standalone tokens. Goes to BERTopic.

# Placeholders injected by the number masker — must survive both views.
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


# ===========================================================================
# Stopwords
# ===========================================================================
#
# Base sets are organized by part-of-speech / function so each list can be
# edited in isolation. ``build_stopwords`` composes them, prefixes each
# with Arabic clitics (و/ف/ب/ل/ك) to catch "وفي", "وكان", etc., and folds
# in Levantine month names so they get filtered after the LLM splits
# ``أبريل/نيسان``-style pairs.

PREPOSITIONS = {
    "من", "الي", "الى", "عن", "علي", "على", "في", "ب", "ل", "ك",
    "بين", "خلال", "حول", "تحت", "فوق", "داخل", "خارج", "امام", "خلف",
    "ضد", "نحو", "عبر", "لدي", "لدى", "عند", "مع", "منذ", "حتى",
}

CONJUNCTIONS = {
    "و", "ف", "ثم", "او", "ام", "بل", "لكن", "لكنها", "لكنهم",
    "كما", "كذلك", "ايضا", "بينما", "حيث", "اذ", "اذا", "ان", "انما",
}

TIME_WORDS = {
    "اليوم", "امس", "غدا", "صباحا", "مساء", "ليلا", "نهارا",
    "الان", "لاحقا", "سابقا", "اخيرا", "مؤخرا", "حاليا",
    "قبل", "بعد", "اثناء", "حين", "عندما", "وقت", "لحظه",
    "السبت", "الاحد", "الاثنين", "الثلاثاء", "الاربعاء", "الخميس", "الجمعه",
    "يناير", "فبراير", "مارس", "ابريل", "مايو", "يونيو",
    "يوليو", "اغسطس", "سبتمبر", "اكتوبر", "نوفمبر", "ديسمبر",
}

PLACE_WORDS = {
    "هنا", "هناك", "هنالك", "اين", "اي", "كل", "بعض", "جميع",
    "وسط", "شمال", "جنوب", "شرق", "غرب", "شرقي", "غربي", "شمالي", "جنوبي",
}

PRONOUNS = {
    "انا", "انت", "انتم", "انتن", "هو", "هي", "هم", "هن", "نحن",
    "هذا", "هذه", "هؤلاء", "ذلك", "تلك",
    "الذي", "التي", "الذين", "اللذان", "اللتان", "ما", "من",
}

NEGATION_AND_PARTICLES = {
    "لا", "لم", "لن", "ليس", "ليست", "غير", "دون", "قد", "لقد",
    "هل", "الا", "اما", "ان", "انها", "انه", "بان", "بانها", "بانه",
    "كان", "كانت", "كانوا", "يكون", "تكون", "يعد", "تعد",
}

NEWS_FILLER_WORDS = {
    "قال", "قالت", "ذكر", "ذكرت", "اضاف", "اضافت", "اوضح", "اوضحت",
    "اشار", "اشارت", "اكد", "اكدت", "اعلن", "اعلنت", "بحسب", "وفق",
    "جاء", "جاءت", "تابع", "تابعت", "صرح", "صرحت", "بين", "بينت",
    "افاد", "افادت", "نقل", "نقلت", "حسب", "وكاله", "مصادر", "مصدر",
    "تصريح", "بيان", "تقرير", "تقارير", "صحيفه", "قناه", "موقع",
}

# Levantine month names — needed because Al Jazeera writes pairs like
# "أبريل/نيسان". The slash is converted to whitespace by the clustering
# normalizer, so both halves need to be filtered.
LEVANTINE_MONTHS = {
    "كانون الثاني", "شباط", "اذار", "نيسان", "ايار", "حزيران",
    "تموز", "اب", "ايلول", "تشرين الاول", "تشرين الثاني", "كانون الاول",
    "كانون", "تشرين",
}

# Phrasal-preposition nouns that lemmatize to bare content (بشأن → شأن).
# Adding the bare lemma here drops them from BERTopic vocabulary.
PHRASAL_PREPOSITION_LEMMAS = {"شان"}

BASE_STOPWORDS: frozenset[str] = frozenset(
    PREPOSITIONS | CONJUNCTIONS | TIME_WORDS | PLACE_WORDS |
    PRONOUNS | NEGATION_AND_PARTICLES | NEWS_FILLER_WORDS
)

CLITIC_PREFIXES: tuple[str, ...] = ("و", "ف", "ب", "ل", "ك")


def build_stopwords(extra: Iterable[str] = ()) -> frozenset[str]:
    """Compose the runtime stopword set.

    Expands the base set with every clitic prefix to catch "وفي", "وكان",
    "فمن", etc. Adds Levantine months and any extra terms supplied by
    the caller.
    """
    extra_set = set(extra)
    return frozenset(
        BASE_STOPWORDS
        | LEVANTINE_MONTHS
        | PHRASAL_PREPOSITION_LEMMAS
        | extra_set
        | {p + w for w in BASE_STOPWORDS for p in CLITIC_PREFIXES}
    )


# ===========================================================================
# Tokenization + stopword filtering
# ===========================================================================


def tokenize(text: str) -> list[str]:
    """Whitespace split. ``text`` is expected to already be normalized."""
    return text.split()


def filter_stopwords(
    tokens: Iterable[str],
    stopwords: frozenset[str],
    min_len: int = 2,
) -> list[str]:
    """Drop tokens that are stopwords or below the length floor.

    Placeholder tokens (``YEAR_TOKEN``, ...) are always preserved.
    """
    out: list[str] = []
    for t in tokens:
        if t in PLACEHOLDER_TOKENS:
            out.append(t)
            continue
        if t in stopwords:
            continue
        if len(t) <= min_len:
            continue
        out.append(t)
    return out


# ===========================================================================
# Number / date extraction and masking
# ===========================================================================
#
# * ``E5`` and ``mT5`` tokenize digits natively, so the embedding view
#   keeps raw ASCII digits — they carry meaning for semantic search.
# * ``BERTopic`` uses class-TF-IDF over tokens, so the clustering view
#   masks numbers with placeholder tags to keep topic words meaningful.
# * Dates are extracted into structured ``DateHit`` records before masking,
#   so they remain queryable for NER / ambiguity / search filters.

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


# ===========================================================================
# Lemmatization
# ===========================================================================
#
# ``MLELemmatizer`` is fast and context-free — picks the corpus-wide most
# likely analysis per token. Suitable for an MVP run.


class BaseLemmatizer(ABC):
    """Interface for any lemmatizer the pipeline can use."""

    @abstractmethod
    def lemmatize(self, tokens: list[str]) -> list[str]:
        """Return one bare (diacritic-stripped) lemma per input token.

        Placeholders must be preserved unchanged.
        """


class MLELemmatizer(BaseLemmatizer):
    """CAMeL ``MLEDisambiguator`` — fast, context-free."""

    def __init__(self) -> None:
        # deferred: camel-tools loads a pretrained model — only pay for it
        # when this lemmatizer is actually built (and skip if unused).
        from camel_tools.disambig.mle import MLEDisambiguator
        self._mle = MLEDisambiguator.pretrained()

    def lemmatize(self, tokens: list[str]) -> list[str]:
        if not tokens:
            return []

        to_disambig: list[str] = []
        is_placeholder: list[bool] = []
        for t in tokens:
            if t in PLACEHOLDER_TOKENS:
                is_placeholder.append(True)
            else:
                is_placeholder.append(False)
                to_disambig.append(t)

        analyzed = iter(self._mle.disambiguate(to_disambig)) if to_disambig else iter([])

        out: list[str] = []
        for tok, ph in zip(tokens, is_placeholder):
            if ph:
                out.append(tok)
                continue
            d = next(analyzed)
            raw = d.analyses[0].analysis.get("lex", d.word) if d.analyses else d.word
            out.append(strip_diacritics(raw))
        return out


@lru_cache(maxsize=2)
def get_lemmatizer(kind: str) -> BaseLemmatizer:
    """Lazily build a lemmatizer; cached so we never load the model twice."""
    if kind == "mle":
        return MLELemmatizer()
    raise ValueError(f"unknown lemmatizer kind: {kind!r}")


def filter_by_lemma(
    filtered_tokens: list[str],
    lemmas: list[str],
    stopwords: frozenset[str],
    min_len: int = 2,
) -> tuple[list[str], list[str]]:
    """Drop ``(token, lemma)`` pairs whose lemma is a stopword.

    Keeps the two arrays index-aligned so downstream code can zip them.
    """
    out_t: list[str] = []
    out_l: list[str] = []
    for tok, lem in zip(filtered_tokens, lemmas):
        if lem in PLACEHOLDER_TOKENS:
            out_t.append(tok); out_l.append(lem); continue
        if lem in stopwords or len(lem) <= min_len:
            continue
        out_t.append(tok); out_l.append(lem)
    return out_t, out_l


# ===========================================================================
# Pipeline — composes all of the above
# ===========================================================================


@dataclass
class PipelineConfig:
    """Knobs for ``ArabicTextPipeline``.

    Defaults reflect the validated v2 behavior.
    """
    enable_number_masking:    bool = True
    enable_date_extraction:   bool = True
    enable_lemmatization:     bool = True
    enable_post_lemma_filter: bool = True
    lemmatizer:               Literal["mle"] = "mle"
    min_token_len:            int = 2
    extra_stopwords:          frozenset[str] = field(default_factory=frozenset)


class ArabicTextPipeline:
    """Composes the preprocessing steps into a single ``process`` call.

    The lemmatizer is lazily constructed from ``config.lemmatizer`` on
    first use (so importing the pipeline is fast). Pass ``lemmatizer=``
    explicitly to bypass — useful in tests.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        lemmatizer: BaseLemmatizer | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self._lemmatizer = lemmatizer
        self.stopwords = build_stopwords(extra=self.config.extra_stopwords)

    @property
    def lemmatizer(self) -> BaseLemmatizer:
        if self._lemmatizer is None:
            self._lemmatizer = get_lemmatizer(self.config.lemmatizer)
        return self._lemmatizer

    def process(self, text: str | None) -> PreprocessedDocument:
        text = text or ""

        digit_norm = normalize_digits(text)

        dates = (
            extract_dates(digit_norm)
            if self.config.enable_date_extraction
            else []
        )

        embedding_view = normalize_for_embedding(digit_norm)

        masked = mask_numbers(digit_norm) if self.config.enable_number_masking else digit_norm
        clustering_view = normalize_for_clustering(masked)

        tokens = tokenize(clustering_view)
        filtered = filter_stopwords(
            tokens, self.stopwords, min_len=self.config.min_token_len
        )

        if self.config.enable_lemmatization:
            lemmas = self.lemmatizer.lemmatize(filtered)
            if self.config.enable_post_lemma_filter:
                filtered, lemmas = filter_by_lemma(
                    filtered, lemmas, self.stopwords,
                    min_len=self.config.min_token_len,
                )
        else:
            lemmas = list(filtered)

        return PreprocessedDocument(
            embedding_view=embedding_view,
            clustering_view=clustering_view,
            tokens=tokens,
            filtered_tokens=filtered,
            lemmas=lemmas,
            dates=dates,
        )

    def process_batch(self, texts: Iterable[str | None]) -> Iterator[PreprocessedDocument]:
        for t in texts:
            yield self.process(t)

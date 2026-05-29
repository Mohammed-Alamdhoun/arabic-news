"""Arabic stopword sets used by the preprocessing pipeline.

The base sets are organized by part-of-speech / function so future
maintainers can edit each list in isolation. The exported
``build_stopwords`` function composes the base sets, prefixes each with
Arabic clitics (و/ف/ب/ل/ك) to catch "وفي", "وكان", etc., and folds in
Levantine month names so they get filtered after the LLM splits
``أبريل/نيسان``-style pairs.
"""

from __future__ import annotations

from typing import Iterable

# ── Base stopword groups ────────────────────────────────────────────────────

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

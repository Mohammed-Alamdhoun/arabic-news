from __future__ import annotations

import os
import re

import numpy as np

__all__ = [
    "ExtractiveSummarizer",
    "LLMSummarizer",
]


_WHITESPACE = re.compile(r"\s+")

_SENT_SPLIT = re.compile(r"(?<=[.؟!…])\s+")


def _clean(text: str) -> str:
    """Collapse all whitespace (incl. newlines) to single spaces and strip."""
    return _WHITESPACE.sub(" ", text or "").strip()


# ===========================================================================
# Extractive - faithful by construction (selects original sentences)
# ===========================================================================


class ExtractiveSummarizer:
    """Adaptive, multi-sentence **extractive** summarizer (TextRank + MMR).

    Unlike the abstractive model, this **selects original sentences** from the
    article, so it *cannot hallucinate* — every word in the summary appears
    verbatim in the source. It's the faithful choice for analysis/opinion
    articles where the small abstractive model confabulates.

    Pipeline:

    1. split into sentences (very short fragments dropped);
    2. embed each sentence — multilingual-E5 by default (``embed_fn`` lets the
       app reuse its already-loaded encoder), or a TF-IDF fallback (no model);
    3. build a cosine sentence-similarity graph and score **centrality** with
       PageRank (the classic TextRank);
    4. select ``target_sentences(n_words)`` sentences with **MMR** so they are
       both central *and* non-redundant (covers more of the article);
    5. emit them in **original order** for readability.

    Length adapts to the source exactly like the abstractive summarizer
    (``round(n_words / words_per_sentence)`` clamped to
    ``[min_sentences, max_sentences]``).

    Parameters
    ----------
    embed_fn:
        Optional ``list[str] -> ndarray`` returning sentence vectors (ideally
        L2-normalized). When given, no model is loaded — pass the app's E5
        encoder here. Ignored when ``method="tfidf"``.
    method:
        ``"embedding"`` (semantic, default) or ``"tfidf"`` (lexical, no model).
    mmr_lambda:
        Trade-off in selection: 1.0 = pure centrality, 0.0 = pure diversity.
    damping:
        PageRank damping factor.
    """

    def __init__(
        self,
        *,
        embed_fn=None,
        device: str | None = None,
        method: str = "embedding",
        words_per_sentence: int = 150,
        min_sentences: int = 1,
        max_sentences: int = 6,
        min_sentence_words: int = 4,
        mmr_lambda: float = 0.7,
        damping: float = 0.85,
    ) -> None:
        self.embed_fn = embed_fn
        self.method = method
        self.words_per_sentence = words_per_sentence
        self.min_sentences = max(1, min_sentences)
        self.max_sentences = max(self.min_sentences, max_sentences)
        self.min_sentence_words = min_sentence_words
        self.mmr_lambda = mmr_lambda
        self.damping = damping

        self._device = device
        self._encoder = None

    # ------------------------------------------------------------------ #
    # Length policy (mirrors the abstractive summarizer)
    # ------------------------------------------------------------------ #
    def target_sentences(self, n_words: int) -> int:
        n = round(n_words / self.words_per_sentence)
        return max(self.min_sentences, min(self.max_sentences, n))

    def plan(self, text: str) -> dict:
        """Length plan for ``text`` (no model needed): words, target, available."""
        cleaned = _clean(text)
        sents = self._sentences(cleaned)
        return {
            "words": len(cleaned.split()),
            "target_sentences": self.target_sentences(len(cleaned.split())),
            "sentences_available": len(sents),
        }

    # ------------------------------------------------------------------ #
    # Sentences + embeddings + similarity
    # ------------------------------------------------------------------ #
    def _sentences(self, text: str) -> list[str]:
        sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
        return [s for s in sents if len(s.split()) >= self.min_sentence_words]

    def _ensure_encoder(self):
        if self._encoder is None:
            from arnlp.embeddings.e5 import E5Encoder

            self._encoder = E5Encoder(device=self._device, show_progress=False)
        return self._encoder

    def _embed(self, sentences: list[str]) -> np.ndarray:
        if self.embed_fn is not None:
            V = np.asarray(self.embed_fn(sentences), dtype=np.float32)
        else:
            V = np.asarray(self._ensure_encoder().encode(sentences), dtype=np.float32)
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return V / norms

    def _similarity(self, sentences: list[str]) -> np.ndarray:
        if self.method == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer

            from arnlp.preprocessing import normalize_for_embedding

            norm = [normalize_for_embedding(s) for s in sentences]
            X = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b").fit_transform(norm)
            sim = (X @ X.T).toarray()  # TF-IDF rows are L2-normalized → cosine
        else:
            V = self._embed(sentences)
            sim = V @ V.T
        sim = np.clip(sim, 0.0, None)
        np.fill_diagonal(sim, 0.0)
        return sim

    @staticmethod
    def _textrank(sim: np.ndarray, damping: float, iters: int = 100, tol: float = 1e-6) -> np.ndarray:
        """PageRank centrality over the sentence-similarity graph."""
        n = len(sim)
        if n == 0:
            return np.zeros(0)
        deg = sim.sum(axis=1, keepdims=True)
        deg[deg == 0] = 1.0
        trans = sim / deg                      # row-stochastic transition matrix
        rank = np.full(n, 1.0 / n)
        teleport = (1.0 - damping) / n
        for _ in range(iters):
            nxt = teleport + damping * (trans.T @ rank)
            if np.abs(nxt - rank).sum() < tol:
                rank = nxt
                break
            rank = nxt
        return rank

    def _select(self, scores: np.ndarray, sim: np.ndarray, n: int) -> list[int]:
        """MMR: greedily pick central sentences that aren't redundant."""
        scores = scores / (scores.max() or 1.0)
        chosen: list[int] = []
        candidates = list(range(len(scores)))
        lam = self.mmr_lambda
        while candidates and len(chosen) < n:
            if not chosen:
                best = max(candidates, key=lambda i: scores[i])
            else:
                best = max(
                    candidates,
                    key=lambda i: lam * scores[i] - (1 - lam) * max(sim[i][j] for j in chosen),
                )
            chosen.append(best)
            candidates.remove(best)
        return chosen

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def _summarize_one(self, text: str) -> str:
        sents = self._sentences(text)
        if not sents:
            return ""
        n = self.target_sentences(len(text.split()))
        if len(sents) <= n:
            return " ".join(sents)
        sim = self._similarity(sents)
        scores = self._textrank(sim, self.damping)
        chosen = sorted(self._select(scores, sim, n))  # original reading order
        return " ".join(sents[i] for i in chosen)

    def summarize(self, text: str) -> str:
        """Faithful extractive summary (adaptive multi-sentence)."""
        cleaned = _clean(text)
        return self._summarize_one(cleaned) if cleaned else ""

    def summarize_batch(self, texts: list[str], *, batch_size: int | None = None) -> list[str]:
        """Summarize many articles (each handled independently)."""
        _ = batch_size
        return [self.summarize(t) for t in texts]

    def summarize_article(
        self,
        article: dict,
        *,
        text_field: str = "body",
        summary_field: str = "summary",
    ) -> dict:
        """Return a copy of ``article`` with ``summary_field`` added."""
        return {**article, summary_field: self.summarize(article.get(text_field, ""))}


# ===========================================================================
# LLM — OpenAI: fix merged words + faithful summary, in one call (on demand)
# ===========================================================================

# Two-step instruction: repair word-boundary errors, then summarize using ONLY
# the article's own information. Placeholders are substituted with str.replace
# (not .format) so literal "{ }" in article text can't break it.
_LLM_SYSTEM = """\
أنت محرّر وملخّص محترف للأخبار العربية. نفّذ خطوتين على النص المُعطى:

١) أصلح بصمت أخطاء حدود الكلمات:
   - الدمج: كلمتان التصقتا بلا مسافة، مثل «توظيفالطائرات» ← «توظيف الطائرات»، «حزبالله» ← «حزب الله».
   - التجزئة: مسافة زائدة قسّمت كلمة واحدة، مثل «الع المية» ← «العالمية».

٢) لخّص النص الناتج بأمانة تامة:
   - استخدم معلومات النص فقط. يُمنع منعًا باتًا إضافة أي معلومة أو اسم أو رقم أو رأي من خارج النص.
   - لا تعكس المعنى، ولا تخمّن، ولا تستنتج ما لم يُذكر صراحةً في النص.
   - اكتب بعربية فصيحة واضحة ومترابطة، والتزم بعدد الجُمل المطلوب.

أخرج الملخّص فقط داخل الوسم، بلا أي مقدّمات أو شرح:
<SUMMARY>الملخّص هنا</SUMMARY>"""

_LLM_USER = """\
عدد جُمل الملخّص المطلوب: N_PLACEHOLDER

النص المراد تصحيحه وتلخيصه:
TEXT_PLACEHOLDER"""

_LLM_SUMMARY_RE = re.compile(r"<SUMMARY>(.*?)</SUMMARY>", re.DOTALL)


class LLMSummarizer:
    """OpenAI-backed summarizer that **fixes word-boundary errors** *and* writes a
    **faithful** Arabic summary in one call.

    A capable LLM reads through merged/fragmented words (``توظيفالطائرات`` →
    ``توظيف الطائرات``) and, at ``temperature=0`` with an explicit "use only the
    article's information" instruction, stays close to the source — the
    higher-quality, on-demand option in the app. Needs ``OPENAI_API_KEY`` in the
    environment and the ``openai`` package (both loaded lazily, so importing this
    module never requires either).

    Length adapts to the source like the other summarizers
    (``round(n_words / words_per_sentence)`` clamped to
    ``[min_sentences, max_sentences]``). Same drop-in API:
    ``summarize`` / ``summarize_batch`` / ``summarize_article`` / ``plan``.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 512,
        words_per_sentence: int = 150,
        min_sentences: int = 1,
        max_sentences: int = 6,
        max_input_chars: int = 24000,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self.model = model or os.environ.get("OPENAI_SUMMARY_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.words_per_sentence = words_per_sentence
        self.min_sentences = max(1, min_sentences)
        self.max_sentences = max(self.min_sentences, max_sentences)
        self.max_input_chars = max_input_chars
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = None

    def target_sentences(self, n_words: int) -> int:
        n = round(n_words / self.words_per_sentence)
        return max(self.min_sentences, min(self.max_sentences, n))

    def plan(self, text: str) -> dict:
        cleaned = _clean(text)
        return {
            "words": len(cleaned.split()),
            "target_sentences": self.target_sentences(len(cleaned.split())),
            "model": self.model,
        }

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY غير مضبوط — عيّنه في البيئة: export OPENAI_API_KEY='sk-...'"
            )
        from openai import OpenAI  # lazy: module imports without the SDK installed

        self._client = OpenAI(
            api_key=self.api_key, timeout=self.timeout, max_retries=self.max_retries
        )
        return self._client

    def _summarize_one(self, text: str) -> str:
        n = self.target_sentences(len(text.split()))
        if len(text) > self.max_input_chars:  # cost guard (LLM context is large)
            text = text[: self.max_input_chars]
        user = _LLM_USER.replace("N_PLACEHOLDER", str(n)).replace("TEXT_PLACEHOLDER", text)

        resp = self._ensure_client().chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            messages=[
                {"role": "system", "content": _LLM_SYSTEM},
                {"role": "user", "content": user},
            ],
        )
        content = (resp.choices[0].message.content or "").strip()
        m = _LLM_SUMMARY_RE.search(content)
        return (m.group(1) if m else content).strip()

    def summarize(self, text: str) -> str:
        """Fix merged words + faithful summary (adaptive multi-sentence)."""
        cleaned = _clean(text)
        return self._summarize_one(cleaned) if cleaned else ""

    def summarize_batch(self, texts: list[str], *, batch_size: int | None = None) -> list[str]:
        """Summarize many articles (one call each)."""
        _ = batch_size
        return [self.summarize(t) for t in texts]

    def summarize_article(
        self,
        article: dict,
        *,
        text_field: str = "body",
        summary_field: str = "summary",
    ) -> dict:
        """Return a copy of ``article`` with ``summary_field`` added."""
        return {**article, summary_field: self.summarize(article.get(text_field, ""))}

"""Arabic News — Semantic Search & 3D Topic Galaxy (Streamlit).

A dark, RTL "news dashboard" over the Al Jazeera Arabic corpus:
  • semantic search (multilingual-E5 + cosine) over the cleaned preprocessed text,
  • a 3D UMAP galaxy of the BERTopic clusters that **follows your search** —
    the result points light up gold and their cluster is spotlighted,
  • faithful extractive summaries (TextRank over E5).

Run from the repo root:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# On this CPU box, torch._dynamo probes `triton`, whose native import segfaults.
# Register it as unimportable up-front so every later torch import (E5, mT5,
# UMAP's transitive deps) skips the probe safely. MUST run before those imports.
if "triton" not in sys.modules and "torch" not in sys.modules:
    sys.modules["triton"] = None

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cluster_view  # noqa: E402
import theme  # noqa: E402
from search_engine import SemanticSearchEngine  # noqa: E402

st.set_page_config(
    page_title="بحث الأخبار الدلالي · Arabic News",
    layout="wide",
    initial_sidebar_state="collapsed",
)
theme.inject_theme(st)


# ===========================================================================
# Cached resources
# ===========================================================================

@st.cache_resource(show_spinner="…يحمّل الأخبار المعالَجة والتضمينات")
def get_engine() -> SemanticSearchEngine:
    return SemanticSearchEngine(
        articles_path=str(DATA / "processed" / "preprocessed.jsonl"),  # cleaned body
        embeddings_path=str(DATA / "embeddings" / "e5_large.npy"),
        clusters_path=str(DATA / "clusters" / "clusters.jsonl"),
        topics_path=str(DATA / "clusters" / "topics_info.csv"),
        labels_path=str(DATA / "clusters" / "topic_labels.json"),
        news_only=True,
    )


@st.cache_resource(show_spinner=False)
def get_coords():
    eng = get_engine()
    return cluster_view.load_or_compute_umap3d(eng.embeddings, DATA / "clusters" / "umap_3d.npy")


@st.cache_resource(show_spinner=False)
def get_summarizer():
    """Faithful **extractive** summarizer (TextRank over E5). Reuses the engine's
    already-loaded E5 encoder, so no second model is loaded — and it cannot
    hallucinate (it selects original article sentences)."""
    from arnlp.summarization import ExtractiveSummarizer

    eng = get_engine()

    def embed(sentences: list[str]):
        return eng.model.encode(
            [f"passage: {s}" for s in sentences],
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
        )

    return ExtractiveSummarizer(embed_fn=embed)


@st.cache_resource(show_spinner=False)
def get_llm_summarizer():
    """On-demand OpenAI summarizer: fixes merged/fragmented words and writes a
    faithful summary. Needs OPENAI_API_KEY in the environment (and `pip install
    openai`); errors are surfaced to the user if either is missing."""
    from arnlp.summarization import LLMSummarizer

    return LLMSummarizer()


@st.cache_data(show_spinner=False)
def run_search(query: str, top_k: int):
    return get_engine().search(query, top_k=top_k)


@st.cache_resource(show_spinner=False)
def get_fixed_bodies() -> dict[str, str]:
    """url -> original (word-fixed) article body, for display & summarization."""
    import json

    out: dict[str, str] = {}
    with open(DATA / "raw" / "articles_2026_05_fixed.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            out[d.get("url", "")] = d.get("body", "")
    return out


def format_body(text: str) -> str:
    """Break a run-on article body into readable paragraphs (~3 sentences)."""
    import re

    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return "—"
    sentences = [s for s in re.split(r"(?<=[.؟!])\s+", text) if s]
    paras = [" ".join(sentences[i : i + 3]) for i in range(0, len(sentences), 3)]
    return "\n\n".join(paras)


# ===========================================================================
# Header + ticker
# ===========================================================================

engine = get_engine()

st.markdown(
    theme.hero_html(
        "مجرّة الأخبار الدلاليّة",
        "ابحث بالمعنى لا بالكلمات — تضمينات multilingual-E5 وتشابه جيب التمام فوق أخبار الجزيرة. "
        "نتائج بحثك تُضيء موضعها داخل خريطة المواضيع ثلاثية الأبعاد.",
    ),
    unsafe_allow_html=True,
)
st.markdown(theme.ticker_html([]), unsafe_allow_html=True)

tab_search, tab_galaxy = st.tabs(
    ["🔎 البحث الدلالي", "خريطة المواضيع ثلاثية الأبعاد"]
)


# ===========================================================================
# Tab 1 — Search
# ===========================================================================

with tab_search:
    with st.form("search_form", enter_to_submit=False):
        c1, c2 = st.columns([6, 1])
        query = c1.text_input(
            "query",
            label_visibility="collapsed",
        )
        submitted = c2.form_submit_button("🔍 بحث")
    top_k = st.slider("عدد النتائج", min_value=3, max_value=12, value=6)

    if query:
        with st.spinner("…يفهم معنى الاستعلام ويطابقه دلاليًا"):
            results = run_search(query, top_k)

        # Remember the hit points + dominant cluster so the galaxy can focus.
        hit_topics = [r["topic"] for r in results if r.get("topic") not in (None, -1)]
        dominant = max(set(hit_topics), key=hit_topics.count) if hit_topics else None
        st.session_state["search"] = {
            "query": query,
            "hits": [r["idx"] for r in results],
            "dominant": dominant,
        }

        st.markdown(
            f'<div class="section-title">أقرب النتائج دلاليًا إلى: «{query}»</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="hint">افتح تبويب «خريطة المواضيع» لرؤية أين تقع هذه النتائج داخل المجرّة.</p>',
            unsafe_allow_html=True,
        )
        fixed_bodies = get_fixed_bodies()
        for i, r in enumerate(results, start=1):
            st.markdown(theme.result_card_html(i, r), unsafe_allow_html=True)
            body = fixed_bodies.get(r["url"]) or r["body"]
            with st.expander("اقرأ الخبر الكامل · ملخّص"):
                scol1, scol2 = st.columns(2)
                if scol1.button("📑 استخلاصي (محلي · مجاني)", key=f"sum_{i}", use_container_width=True):
                    with st.spinner("…يستخلص أهمّ جُمل الخبر"):
                        try:
                            st.success(get_summarizer().summarize(body) or "—")
                        except Exception as exc:  # noqa: BLE001
                            st.warning(f"تعذّر إنشاء الملخّص: {exc}")
                if scol2.button("🤖 OpenAI (يصحّح ويلخّص)", key=f"llm_{i}", use_container_width=True):
                    with st.spinner("…يُصحّح الكلمات الملتصقة ثم يلخّص (OpenAI)"):
                        try:
                            st.success(get_llm_summarizer().summarize(body) or "—")
                        except Exception as exc:  # noqa: BLE001
                            st.warning(f"تعذّر تشغيل ملخّص OpenAI: {exc}")
                st.markdown("---")
                st.markdown(format_body(body))
                if r.get("url"):
                    st.markdown(f"🔗 [المصدر]({r['url']})")
    else:
        st.markdown(
            '<p class="hint">اكتب استعلامًا بالعربية ثم اضغط <span class="kbd">بحث</span> — '
            'يفهم النظام المعنى ويعيد أخبارًا مرتبطة حتى لو اختلفت الكلمات.</p>',
            unsafe_allow_html=True,
        )


# ===========================================================================
# Tab 2 — 3D Topic Galaxy (follows the search)
# ===========================================================================

with tab_galaxy:
    search = st.session_state.get("search")

    if search:
        label = engine.get_topic_name(search["dominant"]) if search.get("dominant") is not None else "عدة مواضيع"
        st.markdown(
            f'<div class="section-title">🔎 نتائج «{search["query"]}» تتركّز في: '
            f'<span style="color:#ffd166">{label}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="section-title">مجرّة المواضيع — كل نقطة خبر، وكل لون موضوع</div>'
            '<p class="hint">ابحث أولًا لتُضيء نتائجك هنا، أو استكشف المواضيع يدويًا. اسحب للتدوير، مرّر للتكبير.</p>',
            unsafe_allow_html=True,
        )

    with st.container(key="galaxy_controls"):
        c1, c2, _ = st.columns([1, 1, 4])
        rotate = c1.toggle("تدوير تلقائي", value=False)
        follow = c2.toggle("تتبّع البحث", value=True)

    size = 4
    highlight = None
    hit_idxs = search["hits"] if (search and follow) else None
    hit_query = search["query"] if (search and follow) else ""

    with st.spinner("…يحسب الإسقاط ثلاثي الأبعاد (مرة واحدة، ثم يُخزَّن)"):
        coords = get_coords()

    # Map data → camera-cube coords so clicking any galaxy point flies to it.
    norm = cluster_view.camera_norm(coords)

    fig = cluster_view.build_galaxy_figure(
        coords, engine.df, engine.topic_name_map,
        point_size=size, highlight_topic=highlight,
        hit_idxs=hit_idxs, hit_query=hit_query, height=720,
    )
    leg_col, gal_col = st.columns([1, 4])
    with leg_col:
        st.markdown(
            cluster_view.legend_html(engine.df, engine.topic_name_map),
            unsafe_allow_html=True,
        )
    with gal_col:
        components.html(
            cluster_view.galaxy_html(fig, rotate=rotate, height=720, norm=norm),
            height=742, scrolling=False,
        )

    focus_topic = highlight if highlight is not None else (search["dominant"] if search else None)
    if focus_topic is not None:
        titles = engine.df.loc[engine.df["topic"] == focus_topic, "title"].head(8).tolist()
        st.markdown(
            f'<div class="section-title">عناوين من: {engine.get_topic_name(focus_topic)}</div>',
            unsafe_allow_html=True,
        )
        for t in titles:
            st.markdown(f"- {t}")

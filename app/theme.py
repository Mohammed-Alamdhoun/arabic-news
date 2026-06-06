"""Dark "news dashboard" theme for the Streamlit app.

Pure string builders (no Streamlit import except in :func:`inject_theme`), so
the HTML/CSS can be unit-tested headless. The palette is a glassy dark look
with neon per-topic chips and category accents.
"""

from __future__ import annotations

import colorsys
import html

# ===========================================================================
# Palette
# ===========================================================================

PALETTE = {
    "bg0": "#070b16",
    "bg1": "#0d1426",
    "surface": "rgba(255,255,255,0.045)",
    "surface_strong": "rgba(255,255,255,0.08)",
    "border": "rgba(255,255,255,0.10)",
    "text": "#e9eef7",
    "muted": "#9aa7bd",
    "accent": "#5b8cff",
    "accent2": "#22d3ee",
    "good": "#34d399",
}

# Per-category icon + accent colour (the 8 Al Jazeera sections in the corpus).
CATEGORY_META = {
    "أخبار":      ("📰", "#3b82f6"),
    "غير محدد":   ("📨", "#64748b"),
    "رياضة":      ("⚽", "#22c55e"),
    "اقتصاد":     ("💹", "#f59e0b"),
    "تقنية":      ("💻", "#06b6d4"),
    "آراء":       ("💬", "#a855f7"),
    "ثقافة":      ("🎭", "#ec4899"),
    "أسلوب حياة": ("🌿", "#84cc16"),
}

# Vivid qualitative palette cycled by topic id.
_TOPIC_COLORS = [
    "#60a5fa", "#f472b6", "#34d399", "#fbbf24", "#a78bfa", "#22d3ee",
    "#fb7185", "#4ade80", "#facc15", "#c084fc", "#38bdf8", "#f97316",
    "#2dd4bf", "#e879f9", "#a3e635", "#fca5a5", "#93c5fd", "#fdba74",
    "#5eead4", "#d8b4fe",
]


def topic_color(topic_id) -> str:
    """Stable colour for a topic id (-1 / noise → muted slate)."""
    try:
        t = int(topic_id)
    except (TypeError, ValueError):
        return "#475569"
    if t < 0:
        return "#475569"
    return _TOPIC_COLORS[t % len(_TOPIC_COLORS)]


def category_meta(category: str) -> tuple[str, str]:
    return CATEGORY_META.get(category, ("🗞️", "#64748b"))


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# ===========================================================================
# CSS
# ===========================================================================

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;900&family=Inter:wght@400;600;800&display=swap');

:root {{
  --accent: {PALETTE['accent']};
  --accent2: {PALETTE['accent2']};
  --text: {PALETTE['text']};
  --muted: {PALETTE['muted']};
  --border: {PALETTE['border']};
}}

/* ---- page ------------------------------------------------------------- */
.stApp {{
  background:
    radial-gradient(1100px 600px at 12% -8%, rgba(91,140,255,0.18), transparent 60%),
    radial-gradient(900px 600px at 100% 0%, rgba(34,211,238,0.14), transparent 55%),
    linear-gradient(180deg, {PALETTE['bg0']} 0%, {PALETTE['bg1']} 100%);
  background-attachment: fixed;
  color: var(--text);
  font-family: 'Cairo', 'Inter', system-ui, sans-serif;
}}
[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer, [data-testid="stToolbar"] {{ visibility: hidden; }}
.block-container {{ padding-top: 1.2rem; max-width: 1250px; position: relative; z-index: 1; }}

/* ---- galaxy starfield -------------------------------------------------- */
.stApp::before {{
  content:""; position:fixed; inset:0; z-index:0; pointer-events:none;
  background-image:
    radial-gradient(1px 1px at 25px 35px, #ffffff, transparent),
    radial-gradient(1px 1px at 80px 120px, rgba(255,255,255,.7), transparent),
    radial-gradient(1.6px 1.6px at 160px 60px, rgba(180,210,255,.95), transparent),
    radial-gradient(1px 1px at 220px 180px, rgba(255,255,255,.55), transparent),
    radial-gradient(1.4px 1.4px at 300px 240px, rgba(150,200,255,.9), transparent),
    radial-gradient(1px 1px at 340px 90px, rgba(255,255,255,.8), transparent),
    radial-gradient(1px 1px at 120px 300px, rgba(255,255,255,.6), transparent);
  background-repeat: repeat;
  background-size: 360px 360px;
  animation: starDrift 140s linear infinite, twinkle 5.5s ease-in-out infinite alternate;
}}
@keyframes starDrift {{ from {{ background-position: 0 0; }} to {{ background-position: 360px 720px; }} }}
@keyframes twinkle {{ from {{ opacity:.45; }} to {{ opacity:.95; }} }}

/* ---- hero ------------------------------------------------------------- */
.hero {{
  position: relative; border-radius: 26px; padding: 30px 34px; margin-bottom: 14px;
  background: linear-gradient(135deg, rgba(91,140,255,0.20), rgba(34,211,238,0.10)),
              {PALETTE['surface']};
  border: 1px solid var(--border);
  box-shadow: 0 20px 60px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.06);
  overflow: hidden;
}}
.hero::after {{
  content:""; position:absolute; inset:-40% -10% auto auto; width:420px; height:420px;
  background: radial-gradient(circle, rgba(91,140,255,0.35), transparent 60%);
  filter: blur(20px); animation: float 12s ease-in-out infinite;
}}
@keyframes float {{ 0%,100%{{transform:translateY(0)}} 50%{{transform:translateY(26px)}} }}
.hero h1 {{
  margin:0; font-size: 2.5rem; font-weight: 900; letter-spacing:.3px;
  background: linear-gradient(90deg, #fff, #a9c2ff 40%, #6ee7ff);
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}}
.hero p {{ margin:.5rem 0 0; color: var(--muted); font-size: 1.05rem; max-width: 760px; }}
.hero .kbd {{ color:#cdd9f2; background:rgba(255,255,255,.06); border:1px solid var(--border);
  padding:1px 8px; border-radius:7px; font-size:.85rem; }}

/* ---- stat pills ------------------------------------------------------- */
.pills {{ display:flex; gap:12px; flex-wrap:wrap; margin: 4px 0 18px; }}
.pill {{ flex:1; min-width:150px; background:{PALETTE['surface']}; border:1px solid var(--border);
  border-radius:16px; padding:14px 18px; backdrop-filter: blur(8px); }}
.pill .v {{ font-size:1.6rem; font-weight:800; color:#fff; }}
.pill .l {{ font-size:.82rem; color:var(--muted); }}

/* ---- news cards ------------------------------------------------------- */
.news-card {{
  position:relative; background:{PALETTE['surface']}; border:1px solid var(--border);
  border-radius:18px; padding:18px 20px 16px; margin: 0 0 6px;
  backdrop-filter: blur(10px);
  box-shadow: 0 10px 30px rgba(0,0,0,0.35);
  transition: transform .15s ease, border-color .15s ease, box-shadow .15s ease;
}}
.news-card:hover {{ transform: translateY(-3px); border-color: rgba(120,160,255,.45);
  box-shadow: 0 18px 44px rgba(40,80,200,0.28); }}
.news-card .rank {{ position:absolute; inset-inline-start:-10px; top:-10px; width:30px; height:30px;
  border-radius:50%; display:grid; place-items:center; font-weight:800; font-size:.85rem; color:#06122e;
  background: linear-gradient(135deg, #9ec5ff, #5b8cff); box-shadow:0 6px 16px rgba(91,140,255,.5); }}
.news-card .title {{ font-size:1.22rem; font-weight:700; color:#f4f7ff; line-height:1.6; }}
.row {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-top:12px; }}

.chip {{ display:inline-flex; align-items:center; gap:7px; padding:5px 12px; border-radius:999px;
  font-size:.82rem; font-weight:600; border:1px solid var(--border); white-space:nowrap; }}
.chip .dot {{ width:9px; height:9px; border-radius:50%; }}

.cat-badge {{ display:inline-flex; align-items:center; gap:6px; padding:5px 11px; border-radius:10px;
  font-size:.8rem; font-weight:600; }}

.score {{ display:flex; align-items:center; gap:9px; margin-inline-start:auto; }}
.score .track {{ width:120px; height:8px; border-radius:99px; background:rgba(255,255,255,.08); overflow:hidden; }}
.score .fill {{ height:100%; border-radius:99px;
  background:linear-gradient(90deg, var(--accent), var(--accent2)); box-shadow:0 0 12px rgba(34,211,238,.6); }}
.score .val {{ font-size:.8rem; color:#cfe0ff; font-variant-numeric: tabular-nums; }}
.score-val {{ margin-inline-start:auto; font-size:.8rem; color:#cfe0ff; font-variant-numeric: tabular-nums;
  background:rgba(255,255,255,.05); border:1px solid var(--border); padding:4px 11px; border-radius:999px; }}

/* ---- inputs / tabs / expander ---------------------------------------- */
.stTextInput input, .stTextArea textarea {{
  background: rgba(91,140,255,.12) !important; color: #eaf1ff !important;
  border:1px solid var(--accent) !important; border-radius:14px !important;
  direction: rtl; text-align: right; font-size:1.05rem !important; padding:14px 16px !important;
}}
/* hide Streamlit's default "Press Enter to submit form" placeholder */
.stTextInput input::placeholder {{ color: transparent !important; }}
.stTextInput input:focus {{ border-color: var(--accent) !important; box-shadow:0 0 0 3px rgba(91,140,255,.25)!important; }}
/* kill the red baseweb wrapper border/ring so only the blue input shows */
.stTextInput div[data-baseweb="input"], .stTextInput div[data-baseweb="base-input"] {{
  background: transparent !important; border: none !important; box-shadow: none !important; }}
.stTextInput div[data-baseweb="input"]:focus-within {{ border: none !important; box-shadow: none !important; }}
div.stButton > button {{
  border-radius:12px; border:1px solid var(--border);
  background: linear-gradient(135deg, rgba(91,140,255,.9), rgba(34,211,238,.75));
  color:#03102b; font-weight:800; padding:.55rem 1.1rem; transition: filter .15s ease;
}}
div.stButton > button:hover {{ filter: brightness(1.08); border-color: rgba(120,160,255,.6); }}

.stTabs [data-baseweb="tab-list"] {{ gap:6px; background:transparent; border-bottom:1px solid var(--border); }}
.stTabs [data-baseweb="tab"] {{ background:{PALETTE['surface']}; border:1px solid var(--border);
  border-bottom:none; border-radius:12px 12px 0 0; padding:8px 16px; color:var(--muted); }}
.stTabs [aria-selected="true"] {{ color:#fff !important;
  background: linear-gradient(135deg, rgba(91,140,255,.30), rgba(34,211,238,.18)) !important; }}

[data-testid="stExpander"] {{ border:1px solid var(--border); border-radius:14px;
  background: rgba(255,255,255,.03); }}
[data-testid="stExpander"] summary {{ color:#cdd9f2; }}

::-webkit-scrollbar {{ width:10px; height:10px; }}
::-webkit-scrollbar-thumb {{ background: rgba(120,150,210,.35); border-radius:10px; }}
.section-title {{ font-weight:800; font-size:1.25rem; color:#eaf1ff; margin:6px 0 2px; }}
.hint {{ color: var(--muted); font-size:.92rem; }}

/* ---- custom galaxy legend (replaces Plotly's clipped RTL legend) ------- */
.galaxy-legend {{ direction:rtl; max-height:700px; overflow-y:auto; padding:10px 12px;
  background:{PALETTE['surface']}; border:1px solid var(--border); border-radius:14px;
  backdrop-filter: blur(8px); }}
.galaxy-legend .lg-item {{ display:flex; align-items:center; gap:8px; padding:4px 2px; }}
.galaxy-legend .lg-dot {{ width:11px; height:11px; border-radius:50%; flex:0 0 auto; }}
.galaxy-legend .lg-name {{ font-size:.82rem; color:#cdd9f2; white-space:nowrap;
  overflow:hidden; text-overflow:ellipsis; }}

/* ---- RTL --------------------------------------------------------------- */
.stApp, .block-container, [data-testid="stMarkdownContainer"] {{ direction: rtl; }}
.news-card, .hero, .section-title, .hint, .pill, [data-testid="stMarkdownContainer"] p {{
  text-align: right; unicode-bidi: plaintext;
}}
.stTabs [data-baseweb="tab-list"], .stExpander {{ direction: rtl; }}

/* galaxy controls row: force LTR (left-to-right columns) */
.st-key-galaxy_controls {{ direction: ltr; }}
.st-key-galaxy_controls label {{ text-align: left; }}

/* sliders read left→right (min on left, fill aligned with the handle) */
.stSlider [data-baseweb="slider"] {{ direction: ltr; }}

/* ---- newsprint texture on the hero ------------------------------------ */
.hero::before {{
  content:""; position:absolute; inset:0; opacity:.6; pointer-events:none; z-index:0;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='26' height='26'%3E%3Ccircle cx='2' cy='2' r='1' fill='%23ffffff' fill-opacity='0.05'/%3E%3C/svg%3E");
}}
.hero h1, .hero p, .hero .live-badge {{ position: relative; z-index: 1; }}

/* ---- live badge -------------------------------------------------------- */
.live-badge {{ display:inline-flex; align-items:center; gap:7px; font-size:.78rem; font-weight:800;
  color:#fff; background:linear-gradient(90deg,#ef4444,#b91c1c); padding:4px 11px; border-radius:999px;
  margin-bottom:10px; box-shadow:0 6px 18px rgba(239,68,68,.35); }}
.live-badge .dot, .ticker .live .dot {{ width:8px; height:8px; border-radius:50%; background:#fff;
  animation: pulse 1.1s infinite; }}

/* ---- breaking-news ticker --------------------------------------------- */
.ticker {{ position:relative; overflow:hidden; white-space:nowrap; direction:rtl;
  border:1px solid var(--border); border-radius:14px; margin: 2px 0 16px; padding:9px 0;
  background:linear-gradient(90deg, rgba(239,68,68,.16), {PALETTE['surface']}); }}
.ticker .live {{ position:absolute; inset-inline-start:0; top:0; bottom:0; display:flex; align-items:center;
  gap:7px; padding:0 14px; font-weight:800; color:#fff; z-index:2;
  background:linear-gradient(90deg, #ef4444, #b91c1c); }}
.ticker-track {{ display:inline-block; padding-inline-start:130px; color:#e9eef7; font-weight:600;
  animation: ticker 100s linear infinite; }}
.ticker-track b {{ color:#ffd166; margin:0 5px; }}
/* acknowledgement overlay — slides across over the headlines on a loop */
.ticker-thanks {{ position:absolute; left:0; top:0; bottom:0; z-index:1;
  display:flex; align-items:center; white-space:nowrap; pointer-events:none;
  color:#22d3ee; font-weight:800; text-shadow:0 0 14px rgba(126,240,194,.55);
  animation: thanksSlide 14s linear infinite; }}
@keyframes thanksSlide {{ from {{ transform: translateX(-100%); }} to {{ transform: translateX(100vw); }} }}
@keyframes ticker {{ from {{ transform: translateX(-50%); }} to {{ transform: translateX(0); }} }}
@keyframes pulse {{ 0%,100% {{ opacity:1; transform:scale(1); }} 50% {{ opacity:.35; transform:scale(.65); }} }}
</style>
"""


def inject_theme(st) -> None:
    """Inject the global CSS (call once, right after ``set_page_config``)."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ===========================================================================
# HTML builders
# ===========================================================================

def hero_html(title: str, subtitle: str) -> str:
    return f"""
    <div class="hero">
      <span class="live-badge"><span class="dot"></span> مباشر · LIVE</span>
      <h1>{html.escape(title)}</h1>
      <p>{html.escape(subtitle)}</p>
    </div>
    """


# Acknowledgement woven into the breaking-news ticker (shown ~every 20s).
THANKS_MSG = (
    "كل الشكر والتقدير للدكتورة ماريا يوسف على دورة NLP الرائعة، "
    "وعلى دعمها المستمر وإثرائها لمسيرتنا العلمية بالمعرفة والخبرة القيمة."
)


def ticker_html(headlines: list[str]) -> str:
    """Breaking-news marquee. Headlines are duplicated for a seamless loop.

    The acknowledgement (:data:`THANKS_MSG`) is a separate overlay that fades in
    over the headlines every 10 seconds (see ``.ticker-thanks`` CSS), so it is
    always surfaced on a fixed cadence regardless of the scroll position.
    """
    items = "".join(f"<b>◆</b> {html.escape(h)} " for h in headlines if h)
    thanks = f'<span class="ticker-thanks">{html.escape(THANKS_MSG)}</span>'
    return (
        '<div class="ticker"><span class="live"><span class="dot"></span> عاجل</span>'
        f'<div class="ticker-track">{items}{items}</div>{thanks}</div>'
    )


def stat_pills_html(stats: list[tuple[str, str]]) -> str:
    cells = "".join(
        f'<div class="pill"><div class="v">{html.escape(str(v))}</div>'
        f'<div class="l">{html.escape(l)}</div></div>'
        for v, l in stats
    )
    return f'<div class="pills">{cells}</div>'


def _score_pct(score: float) -> int:
    # map a typical E5 cosine range (0.40–0.95) onto 0–100% for a readable bar
    pct = (float(score) - 0.40) / (0.95 - 0.40)
    return max(4, min(100, round(pct * 100)))


def topic_chip_html(topic_id, topic_name: str) -> str:
    color = topic_color(topic_id)
    return (
        f'<span class="chip" style="background:{_hex_to_rgba(color, 0.14)};'
        f'border-color:{_hex_to_rgba(color, 0.5)};color:#eaf1ff">'
        f'<span class="dot" style="background:{color};box-shadow:0 0 8px {color}"></span>'
        f'{html.escape(_short_topic(topic_name))}</span>'
    )


def category_badge_html(category: str) -> str:
    icon, color = category_meta(category)
    return (
        f'<span class="cat-badge" style="background:{_hex_to_rgba(color, 0.16)};'
        f'color:#eef3ff;border:1px solid {_hex_to_rgba(color, 0.5)}">'
        f'{icon} {html.escape(category or "—")}</span>'
    )


def _short_topic(name: str, words: int = 4) -> str:
    name = (name or "").strip()
    if not name:
        return "—"
    parts = name.split("_")
    # raw BERTopic name like "12_غزة_num_token_القطاع" -> clean it; clean labels
    # (already "غزة · القطاع · ...") pass straight through.
    if len(parts) > 1 and parts[0].lstrip("-").isdigit():
        head, *rest = parts
        rest = [w for w in rest if w.lower() not in {"num_token", "year_token", "pct_token", "money_token"}]
        return f"#{head} · " + " · ".join(rest[:words]) if rest else f"#{head}"
    return name


def result_card_html(rank: int, result: dict) -> str:
    score = float(result.get("score", 0.0))
    title = html.escape(result.get("title", "") or "")
    return f"""
    <div class="news-card">
      <div class="rank">{rank}</div>
      <div class="title">{title}</div>
      <div class="row">
        {topic_chip_html(result.get("topic"), result.get("topic_name", ""))}
        <span class="score-val">تشابه {score:.3f}</span>
      </div>
    </div>
    """

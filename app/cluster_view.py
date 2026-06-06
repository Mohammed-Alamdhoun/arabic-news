"""3D cluster 'galaxy' — UMAP→3D of the news embeddings + a Plotly figure.

The reduction reuses ``arnlp.clustering.build_umap`` (cosine — the same metric
used to actually cluster), and is cached to ``.npy``. A search can light up the
exact result points (gold stars) and dim everything but their cluster, so the
viewer's attention follows the query. Auto-rotation is done in real JS
(``galaxy_html``) rather than a Plotly frame animation, so it actually spins.
"""

from __future__ import annotations

import html as _html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _guard_triton() -> None:
    """Pre-register ``triton`` as unimportable before any torch import.

    On this CPU box ``torch._dynamo`` probes ``triton`` whose native import
    segfaults; registering it as ``None`` makes torch catch the ImportError
    and carry on (the same guard as ``arnlp.clustering._disable_triton``).
    UMAP's transitive imports can pull torch in, so we guard here too.
    """
    if "triton" not in sys.modules and "torch" not in sys.modules:
        sys.modules["triton"] = None  # type: ignore[assignment]


_guard_triton()

import theme  # noqa: E402

HOVER_TITLE_LEN = 70


def _trim(text: str, n: int = 30) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


# ===========================================================================
# UMAP → 3D (cached)
# ===========================================================================

def _build_umap_3d(seed: int = 42):
    try:
        from arnlp.clustering import build_umap

        return build_umap(n_components=3, random_state=seed)
    except Exception:  # pragma: no cover
        from umap import UMAP

        return UMAP(n_components=3, metric="cosine", min_dist=0.0, random_state=seed)


def compute_umap3d(embeddings: np.ndarray, *, seed: int = 42) -> np.ndarray:
    coords = _build_umap_3d(seed=seed).fit_transform(np.asarray(embeddings))
    return np.asarray(coords, dtype=np.float32)


def load_or_compute_umap3d(embeddings: np.ndarray, cache_path: str | Path, *, seed: int = 42) -> np.ndarray:
    cache_path = Path(cache_path)
    n = len(embeddings)
    if cache_path.exists():
        coords = np.load(cache_path)
        if coords.shape == (n, 3):
            return coords
    coords = compute_umap3d(embeddings, seed=seed)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, coords)
    return coords


# ===========================================================================
# Data point → camera coordinates
# ===========================================================================
#
# Plotly's 3D camera (eye/center) lives in a normalized cube where each axis
# spans ~[-1, 1] (aspectmode="cube"). To aim the camera at a *data* point we
# map it per axis: (val - mid) / half_range. ``camera_norm`` returns those
# constants (also handed to the JS so a click anywhere can be targeted).

def camera_norm(coords: np.ndarray) -> dict:
    """Per-axis ``{mx,my,mz,hx,hy,hz}`` mapping data coords → cube [-1, 1]."""
    coords = np.asarray(coords)
    mins, maxs = coords.min(0), coords.max(0)
    mids = (maxs + mins) / 2.0
    halfs = np.maximum((maxs - mins) / 2.0, 1e-6)
    return {
        "mx": float(mids[0]), "my": float(mids[1]), "mz": float(mids[2]),
        "hx": float(halfs[0]), "hy": float(halfs[1]), "hz": float(halfs[2]),
    }


def point_camera_target(coords: np.ndarray, idx: int, norm: dict | None = None) -> dict:
    """Normalized ``{x,y,z}`` (cube [-1, 1]) the camera should fly to for row ``idx``."""
    norm = norm or camera_norm(coords)
    p = np.asarray(coords)[idx]
    return {
        "x": (float(p[0]) - norm["mx"]) / norm["hx"],
        "y": (float(p[1]) - norm["my"]) / norm["hy"],
        "z": (float(p[2]) - norm["mz"]) / norm["hz"],
    }


# ===========================================================================
# Figure
# ===========================================================================

def _frame(df: pd.DataFrame, coords: np.ndarray) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    out["_x"], out["_y"], out["_z"] = coords[:, 0], coords[:, 1], coords[:, 2]
    out["_topic"] = out.get("topic").fillna(-1).astype(int)
    return out


def _hover(title: str, label: str, category: str) -> str:
    t = (title or "")[:HOVER_TITLE_LEN] + ("…" if len(title or "") > HOVER_TITLE_LEN else "")
    return f"<b>{t}</b><br>🏷️ {label}<br>🗂️ {category or '—'}<extra></extra>"


def build_galaxy_figure(
    coords: np.ndarray,
    df: pd.DataFrame,
    topic_name_map: dict,
    *,
    point_size: int = 3,
    highlight_topic: int | None = None,
    hit_idxs: list[int] | None = None,
    hit_query: str = "",
    selected_idx: int | None = None,
    height: int = 720,
) -> go.Figure:
    """Scatter3d galaxy. ``hit_idxs`` adds a gold 'search results' layer and,
    with ``highlight_topic``, dims everything but the results' cluster.
    ``selected_idx`` marks one article as the cinematic fly-to *destination*
    (a glowing halo + core star that the camera warps to)."""
    data = _frame(df, coords)
    fig = go.Figure()
    focusing = highlight_topic is not None or bool(hit_idxs)

    for topic in data["_topic"].value_counts().index.tolist():
        sub = data[data["_topic"] == topic]
        color = theme.topic_color(topic)
        label = _trim(topic_name_map.get(topic, f"موضوع {topic}"))
        is_noise = topic == -1

        if focusing:
            on = topic == highlight_topic
            opacity = 0.95 if on else 0.06
            size = (point_size + 2) if on else point_size
        else:
            opacity = 0.30 if is_noise else 0.82
            size = point_size

        fig.add_trace(
            go.Scatter3d(
                x=sub["_x"], y=sub["_y"], z=sub["_z"],
                mode="markers",
                name=("⚫ ضجيج" if is_noise else label)[:30],
                marker=dict(size=size, color=color, opacity=opacity, line=dict(width=0)),
                hovertemplate=[
                    _hover(t, label, c) for t, c in zip(sub.get("title", ""), sub.get("category", ""))
                ],
                visible=("legendonly" if is_noise and not focusing else True),
            )
        )

    if hit_idxs:
        hits = data.iloc[[i for i in hit_idxs if 0 <= i < len(data)]]
        fig.add_trace(
            go.Scatter3d(
                x=hits["_x"], y=hits["_y"], z=hits["_z"],
                mode="markers",
                name=f"🔎 {_trim(hit_query, 18)}" if hit_query else "🔎 نتائج البحث",
                marker=dict(
                    size=point_size + 7, color="#ffd166", symbol="diamond",
                    opacity=1.0, line=dict(width=1.5, color="#fff3c4"),
                ),
                hovertemplate=[
                    _hover(t, _trim(topic_name_map.get(int(tp), "")), c)
                    for t, tp, c in zip(hits.get("title", ""), hits["_topic"], hits.get("category", ""))
                ],
            )
        )

    if selected_idx is not None and 0 <= selected_idx < len(data):
        _add_destination(fig, data.iloc[selected_idx], point_size)

    _style_scene(fig, height)
    return fig


def _add_destination(fig: go.Figure, row: pd.Series, point_size: int) -> None:
    """A glowing 'destination' beacon (soft halo + bright core) the camera
    warps to. Both traces carry the 🛰️ marker so the JS knows to pulse them."""
    xyz = dict(x=[row["_x"]], y=[row["_y"]], z=[row["_z"]])
    title = _trim(str(row.get("title", "")), HOVER_TITLE_LEN)
    fig.add_trace(
        go.Scatter3d(
            **xyz, mode="markers", name="🛰️ الوجهة",
            marker=dict(size=point_size + 18, color="#fff3c4", opacity=0.22,
                        symbol="circle", line=dict(width=0)),
            hoverinfo="skip", showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            **xyz, mode="markers", name="🛰️ وجهتك",
            marker=dict(size=point_size + 9, color="#ffd166", opacity=1.0,
                        symbol="diamond", line=dict(width=2, color="#fff")),
            hovertemplate=f"<b>🛰️ {title}</b><extra></extra>", showlegend=False,
        )
    )


def _style_scene(fig: go.Figure, height: int) -> None:
    axis = dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False,
                title="", showspikes=False)
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        scene=dict(xaxis=axis, yaxis=axis, zaxis=axis,
                   camera=dict(eye=dict(x=1.7, y=1.7, z=0.6)),
                   aspectmode="cube", bgcolor="rgba(0,0,0,0)"),
        showlegend=False,  # replaced by a custom RTL HTML legend (legend_html)
        hoverlabel=dict(bgcolor="#0d1426", bordercolor="#5b8cff", font_size=12),
    )


# ===========================================================================
# Cinematic camera: warp-to-point fly + auto-rotate (components.html + JS)
# ===========================================================================
#
# The galaxy lives in a Plotly Scatter3d whose camera we drive in real JS via
# Plotly.relayout. Two behaviours share one controller:
#   • fly-to  — when a result card is "launched", we ease the camera's *center*
#     and *eye* from the overview onto that article's normalized data point
#     (a hyperspace overlay flashes and the gold beacons pulse), then settle
#     into a tight orbit around it;
#   • rotate  — the idle auto-spin (unchanged default radius).
# Clicking any point also flies to it. NORM maps data coords → the cube's
# [-1,1] camera space (per axis), so a click anywhere can be targeted too.

# Hyperspace overlay drawn over the plot during a warp.
_WARP_CSS = """
<style>
  .galaxy-wrap { position: relative; width: 100%; }
  #warp { position:absolute; inset:0; z-index:6; pointer-events:none;
          opacity:0; transition:opacity .35s ease; overflow:hidden; }
  #warp .rings { position:absolute; inset:-30%;
    background:radial-gradient(circle at 50% 50%,
      rgba(150,200,255,0) 30%, rgba(120,170,255,.12) 55%, rgba(80,140,255,0) 75%);
    animation:warpPulse 1.05s ease-out infinite; }
  #warp .streaks { position:absolute; inset:-55%; mix-blend-mode:screen; opacity:.55;
    background:repeating-conic-gradient(from 0deg at 50% 50%,
      rgba(190,225,255,0) 0deg, rgba(190,225,255,.22) 1deg, rgba(190,225,255,0) 3deg);
    animation:warpSpin 1.2s linear infinite; }
  #warp .core { position:absolute; left:50%; top:50%; width:8px; height:8px;
    border-radius:50%; transform:translate(-50%,-50%);
    box-shadow:0 0 40px 18px rgba(150,200,255,.5), 0 0 130px 70px rgba(91,140,255,.32); }
  @keyframes warpPulse { 0%{transform:scale(.6);opacity:0} 40%{opacity:.95} 100%{transform:scale(1.75);opacity:0} }
  @keyframes warpSpin { from{transform:rotate(0) scale(1)} to{transform:rotate(38deg) scale(1.28)} }
</style>
"""

_GALAXY_JS = """
<script>
(function () {
  var ID = "galaxy", ROTATE = %ROTATE%, FLY = %FLY%, NORM = %NORM%;
  var orbitTimer = null, pulseTimer = null, ang = 0;

  function ease(t){ return t < 0.5 ? 4*t*t*t : 1 - Math.pow(-2*t+2, 3)/2; }
  function lerp(a,b,t){ return {x:a.x+(b.x-a.x)*t, y:a.y+(b.y-a.y)*t, z:a.z+(b.z-a.z)*t}; }
  function gdOk(){ var g=document.getElementById(ID); return (g && window.Plotly && g._fullLayout) ? g : null; }
  function warp(on){ var w=document.getElementById("warp"); if(w) w.style.opacity = on ? "1" : "0"; }

  // Pulse the gold beacon traces (search hits + destination) so the target glows.
  function startPulse(){
    var gd = gdOk(); if(!gd) return;
    var idx = [], base = [];
    for (var i=0;i<gd.data.length;i++){
      var nm = gd.data[i].name || "";
      if (nm.indexOf("\\uD83D\\uDD0E") >= 0 || nm.indexOf("\\uD83D\\uDEF0") >= 0){
        idx.push(i); base.push((gd.data[i].marker && gd.data[i].marker.size) || 6);
      }
    }
    if(!idx.length) return;
    var ph = 0;
    pulseTimer = setInterval(function(){
      var g = gdOk(); if(!g) return;
      ph += 0.16;
      for (var k=0;k<idx.length;k++){
        Plotly.restyle(g, {'marker.size':[ base[k]*(1+0.30*Math.sin(ph)) ]}, [idx[k]]);
      }
    }, 60);
  }

  function startOrbit(c, radius, zoff){
    if(orbitTimer) clearInterval(orbitTimer);
    orbitTimer = setInterval(function(){
      var gd = gdOk(); if(!gd) return;
      ang += 0.01;
      Plotly.relayout(gd, {'scene.camera.eye':
        {x:c.x+radius*Math.cos(ang), y:c.y+radius*Math.sin(ang), z:c.z+zoff}});
    }, 50);
  }

  // Ease the camera onto `target` (cube [-1,1] coords): wind back, then dive in.
  function flyTo(target, done){
    var gd = gdOk(); if(!gd){ if(done) done(); return; }
    if(orbitTimer){ clearInterval(orbitTimer); orbitTimer = null; }
    var cam = gd._fullLayout.scene.camera;
    var startEye = cam.eye || {x:1.7,y:1.7,z:0.6};
    var startCtr = cam.center || {x:0,y:0,z:0};
    var midEye = {x:startEye.x*1.7, y:startEye.y*1.7, z:(startEye.z||0.6)+1.4}; // hyperspace wind-up
    var endCtr = {x:target.x, y:target.y, z:target.z};
    var endEye = {x:target.x+0.95, y:target.y+0.95, z:target.z+0.65};          // tight on the point
    var T = 2200, t0 = performance.now();
    warp(true);
    (function frame(now){
      var g = gdOk(); if(!g){ warp(false); if(done) done(); return; }
      var p = Math.min(1, (now - t0)/T), eye, ctr;
      if (p < 0.30){ eye = lerp(startEye, midEye, ease(p/0.30)); ctr = startCtr; }
      else { var q = ease((p-0.30)/0.70); eye = lerp(midEye, endEye, q); ctr = lerp(startCtr, endCtr, q); }
      Plotly.relayout(g, {'scene.camera.eye':eye, 'scene.camera.center':ctr});
      if (p < 0.82) warp(true); else warp(false);
      if (p < 1) requestAnimationFrame(frame);
      else if (done) done(endCtr);
    })(performance.now());
  }

  function clickToFly(){
    var gd = gdOk(); if(!gd || !NORM) return;
    gd.on('plotly_click', function(ev){
      if(!ev || !ev.points || !ev.points.length) return;
      var p = ev.points[0];
      var t = {x:(p.x-NORM.mx)/NORM.hx, y:(p.y-NORM.my)/NORM.hy, z:(p.z-NORM.mz)/NORM.hz};
      flyTo(t, function(c){ if(ROTATE) startOrbit(c, 1.0, 0.5); });
    });
  }

  var wait = setInterval(function(){
    var gd = gdOk(); if(!gd) return;
    clearInterval(wait);
    try {
      startPulse();
      clickToFly();
      if (FLY){ flyTo(FLY, function(c){ if(ROTATE) startOrbit(c, 1.0, 0.5); }); }
      else if (ROTATE){ startOrbit({x:0,y:0,z:0}, 1.9, 0.6); }
    } catch (e) { /* never let the camera script break the plot */ }
  }, 120);
})();
</script>
"""


def galaxy_html(
    fig: go.Figure,
    *,
    rotate: bool,
    height: int = 720,
    fly_to: dict | None = None,
    norm: dict | None = None,
) -> str:
    """Standalone HTML for the figure with the cinematic camera controller.

    ``fly_to`` is a ``{"x","y","z"}`` target in the cube's [-1,1] camera space
    (the selected article's normalized data point); ``norm`` is
    ``{"mx","my","mz","hx","hy","hz"}`` so click-to-fly can normalize any point.
    """
    plot = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        div_id="galaxy",
        config={"displayModeBar": False, "responsive": True},
    )
    js = (
        _GALAXY_JS.replace("%ROTATE%", "true" if rotate else "false")
        .replace("%FLY%", json.dumps(fly_to) if fly_to else "null")
        .replace("%NORM%", json.dumps(norm) if norm else "null")
    )
    overlay = '<div id="warp"><div class="rings"></div><div class="streaks"></div><div class="core"></div></div>'
    return f'{_WARP_CSS}<div class="galaxy-wrap">{plot}{overlay}</div>{js}'


# ===========================================================================
# Side-panel helper
# ===========================================================================

def legend_html(df: pd.DataFrame, topic_name_map: dict, *, top_n: int = 40) -> str:
    """Custom RTL legend (Plotly's clips long Arabic names). Ordered by size."""
    counts = df.get("topic").fillna(-1).astype(int).value_counts()
    items: list[str] = []
    for t, _c in counts.items():
        t = int(t)
        if t == -1:
            continue
        color = theme.topic_color(t)
        name = _html.escape(_trim(topic_name_map.get(t, f"موضوع {t}"), 40))
        items.append(
            f'<div class="lg-item"><span class="lg-dot" style="background:{color};'
            f'box-shadow:0 0 8px {color}"></span><span class="lg-name">{name}</span></div>'
        )
        if len(items) >= top_n:
            break
    return f'<div class="galaxy-legend">{"".join(items)}</div>'


def topic_overview(df: pd.DataFrame, topic_name_map: dict, *, top_n: int = 40) -> pd.DataFrame:
    counts = df.get("topic").fillna(-1).astype(int).value_counts()
    rows = [
        {"topic": int(t), "name": topic_name_map.get(int(t), f"موضوع {t}"), "count": int(c)}
        for t, c in counts.items()
        if int(t) != -1
    ]
    return pd.DataFrame(rows[:top_n])

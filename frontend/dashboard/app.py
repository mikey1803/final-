"""Streamlit dashboard for the Failure-Aware Adaptive Reasoning Framework.

Run with:  streamlit run frontend/dashboard/app.py
"""
import random
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

# --- path setup ---
# This file lives at frontend/dashboard/app.py; the backend (src/, data/)
# lives as a sibling under backend/, not under frontend/, so we go up to
# the repo root first and then into backend/.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BACKEND_ROOT = _REPO_ROOT / "backend"
_DATA_DIR = _BACKEND_ROOT / "data"
sys.path.insert(0, str(_BACKEND_ROOT))

# Attempt to load Plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ======================================================================
# Design tokens
# ======================================================================

_BG        = "#F7F5FC"
_SURFACE   = "#FFFFFF"
_BORDER    = "#E4DFF5"
_PRIMARY   = "#7C6FDB"
_SUCCESS   = "#7FB88F"
_COST      = "#D88C9A"
_TEXT      = "#322F4D"
_MUTED     = "#8B87A6"
_SIDEBAR   = "#2E2A47"

# Per-method chart colors — adaptive/bandit use shades of primary,
# baselines use muted tones, cost-aware gets the primary accent.
_METHOD_COLORS = {
    "Adaptive (Thompson)":  _PRIMARY,
    "Adaptive (UCB1)":      "#9B8FE8",
    "Adaptive (e-greedy)":  "#B0A7F0",
    "Adaptive (Decaying e)":"#6A5DC6",
    "Contextual Cost-Aware":"#5B4FBF",
    "Naive Retry":          "#C0BCD5",
    "Fixed Rule":           "#A8A3C0",
    "Random":               "#D2CEEA",
    "UniformRandom":        "#D2CEEA",
    "One-Shot LLM":         "#8B87A6",
    "Oracle":               "#1A1730",
}

# ======================================================================
# Page config
# ======================================================================

st.set_page_config(
    page_title="Failure-Aware Agent Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================================
# Custom CSS — lavender identity
# ======================================================================

st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Global ── */
/* Solid light fallback on html/body so there is never an unstyled flash and
   any translucent part of the gradient reveals lavender, never dark. */
html, body {
    background-color: #F7F5FC !important;
    color: #322F4D !important;
}
/* Containers are transparent so the fixed gradient layer (Part A, below)
   shows through *behind* all content. Text color stays forced dark. */
[data-testid="stApp"],
[data-testid="stAppViewContainer"] {
    background: transparent !important;
    color: #322F4D !important;
}

/* Force light theme colors on all main app elements in case user is in dark mode */
[data-testid="stAppViewContainer"] * {
    color: #322F4D;
}

/* ===================================================================
   PART A — Scroll-reactive lavender gradient background layer.
   A single fixed, full-viewport pseudo-element pinned BEHIND every
   Streamlit element (z-index:-1, pointer-events:none) so it can never
   interfere with hover, clicks, or text contrast. The JS listener
   (one components.html block, injected once) updates
   --scroll-gradient-pos on scroll; background-position is interpolatable
   so the short transition smooths the motion between rAF ticks.
   =================================================================== */
:root { --scroll-gradient-pos: 0%; }

[data-testid="stApp"]::before {
    content: "";
    position: fixed;
    inset: 0;
    z-index: -1;              /* behind all content, above the html fallback */
    pointer-events: none;     /* never intercept hover/clicks */
    background: linear-gradient(
        160deg,
        #F7F5FC 0%,
        #EDE9FA 30%,
        rgba(124, 111, 219, 0.12) 52%,
        #EDE9FA 74%,
        #F7F5FC 100%
    );
    background-size: 100% 240%;
    background-position: 0% var(--scroll-gradient-pos, 0%);
    transition: background-position 120ms linear;
    will-change: background-position;
}

/* Hide Streamlit default header */
[data-testid="stHeader"] {
    display: none !important;
}

h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdown"] h1,
[data-testid="stMarkdown"] h2,
[data-testid="stMarkdown"] h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #322F4D !important;
    font-weight: 600 !important;
    margin-bottom: 12px !important;
}
[data-testid="stMarkdown"] h1 { font-size: 2.2rem !important; }
[data-testid="stMarkdown"] h3 { font-size: 1.4rem !important; margin-top: 10px !important; }

p, li, span, label,
[data-testid="stMarkdown"] p {
    font-family: 'Inter', sans-serif !important;
    color: #322F4D;
}

/* ── Code Blocks ── */
code, .stCode, pre,
[data-testid="stMarkdown"] code {
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stCodeBlock"] {
    background-color: #F7F5FC !important;
    border: 1px solid #E4DFF5 !important;
    border-radius: 8px !important;
    padding: 4px;
}
[data-testid="stCodeBlock"] * {
    color: #322F4D !important;
    background-color: transparent !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #2E2A47 !important;
}
section[data-testid="stSidebar"] * {
    color: #E4DFF5 !important;
}
section[data-testid="stSidebar"] .stRadio label {
    color: #D2CEEA !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    color: #FFFFFF !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(228, 223, 245, 0.2) !important;
}

/* ── Card blocks ── */
.card {
    background: #FFFFFF !important;
    border: 1px solid #E4DFF5 !important;
    border-radius: 10px !important;
    padding: 24px 32px !important;
    margin-bottom: 24px !important;
    box-shadow: 0 2px 8px rgba(124, 111, 219, 0.05) !important;
}

/* ── Metric containers ── */
[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #E4DFF5 !important;
    border-radius: 10px !important;
    padding: 16px 20px !important;
    box-shadow: 0 2px 8px rgba(124, 111, 219, 0.05) !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: #7C6FDB !important;
    font-weight: 600 !important;
}
[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #8B87A6 !important;
}

/* ── Buttons ── */
button[kind="primary"], .stButton > button[kind="primary"] {
    background-color: #7C6FDB !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    color: #FFFFFF !important;
    padding: 8px 24px !important;
}
button[kind="primary"]:hover {
    background-color: #6A5DC6 !important;
    color: #FFFFFF !important;
}

/* ── Info box ── */
[data-testid="stAlert"] {
    background: #F0EDFB !important;
    border-left: 4px solid #7C6FDB !important;
    border-radius: 0 8px 8px 0 !important;
    color: #322F4D !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E4DFF5 !important;
    border-radius: 10px !important;
    margin-bottom: 16px !important;
    box-shadow: 0 2px 8px rgba(124, 111, 219, 0.05) !important;
}
[data-testid="stExpander"] summary {
    background-color: #FFFFFF !important;
    padding: 12px 16px !important;
}
[data-testid="stExpander"] summary p {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    color: #322F4D !important;
    font-size: 1.05rem !important;
}
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    padding: 16px !important;
    background-color: #FFFFFF !important;
    border-top: 1px solid #E4DFF5 !important;
}

/* ── Badges ── */
.badge-success {
    background: #E8F5E9;
    color: #2E7D32;
    border: 1px solid #A5D6A7;
    padding: 3px 10px;
    border-radius: 12px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 0.82rem;
}
.badge-fail {
    background: #FFEBEE;
    color: #C62828;
    border: 1px solid #EF9A9A;
    padding: 3px 10px;
    border-radius: 12px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 0.82rem;
}
.badge-neutral {
    background: #F0EDFB;
    color: #5B4FBF;
    border: 1px solid #D2CEEA;
    padding: 3px 10px;
    border-radius: 12px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 0.82rem;
}

/* ── Stat numbers ── */
.stat-mono {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    color: #322F4D;
}
.stat-primary { color: #7C6FDB; }
.stat-success { color: #7FB88F; }
.stat-cost    { color: #D88C9A; }

/* ── Image containers ── */
[data-testid="stImage"] {
    background: #FFFFFF !important;
    border: 1px solid #E4DFF5 !important;
    border-radius: 10px !important;
    padding: 16px !important;
    box-shadow: 0 2px 8px rgba(124, 111, 219, 0.05) !important;
}

/* ── Caption styling ── */
/* Slightly darker muted tone (#6A6684) than the generic _MUTED #8B87A6:
   captions sit directly on the moving gradient, so this keeps them at
   4.4-5.0:1 contrast across every scroll position (vs 2.8-3.2:1 before),
   strictly better than the old flat-background baseline. */
.stCaption, [data-testid="stCaptionContainer"] {
    font-family: 'Inter', sans-serif !important;
    color: #6A6684 !important;
    font-size: 0.85rem !important;
}

/* ===================================================================
   PART B — Entrance + hover animations (layered over Part A).
   Because navigation is a sidebar radio, each tab switch is a full
   rerun that re-renders the main block, so these CSS entrance
   animations fire on every tab open (no JS tab hook needed).
   Entrance runs on the OUTERMOST block's direct children (containers);
   hover runs on the INNER components — different elements, so the
   entrance transform and hover transform never fight each other.
   =================================================================== */

@keyframes fa-fade-slide-up {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Direct children of the outermost vertical block. Uses `> *` so it is
   agnostic to Streamlit's element/layout wrappers (stElementContainer,
   stLayoutWrapper for column rows, etc.) and only the top level animates —
   nested content rides along with its parent (no compounding transforms). */
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > * {
    animation: fa-fade-slide-up 260ms ease-out both;
}

/* Staggered fade-in for a sequence of blocks (figures, tables, cards, rows) */
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > *:nth-child(1)  { animation-delay: 20ms;  }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > *:nth-child(2)  { animation-delay: 60ms;  }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > *:nth-child(3)  { animation-delay: 100ms; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > *:nth-child(4)  { animation-delay: 140ms; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > *:nth-child(5)  { animation-delay: 180ms; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > *:nth-child(6)  { animation-delay: 220ms; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > *:nth-child(7)  { animation-delay: 260ms; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > *:nth-child(8)  { animation-delay: 300ms; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > *:nth-child(9)  { animation-delay: 340ms; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > *:nth-child(n+10) { animation-delay: 380ms; }

/* Subtle hover lift on cards / metrics / figures / expanders (inner elements) */
.card,
[data-testid="stMetric"],
[data-testid="stImage"],
[data-testid="stExpander"] {
    transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
}
.card:hover,
[data-testid="stMetric"]:hover,
[data-testid="stImage"]:hover,
[data-testid="stExpander"]:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 10px 30px rgba(124, 111, 219, 0.20) !important;
    border-color: #7C6FDB !important;
}
.stButton > button {
    transition: transform 160ms ease, box-shadow 160ms ease, background-color 160ms ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) scale(1.02);
    box-shadow: 0 6px 18px rgba(124, 111, 219, 0.28) !important;
}

/* ===================================================================
   SHARED reduced-motion guard — one media query governing BOTH parts.
   Part A: freeze the gradient at a pleasant static position (no scroll
   tracking; the JS listener also bails out under reduced-motion).
   Part B: disable all entrance + hover motion.
   =================================================================== */
@media (prefers-reduced-motion: reduce) {
    [data-testid="stApp"]::before {
        background-position: 0% 30% !important;
        transition: none !important;
        will-change: auto !important;
    }
    [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > * {
        animation: none !important;
    }
    .card, [data-testid="stMetric"], [data-testid="stImage"],
    [data-testid="stExpander"], .stButton > button {
        transition: none !important;
    }
    .card:hover, [data-testid="stMetric"]:hover, [data-testid="stImage"]:hover,
    [data-testid="stExpander"]:hover, .stButton > button:hover {
        transform: none !important;
    }
}
</style>
""", unsafe_allow_html=True)


# ======================================================================
# Part A — scroll listener (ONE fire-and-forget iframe block).
# `st.iframe` (the successor to the now-past-removal-date
# `st.components.v1.html`) embeds this raw HTML as an iframe srcdoc, so
# the <script> executes. It drives the parent document's
# --scroll-gradient-pos via requestAnimationFrame. Pure client-side: it
# never calls back into Streamlit, so it triggers no backend reruns.
# Falls back gracefully (leaves the CSS static gradient) on reduced-motion
# or any cross-origin restriction on window.parent.
# ======================================================================
st.iframe(
    """
    <script>
    (function () {
      var pdoc, root;
      try {
        pdoc = window.parent.document;      // reach the real app document
        root = pdoc.documentElement;
      } catch (e) { return; }               // cross-origin -> CSS static fallback

      // Respect reduced-motion: leave the gradient at its CSS default (static).
      try {
        if (window.parent.matchMedia &&
            window.parent.matchMedia('(prefers-reduced-motion: reduce)').matches) {
          return;
        }
      } catch (e) {}

      // Tear down a previous rerun's listener so they don't accumulate.
      try { if (window.parent.__faScrollCleanup) window.parent.__faScrollCleanup(); } catch (e) {}

      function candidates() {
        return [
          pdoc.querySelector('section[data-testid="stMain"]'),
          pdoc.querySelector('[data-testid="stAppViewContainer"]'),
          pdoc.scrollingElement,
          root
        ].filter(Boolean);
      }
      function pickScroller() {
        var best = root, bestRange = -1;
        candidates().forEach(function (el) {
          var range = el.scrollHeight - el.clientHeight;
          if (range > bestRange) { bestRange = range; best = el; }
        });
        return best;
      }

      var ticking = false;
      function update() {
        ticking = false;
        try {
          var el = pickScroller();
          var range = Math.max(el.scrollHeight - el.clientHeight, 1);
          var pct = Math.min(Math.max(el.scrollTop / range, 0), 1) * 100;
          root.style.setProperty('--scroll-gradient-pos', pct.toFixed(2) + '%');
        } catch (e) {}
      }
      var raf = window.parent.requestAnimationFrame || window.requestAnimationFrame;
      function onScroll() {
        if (!ticking) { ticking = true; raf(update); }
      }

      var targets = candidates();
      targets.forEach(function (el) { el.addEventListener('scroll', onScroll, { passive: true }); });
      window.parent.addEventListener('resize', onScroll, { passive: true });

      window.parent.__faScrollCleanup = function () {
        try {
          targets.forEach(function (el) { el.removeEventListener('scroll', onScroll); });
          window.parent.removeEventListener('resize', onScroll);
        } catch (e) {}
      };

      update();  // set initial position
    })();
    </script>
    """,
    height=1,   # st.iframe requires a positive height; 1px is effectively invisible
)


# ======================================================================
# Inline SVG — convergence motif
# ======================================================================

_CONVERGENCE_SVG = """
<svg width="260" height="56" viewBox="0 0 260 56" fill="none" xmlns="http://www.w3.org/2000/svg">
  <!-- scattered/tangled exploratory lines (left) -->
  <path d="M8 8 C30 42, 50 6, 80 30 Q110 50, 140 34 L190 28" stroke="#C0BCD5" stroke-width="1.2" fill="none" opacity="0.55"/>
  <path d="M6 44 C25 10, 60 50, 90 22 Q120 6, 150 32 L190 28" stroke="#B0A7F0" stroke-width="1.2" fill="none" opacity="0.50"/>
  <path d="M10 26 C35 50, 55 4, 85 38 Q115 48, 145 30 L190 28" stroke="#D2CEEA" stroke-width="1.2" fill="none" opacity="0.45"/>
  <path d="M4 36 C28 8, 62 46, 92 18 Q125 12, 155 28 L190 28" stroke="#9B8FE8" stroke-width="1.2" fill="none" opacity="0.50"/>
  <path d="M12 18 C38 46, 58 12, 88 34 Q118 42, 148 30 L190 28" stroke="#E4DFF5" stroke-width="1.2" fill="none" opacity="0.40"/>
  <!-- converged bold line (right) -->
  <line x1="190" y1="28" x2="252" y2="28" stroke="#7C6FDB" stroke-width="2.8" stroke-linecap="round"/>
  <circle cx="252" cy="28" r="3.5" fill="#7C6FDB"/>
</svg>
"""


# ======================================================================
# Plotly theme helper
# ======================================================================

def _apply_lavender_layout(fig, height=500):
    """Apply the lavender design system to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#F7F5FC",
        font=dict(family="Inter, sans-serif", color=_TEXT, size=13),
        title_font=dict(family="Space Grotesk, sans-serif", color=_TEXT, size=18),
        xaxis=dict(
            gridcolor=_BORDER,
            linecolor=_BORDER,
            tickfont=dict(family="JetBrains Mono, monospace", size=11, color=_MUTED),
            title_font=dict(family="Inter, sans-serif", color=_MUTED),
        ),
        yaxis=dict(
            gridcolor=_BORDER,
            linecolor=_BORDER,
            tickfont=dict(family="JetBrains Mono, monospace", size=11, color=_MUTED),
            title_font=dict(family="Inter, sans-serif", color=_MUTED),
        ),
        height=height,
        margin=dict(l=40, r=24, t=32, b=40),
    )
    return fig


# ======================================================================
# Data loader (unchanged)
# ======================================================================

@st.cache_data
def load_cost_efficiency_data():
    return pd.read_csv(_DATA_DIR / "cost_efficiency_summary.csv")


# ======================================================================
# Sidebar — with section icons
# ======================================================================

with st.sidebar:
    st.markdown("### 🧭 Navigation")
    tab_choice = st.radio(
        "Navigate to",
        [
            "1. Overview",
            "2. Cost-Efficiency",
            "3. Convergence",
            "4. Classifier Performance",
            "5. Live Episode Walkthrough",
        ],
        format_func=lambda x: {
            "1. Overview":                "📊  Overview",
            "2. Cost-Efficiency":         "💰  Cost-Efficiency",
            "3. Convergence":             "📈  Convergence",
            "4. Classifier Performance":  "🎯  Classifier",
            "5. Live Episode Walkthrough":"⚡  Live Walkthrough",
        }[x],
        index=0,
    )
    st.divider()
    st.markdown("**Failure-Aware Agent**")
    st.caption("Final Validated Results  ·  30 Trials × 500 Episodes")


# ======================================================================
# 1. Overview
# ======================================================================
if tab_choice == "1. Overview":
    # ── Header with convergence SVG ──
    hdr_left, hdr_right = st.columns([3, 1])
    with hdr_left:
        st.markdown("# Failure-Aware Adaptive Recovery Framework")
    with hdr_right:
        st.markdown(
            f'<div style="display:flex;align-items:center;justify-content:flex-end;height:100%;padding-top:12px;">'
            f'{_CONVERGENCE_SVG}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="card">'
        '<p>This project presents a failure-aware adaptive reasoning framework for LLM agents. '
        'It uses a multi-armed bandit strategy selector (including a novel <strong>Contextual '
        'Cost-Aware</strong> variant) to adaptively recover from tool execution failures based on '
        'contextual features and success probabilities. The results presented here are from the '
        'final, validated 500-episode × 30-trial evaluation.</p></div>',
        unsafe_allow_html=True,
    )

    st.markdown("### 9-Method Success Rate Comparison")
    df = load_cost_efficiency_data()

    if PLOTLY_AVAILABLE:
        # compute error bar magnitudes
        df["error_y"] = df["ci95_upper_late_stage"] - df["mean_late_stage_success_rate"]
        df["error_y_minus"] = df["mean_late_stage_success_rate"] - df["ci95_lower_late_stage"]

        # Sort by success rate
        df_sorted = df.sort_values("mean_late_stage_success_rate", ascending=False)

        fig = go.Figure()
        for idx, row in df_sorted.iterrows():
            m = row["method"]
            color = _METHOD_COLORS.get(m, _MUTED)
            fig.add_trace(go.Bar(
                name=m,
                x=[m],
                y=[row["mean_late_stage_success_rate"]],
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=[row["error_y"]],
                    arrayminus=[row["error_y_minus"]],
                    color=_TEXT,
                    thickness=1.4,
                ),
                marker_color=color,
                marker_line=dict(width=0),
            ))
        fig.update_layout(
            yaxis_title="Late-Stage Success Rate",
            xaxis_title="Method",
            yaxis_range=[0, 1.0],
            showlegend=False,
        )
        _apply_lavender_layout(fig, height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(df.set_index("method")["mean_late_stage_success_rate"])


# ======================================================================
# 2. Cost-Efficiency
# ======================================================================
elif tab_choice == "2. Cost-Efficiency":
    st.markdown("# Cost-Efficiency Trade-off")
    st.caption("Average Recovery Cost vs. Late-Stage Success Rate")

    df = load_cost_efficiency_data()

    # Static callout in a card
    st.markdown(
        '<div class="card">'
        '<p style="margin-bottom:8px;"><strong style="font-family:\'Space Grotesk\',sans-serif;">Validated Cost-Aware vs Thompson Sampling</strong> '
        '<span class="stat-mono" style="color:#8B87A6;">(λ = 0.6)</span></p>'
        '<p style="margin:4px 0;">📉 <strong>12.8% cost reduction</strong> — '
        '<span class="stat-mono stat-cost">d = 2.55</span>, '
        '<span class="stat-mono">p &lt; 0.0001</span> (significant)</p>'
        '<p style="margin:4px 0;">📊 <strong>Comparable success rate</strong> — '
        '<span class="stat-mono stat-primary">d = −0.43</span>, '
        '<span class="stat-mono">p = 0.095</span> (not significant)</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        for idx, row in df.iterrows():
            m = row["method"]
            color = _METHOD_COLORS.get(m, _MUTED)
            # Highlight Thompson and Contextual Cost-Aware
            if m in ["Contextual Cost-Aware", "Adaptive (Thompson)"]:
                marker_symbol = "star"
                marker_size = 18
                marker_line = dict(width=2, color=_TEXT)
            else:
                marker_symbol = "circle"
                marker_size = 12
                marker_line = dict(width=1, color=_BORDER)

            fig.add_trace(go.Scatter(
                x=[row["mean_late_stage_success_rate"]],
                y=[row["mean_average_cost"]],
                mode="markers+text",
                name=m,
                text=[m] if m in ["Contextual Cost-Aware", "Adaptive (Thompson)"] else [""],
                textposition="top center",
                textfont=dict(family="Inter, sans-serif", size=11, color=_TEXT),
                marker=dict(
                    color=color,
                    size=marker_size,
                    symbol=marker_symbol,
                    line=marker_line,
                ),
            ))

        fig.update_layout(
            xaxis_title="Late-Stage Success Rate",
            yaxis_title="Average Recovery Cost",
            showlegend=True,
            legend=dict(
                font=dict(family="Inter, sans-serif", size=11, color=_MUTED),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=_BORDER,
                borderwidth=1,
            ),
        )
        _apply_lavender_layout(fig, height=600)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(df[["method", "mean_late_stage_success_rate", "mean_average_cost"]])


# ======================================================================
# 3. Convergence
# ======================================================================
elif tab_choice == "3. Convergence":
    st.markdown("# Cumulative Regret Convergence")
    st.markdown(
        '<div class="card">'
        '<p><strong>Cumulative Regret</strong> measures the total success lost by not always choosing the optimal '
        'strategy in hindsight. A curve that flattens out indicates that the bandit algorithm has '
        'successfully converged on the optimal strategy. Linear growth indicates failure to converge.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    regret_path = _DATA_DIR / "regret_curve.png"
    if regret_path.exists():
        st.image(str(regret_path), use_container_width=True)
    else:
        st.warning("Regret plot not found in data/ directory.")


# ======================================================================
# 4. Classifier Performance
# ======================================================================
elif tab_choice == "4. Classifier Performance":
    st.markdown("# Failure-Type Classifier Performance")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Classifier Accuracy", "95%", help="Performance on held-out simulated failure events")
        st.markdown(
            '<div class="card" style="margin-top:16px;">'
            '<p>The TF-IDF + LinearSVC classifier successfully maps raw error strings '
            'and execution context to one of the predefined taxonomy classes, allowing '
            'the bandit selector to contextualize its strategy choices.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    with c2:
        cm_path = _DATA_DIR / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Confusion Matrix on Test Set", use_container_width=True)
        else:
            st.warning("Confusion matrix plot not found in data/ directory.")


# ======================================================================
# 5. Live Episode Walkthrough
# ======================================================================
elif tab_choice == "5. Live Episode Walkthrough":
    st.markdown("# Live Episode Trace")
    st.caption(
        "Fresh/untrained instance for demonstration — trained results shown "
        "in Overview and Cost-Efficiency tabs reflect 500-episode learning."
    )

    # Only load models when on this tab
    @st.cache_resource
    def load_trace_models():
        from src.failures.classifier import FailureClassifier
        from src.recovery.contextual_selector import ContextualCostAwareSelector

        clf = FailureClassifier()
        joblib_path = _DATA_DIR / "classifier_model.joblib"
        if joblib_path.exists():
            clf.load(joblib_path)

        # As requested, initialized with lambda_cost=0.6
        sel = ContextualCostAwareSelector(lambda_cost=0.6, seed=8888)
        return clf, sel

    clf, trace_sel = load_trace_models()

    if "trace_log" not in st.session_state:
        st.session_state["trace_log"] = []
    if "trace_ep" not in st.session_state:
        st.session_state["trace_ep"] = 0

    if st.button("Run Episode", type="primary"):
        from src.failures.taxonomy import FailureType
        from src.failures.injector import FailureInjector
        from src.recovery.strategies import execute_strategy

        rng = random.Random(st.session_state["trace_ep"])

        injector = FailureInjector(injection_rate=1.0, seed=st.session_state["trace_ep"])
        event = injector.inject({
            "step_index": st.session_state["trace_ep"],
            "tool_name": rng.choice(["web_search", "calculator", "db_query", "code_executor"]),
            "tool_args": {"q": "trace"}, "goal": "Live demo episode",
            "tool_output": "Tool returned a response.",
        })
        if event is None:
            event = injector.get_injection_log()[-1]

        predicted = clf.predict(event.raw_error, event.step_context)
        proba = clf.predict_proba(event.raw_error, event.step_context)
        conf = proba[predicted]

        # Use the classifier prediction instead of ground truth for realism
        strategy = trace_sel.select_strategy(predicted)

        outcome = execute_strategy(strategy, event, event.step_context, rng=rng)

        # In a real environment, it updates with the *predicted* state
        trace_sel.update(predicted, strategy, success=outcome["success"], episode_number=st.session_state["trace_ep"])

        entry = {
            "episode": st.session_state["trace_ep"] + 1,
            "failure_type": event.failure_type.value,
            "raw_error": event.raw_error[:120],
            "classifier_pred": predicted.value,
            "confidence": conf,
            "strategy": strategy.value,
            "success": outcome["success"],
            "cost": outcome.get("cost", 0.0),
            "action": outcome.get("action", ""),
        }
        st.session_state["trace_log"].insert(0, entry)
        st.session_state["trace_log"] = st.session_state["trace_log"][:10]
        st.session_state["trace_ep"] += 1
        st.rerun()

    if st.session_state["trace_log"]:
        st.subheader(f"Last {len(st.session_state['trace_log'])} Episodes")
        for entry in st.session_state["trace_log"]:
            badge = '<span class="badge-success">SUCCESS</span>' if entry["success"] else '<span class="badge-fail">FAILURE</span>'
            with st.expander(
                f"Episode {entry['episode']} | "
                f"{entry['classifier_pred'].replace('_',' ').title()} | "
                f"{entry['strategy'].replace('_',' ').title()} | "
                f"{'SUCCESS' if entry['success'] else 'FAILURE'} (Cost: {entry['cost']})",
                expanded=(entry["episode"] == st.session_state["trace_ep"]),
            ):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Failure detected (Ground truth):** <span class='badge-neutral'>{entry['failure_type']}</span>", unsafe_allow_html=True)
                    st.code(entry["raw_error"], language=None)
                with c2:
                    st.markdown(
                        f"**Classifier predicts:** `{entry['classifier_pred']}` "
                        f"(<span class='stat-mono'>{entry['confidence']:.1%}</span> confidence)",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**Recovery strategy:** `{entry['strategy']}`")
                    st.markdown(f"**Action details:** {entry['action']}")

                st.markdown(f"**Outcome:** {badge} | **Cost Incurred:** <span class='stat-mono stat-cost'>{entry['cost']}</span>", unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="card" style="text-align:center;color:#8B87A6;">'
            '⚡ Click <strong>Run Episode</strong> above to simulate a single failure-recovery cycle.'
            '</div>',
            unsafe_allow_html=True,
        )

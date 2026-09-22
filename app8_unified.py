"""
app8_unified.py — Unified Super-Resolution Evolution Dashboard
==============================================================
One app that walks through the entire research journey —
from the first FSRCNN proof-of-concept to the final spatio-temporal fusion model.
"""

import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SR Evolution Lab",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Global CSS — Premium Dark Glassmorphism Theme
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base & Background ─────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background: radial-gradient(ellipse at 20% 50%, #0d1b2a 0%, #050a0f 60%, #060b14 100%);
    color: #e2e8f0;
}

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #06101e 100%) !important;
    border-right: 1px solid rgba(56,189,248,0.12) !important;
}

[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}

/* ── Metrics ─────────────────────────────────────────────── */
[data-testid="stMetricValue"] {
    font-weight: 700 !important;
    font-size: 1.5rem !important;
    color: #f0f9ff !important;
}

[data-testid="stMetricDelta"] {
    font-weight: 500 !important;
}

/* ── Tabs ──────────────────────────────────────────────────── */
[data-testid="stTabs"] button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    color: #94a3b8 !important;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8 !important;
}

/* ── DataFrames ─────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    background: rgba(15, 30, 50, 0.6) !important;
    border: 1px solid rgba(56, 189, 248, 0.15) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ── Buttons ────────────────────────────────────────────── */
[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #0ea5e9, #2563eb) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}

[data-testid="baseButton-primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(14,165,233,0.4) !important;
}

/* ── Expanders ──────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: rgba(15, 25, 45, 0.5) !important;
    border: 1px solid rgba(56,189,248,0.1) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(10px) !important;
}

/* ── Info / Warning / Success boxes ─────────────────────── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left-width: 4px !important;
}

/* ── Markdown headers ───────────────────────────────────── */
h1, h2, h3 { color: #f0f9ff !important; }
h2 { border-bottom: 1px solid rgba(56,189,248,0.2); padding-bottom: 8px; }

/* ── Code blocks ─────────────────────────────────────────── */
code, pre {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}

/* ── Progress bar ────────────────────────────────────────── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #0ea5e9, #6366f1) !important;
}

/* ── Select box ──────────────────────────────────────────── */
[data-testid="stSelectbox"] label,
[data-testid="stRadio"] label {
    color: #94a3b8 !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
}

/* ── Nav page links ──────────────────────────────────────── */
[data-testid="stSidebarNav"] a {
    border-radius: 8px !important;
    margin: 2px 0 !important;
    transition: background 0.15s !important;
}

[data-testid="stSidebarNav"] a:hover {
    background: rgba(56,189,248,0.1) !important;
}

[data-testid="stSidebarNav"] a[aria-current="page"] {
    background: rgba(56,189,248,0.15) !important;
    border-left: 3px solid #38bdf8 !important;
}

/* ── Slider ──────────────────────────────────────────────── */
[data-testid="stSlider"] [class*="thumb"] {
    background: #38bdf8 !important;
}

/* ── Divider ─────────────────────────────────────────────── */
hr {
    border-color: rgba(56,189,248,0.15) !important;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Page Definitions
# ──────────────────────────────────────────────────────────────────────────────
pages = [
    st.Page("pages/v1_fsrcnn_poc.py",       title="V1 — FSRCNN PoC",          icon="🔬"),
    st.Page("pages/v2_baseline_eval.py",    title="V2 — Classical Baselines",  icon="📐"),
    st.Page("pages/v3_bicubic_baseline.py", title="V3 — Dataset Pipeline",     icon="🗄️"),
    st.Page("pages/v4_spatial_ai.py",       title="V4 — Spatial AI (L1)",      icon="🧠"),
    st.Page("pages/v5_edge_loss.py",        title="V5 — Sobel Edge Loss",       icon="⚡"),
    st.Page("pages/v6_spatial_pro.py",      title="V6 — SpatialPro (+VGG)",    icon="🏆"),
    st.Page("pages/v7_temporal_fusion.py",  title="V7 — Temporal Fusion",      icon="🎬"),
]

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar Hero
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="text-align:center; padding: 20px 0 16px 0;">
    <div style="font-size: 2.8rem; margin-bottom: 6px;">🎮</div>
    <div style="font-size: 1.1rem; font-weight: 700; color: #f0f9ff; letter-spacing: 0.02em;">
        SR Evolution Lab
    </div>
    <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">
        Minecraft Game Texture Super-Resolution
    </div>
</div>
<div style="height:1px; background: linear-gradient(90deg, transparent, rgba(56,189,248,0.3), transparent); margin: 0 -16px 16px -16px;"></div>

<div style="font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px; font-weight: 600;">
    Research Timeline
</div>
""", unsafe_allow_html=True)

    timeline = [
        ("🔬", "V1", "FSRCNN Proof of Concept",      "#64748b", "Generic pre-trained CNN baseline"),
        ("📐", "V2", "Classical Baselines",           "#64748b", "Nearest / Bilinear / Bicubic"),
        ("🗄️", "V3", "Dataset Pipeline",              "#64748b", "Minecraft HR↔LR pairs"),
        ("🧠", "V4", "Spatial AI — L1 Loss",          "#3b82f6", "First custom neural network"),
        ("⚡", "V5", "Sobel Edge Loss",                "#8b5cf6", "Sharp texture boundaries"),
        ("🏆", "V6", "SpatialPro — VGG Loss",         "#f59e0b", "16 blocks + perceptual loss"),
        ("🎬", "V7", "Temporal Fusion",               "#10b981", "6-ch optical flow, no flicker"),
    ]

    for icon, ver, name, color, desc in timeline:
        st.markdown(f"""
<div style="display:flex; align-items:flex-start; gap:10px; margin-bottom:10px;">
    <div style="width:28px; height:28px; border-radius:6px;
                background:rgba(255,255,255,0.06); display:flex; align-items:center;
                justify-content:center; font-size:0.9rem; flex-shrink:0;">
        {icon}
    </div>
    <div>
        <div style="font-size:0.82rem; font-weight:600; color:#e2e8f0; line-height:1.2;">
            {ver} · {name}
        </div>
        <div style="font-size:0.7rem; color:#64748b; margin-top:2px;">{desc}</div>
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div style="height:1px; background:rgba(56,189,248,0.15); margin: 12px -16px;"></div>
<div style="font-size:0.7rem; color:#475569; text-align:center;">
    Select a version above to explore
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Navigation
# ──────────────────────────────────────────────────────────────────────────────
pg = st.navigation(pages, position="sidebar")
pg.run()

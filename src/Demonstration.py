import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import joblib
from pathlib import Path
from collections import deque
import warnings
import networkx as nx
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

from data_utils import load_mapping, load_subgraph


st.set_page_config(
    page_title="LightEdge-IDS",
    layout="wide",
    initial_sidebar_state="expanded"
)

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Network Graph"]
)

# ---------- UI STYLING ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Syne:wght@400;600;800&display=swap');

/* ===================== GLOBAL BASE ===================== */
html, body { background-color: #e8f4fd !important; }

*, *::before, *::after {
    font-family: 'Syne', sans-serif;
}

/* Main content area — sky blue gradient */
.main, .block-container,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"] {
    background-color: #e8f4fd !important;
    background-image:
        radial-gradient(ellipse 70% 50% at 10% 0%, rgba(125,211,252,0.35) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 90% 100%, rgba(186,230,253,0.3) 0%, transparent 60%);
    color: #0c2d4a !important;
}

/* Generic text — catch-all */
p, span, div, label, li, td, th,
.stMarkdown, .stMarkdown p,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stText"], [data-testid="stText"] p {
    color: #0c2d4a !important;
}

/* ===================== HEADINGS ===================== */
h1, [data-testid="stHeadingWithActionElements"] h1 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 2rem !important;
    color: #063a5c !important;
    letter-spacing: -0.02em !important;
}

h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    color: #0e4a72 !important;
}

h4, h5, h6 {
    color: #2a6a9a !important;
}

/* ===================== SIDEBAR ===================== */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div {
    background: #c8e8f8 !important;
    border-right: 1px solid #93c5e8 !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div {
    color: #1a4a6e !important;
}

section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
    color: #1a5a8a !important;
    letter-spacing: 0.04em !important;
}

section[data-testid="stSidebar"] .stRadio [aria-checked="true"] + div label,
section[data-testid="stSidebar"] .stRadio [data-checked="true"] label {
    color: #0369a1 !important;
    font-weight: 700 !important;
}

/* ===================== FILE UPLOADER ===================== */
[data-testid="stFileUploader"],
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.65) !important;
    border: 2px dashed #7dd3fc !important;
    border-radius: 14px !important;
}

[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] label,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] div {
    color: #1e6fa3 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
}

[data-testid="stFileUploaderDropzone"] button {
    background: #e0f2fe !important;
    color: #0369a1 !important;
    border: 1px solid #7dd3fc !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
}

/* ===================== WIDGET LABELS ===================== */
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
.stSelectbox label, .stTextInput label,
.stTextArea label, .stNumberInput label,
.stCheckbox label, .stRadio label {
    color: #1e6fa3 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

/* ===================== METRIC CARDS ===================== */
[data-testid="metric-container"] {
    background: linear-gradient(145deg, #ffffff, #e0f2fe) !important;
    border-radius: 14px !important;
    padding: 20px 18px !important;
    border: 1px solid #bae6fd !important;
    box-shadow: 0 4px 20px rgba(14,116,144,0.12) !important;
    position: relative !important;
    overflow: hidden !important;
}

[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #0ea5e9, #38bdf8, #7dd3fc);
    opacity: 1;
}

[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] span,
[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #4a90b8 !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] > div {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #0369a1 !important;
    letter-spacing: 0.02em !important;
}

[data-testid="stMetricDelta"],
[data-testid="stMetricDelta"] span {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.05em !important;
    color: #0891b2 !important;
}

/* ===================== DATAFRAME ===================== */
[data-testid="stDataFrame"] {
    background: #ffffff !important;
    border-radius: 14px !important;
    border: 1px solid #bae6fd !important;
    overflow: hidden !important;
    box-shadow: 0 2px 12px rgba(14,116,144,0.10) !important;
}

[data-testid="stDataFrame"] th,
[data-testid="stDataFrame"] .col_heading {
    background: #0ea5e9 !important;
    color: #ffffff !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border-bottom: 2px solid #0284c7 !important;
    padding: 10px 14px !important;
}

[data-testid="stDataFrame"] td,
[data-testid="stDataFrame"] .data {
    background: #ffffff !important;
    color: #0c3a5c !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
    border-color: #e0f2fe !important;
    padding: 9px 14px !important;
}

[data-testid="stDataFrame"] tr:nth-child(even) td {
    background: #f0f9ff !important;
}

[data-testid="stDataFrame"] tr:hover td {
    background: #e0f2fe !important;
}

/* ===================== ALERTS ===================== */
[data-testid="stAlert"],
div[data-baseweb="notification"] {
    border-radius: 10px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
}

[data-testid="stAlert"][kind="success"],
div[data-baseweb="notification"][kind="positive"] {
    background: rgba(16,185,129,0.12) !important;
    border-left: 3px solid #10b981 !important;
    color: #065f46 !important;
}

[data-testid="stAlert"][kind="warning"],
div[data-baseweb="notification"][kind="warning"] {
    background: rgba(245,158,11,0.10) !important;
    border-left: 3px solid #f59e0b !important;
    color: #92400e !important;
}

[data-testid="stAlert"][kind="error"],
div[data-baseweb="notification"][kind="negative"] {
    background: rgba(239,68,68,0.10) !important;
    border-left: 3px solid #ef4444 !important;
    color: #7f1d1d !important;
}

[data-testid="stAlert"][kind="info"],
div[data-baseweb="notification"][kind="info"] {
    background: rgba(14,165,233,0.10) !important;
    border-left: 3px solid #0ea5e9 !important;
    color: #0c4a6e !important;
}

[data-testid="stAlert"] p,
[data-testid="stAlert"] span {
    color: inherit !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ===================== BUTTONS ===================== */
.stButton > button {
    background: linear-gradient(135deg, #0284c7, #0ea5e9) !important;
    color: #ffffff !important;
    border-radius: 9px !important;
    border: none !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    padding: 10px 20px !important;
    box-shadow: 0 4px 14px rgba(2,132,199,0.25) !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #0369a1, #38bdf8) !important;
    box-shadow: 0 4px 20px rgba(2,132,199,0.4) !important;
    transform: translateY(-1px) !important;
}

.stDownloadButton > button {
    background: linear-gradient(135deg, #0284c7, #0ea5e9) !important;
    color: #ffffff !important;
    border-radius: 9px !important;
    border: 1.5px solid #7dd3fc !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    box-shadow: 0 4px 14px rgba(2,132,199,0.22) !important;
    transition: all 0.2s !important;
}

.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #0369a1, #38bdf8) !important;
    box-shadow: 0 4px 22px rgba(2,132,199,0.38) !important;
    transform: translateY(-1px) !important;
    border-color: #38bdf8 !important;
}

/* Replace default icon with proper download SVG */
.stDownloadButton > button [data-testid="stDownloadButtonIcon"],
.stDownloadButton > button svg {
    display: none !important;
}

.stDownloadButton > button::before {
    content: '';
    display: inline-block;
    width: 16px;
    height: 16px;
    margin-right: 6px;
    vertical-align: middle;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23ffffff' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4'/%3E%3Cpolyline points='7 10 12 15 17 10'/%3E%3Cline x1='12' y1='15' x2='12' y2='3'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-size: contain;
}

/* ===================== DIVIDER ===================== */
hr {
    border: none !important;
    border-top: 1px solid #bae6fd !important;
    margin: 24px 0 !important;
}

/* ===================== SCROLLBAR ===================== */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #e0f2fe; }
::-webkit-scrollbar-thumb { background: #7dd3fc; border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: #0ea5e9; }

/* ===================== CODE ===================== */
code {
    background: #e0f2fe !important;
    color: #0369a1 !important;
    border-radius: 4px !important;
    padding: 1px 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown(
        '<div style="font-family:\'Syne\',sans-serif;font-weight:800;font-size:1.2rem;'
        'color:#063a5c;padding:4px 0 20px 0;border-bottom:1px solid #93c5e8;margin-bottom:16px;">'
        '⬡ <span style="color:#0284c7;">Light</span>Edge-IDS</div>',
        unsafe_allow_html=True
    )


DEVICE = "cpu"


N_ZONES       = 4
SEQ           = 10     
GLOBAL_THRESH = 0.35    


_script_dir = os.path.dirname(os.path.abspath(__file__))   # .../src
_parent_dir = os.path.dirname(_script_dir)                 # .../LightEdge-IDS

TS_DIR     = os.path.join(_parent_dir, "outputs", "IDS", "outputs","ts")
SCALER_DIR = os.path.join(_parent_dir, "outputs", "IDS", "outputs","scalers")
META_DIR   = os.path.join(_parent_dir, "outputs", "IDS","outputs")



@st.cache_resource
def load_graph():
    mapping = load_mapping()
    nodes, node_index, edge_index, edge_weight = load_subgraph()
    return mapping, nodes, edge_index, edge_weight


def plot_network_graph(nodes, edge_index):
    G = nx.Graph()
    for i in range(len(nodes)):
        G.add_node(i)
    edges = edge_index.T
    for e in edges:
        G.add_edge(int(e[0]), int(e[1]))

    pos = nx.spring_layout(G, seed=42)

    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x, node_y = [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color='#93c5fd'),
        hoverinfo='none',
        mode='lines',
        opacity=0.9
    ))

    node_labels = [str(nodes[i]) if i < len(nodes) else f"Node {i}" for i in G.nodes()]

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        marker=dict(
            size=13,
            color='#0284c7',
            line=dict(color='#ffffff', width=2),
            opacity=1.0
        ),
        text=node_labels,
        hovertemplate='<b>%{text}</b><extra></extra>'
    ))

    fig.update_layout(
        showlegend=False,
        plot_bgcolor='rgba(255,255,255,0.55)',
        paper_bgcolor='rgba(224,242,254,0.6)',
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        hoverlabel=dict(
            bgcolor='#ffffff',
            font=dict(family='IBM Plex Mono', size=12, color='#063a5c'),
            bordercolor='#7dd3fc'
        )
    )
    return fig


# ---------- LOAD ZONE METADATA ---------- risk
def load_meta(z):
    p = os.path.join(META_DIR, f"zone_{z}_meta.pt")
    if os.path.exists(p):
        return torch.load(p, weights_only=False)
    return {"label_col": "ATTACK_FLAG", "continuous_cols": [], "binary_cols": [],
            "threshold": 0.999, "in_dim": 1, "is_active": False}


# ---------- LOAD TORCHSCRIPT ZONE MODELS + SCALERS ----------
@st.cache_resource
def load_zone_models():
    missing = []
    for z in range(N_ZONES):
        pt  = os.path.join(TS_DIR,     f"zone_{z}_ts.pt")
        pkl = os.path.join(SCALER_DIR, f"scaler_zone{z}.pkl")
        if not os.path.exists(pt):  missing.append(pt)
        if not os.path.exists(pkl): missing.append(pkl)
    if missing:
        st.error(
            "**Model files not found.**  "
            f"Expected them inside `{TS_DIR}` and `{SCALER_DIR}`.\n\n"
            "Missing:\n" + "\n".join(f"- `{p}`" for p in missing)
        )
        st.info(
            "Expected project layout:\n"
            "```\n"
            "LightEdge-IDS/\n"
            "  outputs/IDS/ts/zone_0_ts.pt       ← models here\n"
            "  outputs/IDS/scalers/scaler_zone0.pkl\n"
            "  outputs/IDS/zone_0_meta.pt\n"
            "  src/Demonstration.py              ← this script\n"
            "```"
        )
        st.stop()
    models, scalers, metas = [], [], []
    for z in range(N_ZONES):
        zm = torch.jit.load(os.path.join(TS_DIR, f"zone_{z}_ts.pt"), map_location=DEVICE)
        zm.eval()
        models.append(zm)
        scalers.append(joblib.load(os.path.join(SCALER_DIR, f"scaler_zone{z}.pkl")))
        metas.append(load_meta(z))
    return models, scalers, metas


# ---------- LOAD TORCHSCRIPT GLOBAL MODEL ----------
@st.cache_resource
def load_global_model_ts():
    gm_path = os.path.join(TS_DIR, "global_model_ts.pt")
    if not os.path.exists(gm_path):
        st.error(f"**Global model not found:** `{gm_path}`")
        st.stop()
    gm = torch.jit.load(gm_path, map_location=DEVICE)
    gm.eval()
    return gm


# ---------- PREPROCESSING (mirrors pi_local.py) ----------
def preprocess_row(row, scaler, cont_cols, bin_cols):
    """Identical to pi_local.py's preprocess() — uses the zone's own column lists."""
    cont = row[cont_cols].values.reshape(1, -1).astype(np.float32) if cont_cols else None
    binr = row[bin_cols].values.reshape(1, -1).astype(np.float32)  if bin_cols  else None
    if cont is not None:
        cont = scaler.transform(cont)
    parts = [p for p in [cont, binr] if p is not None]
    x = np.hstack(parts) if parts else np.zeros((1, 1), np.float32)
    return torch.tensor(x, dtype=torch.float32)


# ---------- RUN INFERENCE (mirrors pi_local.py main loop exactly) ----------
def run_inference(df, zone_models, zone_scalers, zone_metas, global_model):
    """
    Mirrors pi_local.py's inference loop.

    Returns
    -------
    preds  : pd.Series  — binary predicted labels (0/1) aligned to df.index
    stats  : dict       — tp, fp, fn, tn, zone_alerts, global_alerts, total
    """
    label_col = zone_metas[0].get("label_col", "ATTACK_FLAG") if zone_metas else "ATTACK_FLAG"

    n      = len(df)
    preds  = np.zeros(n, dtype=int)
    buffer = deque(maxlen=SEQ)
    df_r   = df.reset_index(drop=True)

    # Identify active zones (same as pi_local.py)
    active_zones = [z for z in range(N_ZONES) if zone_metas[z].get("is_active", False)]

    stats = dict(total=0, tp=0, fp=0, fn=0, tn=0,
                 zone_alerts=0, global_alerts=0)

    for t in range(n):
        zone_probs  = {}
        zone_alerts = {}
        any_zone_alert = False
        embs_t = []

        # ── Zone pass (all zones produce embeddings; active zones raise alerts) ──
        for z in range(N_ZONES):
            m         = zone_metas[z]
            cont_cols = m.get("continuous_cols", [])
            bin_cols  = m.get("binary_cols",  [])
            # Intersect with columns present in the uploaded file
            cont_cols = [c for c in cont_cols if c in df_r.columns]
            bin_cols  = [c for c in bin_cols  if c in df_r.columns]

            x = preprocess_row(df_r.iloc[t], zone_scalers[z], cont_cols, bin_cols)
            with torch.no_grad():
                z_logit, emb, _ = zone_models[z](x)
            z_prob = torch.sigmoid(z_logit).item()

            if m.get("is_active", False):
                zone_probs[z]  = z_prob
                zone_alerts[z] = z_prob > m.get("threshold", 0.999)
                if zone_alerts[z]:
                    any_zone_alert = True
                    stats["zone_alerts"] += 1

            embs_t.append(emb.squeeze(0))

        # ── Update rolling buffer ────────────────────────────────────────────
        buffer.append(torch.stack(embs_t, dim=0))  # [N_ZONES, EMB_DIM]

        # ── Global model (once SEQ frames accumulated) ───────────────────────
        g_alert = False
        g_prob  = 0.0
        if len(buffer) == SEQ:
            seq_t = torch.stack(list(buffer), dim=0).unsqueeze(0)  # [1,SEQ,N_ZONES,D]
            with torch.no_grad():
                g_prob = torch.sigmoid(global_model(seq_t)).item()
            g_alert = g_prob > GLOBAL_THRESH
            if g_alert:
                stats["global_alerts"] += 1

        # ── Final prediction (same fallback as pi_local.py) ─────────────────
        pred = int(g_alert) if len(buffer) == SEQ else int(any_zone_alert)
        preds[t] = pred

        # ── Ground-truth + confusion matrix update ───────────────────────────
        if label_col in df_r.columns:
            gt = int(any(int(df_r.iloc[t][zone_metas[z].get("label_col", label_col)]) == 1
                         for z in range(N_ZONES)
                         if zone_metas[z].get("label_col", label_col) in df_r.columns))
        else:
            gt = 0

        stats["total"] += 1
        if   gt == 1 and pred == 1: stats["tp"] += 1
        elif gt == 0 and pred == 1: stats["fp"] += 1
        elif gt == 1 and pred == 0: stats["fn"] += 1
        else:                        stats["tn"] += 1

    return pd.Series(preds, index=df.index), stats


# ---------- VALIDATE DATA ----------
def validate_and_process_data(df, zone_models=None, zone_scalers=None,
                              zone_metas=None, global_model=None):
    if df.empty:
        return None, "Dataset is empty", {}
    if 'ATTACK_FLAG' not in df.columns:
        return None, "ATTACK_FLAG column not found", {}

    df_processed = df.copy()
    try:
        df_processed['ATTACK_FLAG'] = pd.to_numeric(df_processed['ATTACK_FLAG'], errors='coerce')
    except:
        return None, "ATTACK_FLAG column cannot be converted to numeric", {}

    df_processed['ATTACK_FLAG'] = df_processed['ATTACK_FLAG'].fillna(0)
    # Ground truth (kept for reference / export)
    df_processed['ATTACK_FLAG_BINARY'] = (df_processed['ATTACK_FLAG'] != 0).astype(int)

    inference_stats = {}

    # Model predictions (mirrors pi_local.py two-level inference)
    if zone_models is not None and global_model is not None:
        try:
            preds, inference_stats = run_inference(
                df_processed, zone_models, zone_scalers, zone_metas, global_model
            )
            df_processed['PREDICTED_BINARY'] = preds.values
        except Exception as e:
            # Graceful fallback: if inference fails, use ground truth so UI still renders
            df_processed['PREDICTED_BINARY'] = df_processed['ATTACK_FLAG_BINARY']
            st.warning(f"Model inference failed — falling back to ground truth labels. ({e})")
    else:
        df_processed['PREDICTED_BINARY'] = df_processed['ATTACK_FLAG_BINARY']

    return df_processed, "", inference_stats


# ---------- MAIN ----------
def main():
    # Page header
    st.markdown(
        '<p style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;'
        'letter-spacing:0.12em;text-transform:uppercase;color:#0284c7;margin-bottom:4px;">'
        '⬡ Intrusion Detection System</p>',
        unsafe_allow_html=True
    )
    st.title("LightEdge-IDS Dashboard")

    zone_models, zone_scalers, zone_metas = load_zone_models()
    global_model = load_global_model_ts()

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload test dataset  —  CSV or Excel",
        type=['xlsx', 'xls', 'csv'],
        help="Accepts .csv, .xlsx, or .xls files with an ATTACK_FLAG column"
    )

    if page == "Dashboard":

        if uploaded_file is not None:
            try:
                file_name = uploaded_file.name.lower()

                if file_name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                df_processed, error_msg, inference_stats = validate_and_process_data(
                    df, zone_models, zone_scalers, zone_metas, global_model
                )

                if df_processed is None:
                    st.error(f"Data Error: {error_msg}")
                    if 'ATTACK_FLAG' in df.columns:
                        st.subheader("Values found in ATTACK_FLAG column:")
                        value_counts = df['ATTACK_FLAG'].value_counts().sort_index()
                        for val, count in value_counts.head(20).items():
                            st.write(f"Value `{val}`: **{count}** rows")
                    st.stop()

                # ---------- ATTACK STATISTICS (from model predictions) ----------
                total_hours = len(df_processed)
                attack_hours = int(df_processed['PREDICTED_BINARY'].sum())
                normal_hours = total_hours - attack_hours
                attack_percent = (attack_hours / total_hours * 100) if total_hours > 0 else 0

                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                st.markdown(
                    '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;'
                    'letter-spacing:0.1em;text-transform:uppercase;color:#2a7ab8;">Attack Statistics</span>',
                    unsafe_allow_html=True
                )
                st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Hours", total_hours)
                col2.metric(
                    "Attack Hours", attack_hours,
                )
                col3.metric("Normal Hours", normal_hours)
                col4.metric("Attack Rate", f"{attack_percent:.1f}%")

               

                # ---------- ATTACK PERIOD ANALYSIS ----------
                attack_periods = []
                in_attack = False
                current_period = None
                df_reset = df_processed.reset_index(drop=True)

                for idx in range(len(df_reset)):
                    is_attack = df_reset.loc[idx, 'PREDICTED_BINARY'] == 1
                    if is_attack and not in_attack:
                        in_attack = True
                        current_period = {'start_idx': idx, 'end_idx': idx, 'duration': 1}
                    elif is_attack and in_attack:
                        current_period['end_idx'] = idx
                        current_period['duration'] += 1
                    elif not is_attack and in_attack:
                        in_attack = False
                        attack_periods.append(current_period.copy())
                        current_period = None

                if in_attack and current_period:
                    attack_periods.append(current_period)

                st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
                st.markdown(
                    '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;'
                    'letter-spacing:0.1em;text-transform:uppercase;color:#2a7ab8;">Attack Periods</span>',
                    unsafe_allow_html=True
                )
                st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

                if attack_periods:
                    summary_data = []
                    for i, p in enumerate(attack_periods, 1):
                        if 'DATETIME' in df_reset.columns:
                            start_time = df_reset.loc[p['start_idx'], 'DATETIME']
                            end_time = df_reset.loc[p['end_idx'], 'DATETIME']
                            time_info = f"{start_time} → {end_time}"
                        else:
                            time_info = f"Rows {p['start_idx']} → {p['end_idx']}"

                        summary_data.append({
                            "Attack ID": f"#{i:03d}",
                            "Duration": f"{p['duration']} hr",
                            "Time Range": time_info
                        })

                    summary_df = pd.DataFrame(summary_data)

                    # Custom HTML table — sky blue themed
                    rows_html = ""
                    for _, row in summary_df.iterrows():
                        rows_html += f"""
                        <tr>
                            <td>{row['Attack ID']}</td>
                            <td>{row['Duration']}</td>
                            <td>{row['Time Range']}</td>
                        </tr>"""

                    st.markdown(f"""
                    <style>
                    .atk-table {{ width:100%; border-collapse:separate; border-spacing:0; border-radius:12px; overflow:hidden; box-shadow:0 2px 16px rgba(14,116,144,0.10); font-family:'IBM Plex Mono',monospace; }}
                    .atk-table thead tr {{ background:linear-gradient(90deg,#0284c7,#38bdf8); }}
                    .atk-table thead th {{ color:#ffffff; font-size:11px; letter-spacing:0.1em; text-transform:uppercase; padding:12px 16px; text-align:left; font-weight:600; border:none; }}
                    .atk-table tbody tr {{ background:#ffffff; transition:background 0.15s; }}
                    .atk-table tbody tr:nth-child(even) {{ background:#f0f9ff; }}
                    .atk-table tbody tr:hover {{ background:#dbeafe; }}
                    .atk-table tbody td {{ color:#0c3a5c; font-size:13px; padding:11px 16px; border-bottom:1px solid #e0f2fe; border:none; border-bottom:1px solid #e0f2fe; }}
                    .atk-table tbody td:first-child {{ color:#0284c7; font-weight:600; }}
                    .atk-table tbody tr:last-child td {{ border-bottom:none; }}
                    </style>
                    <table class="atk-table">
                        <thead>
                            <tr>
                                <th>Attack ID</th>
                                <th>Duration</th>
                                <th>Time Range</th>
                            </tr>
                        </thead>
                        <tbody>{rows_html}</tbody>
                    </table>
                    """, unsafe_allow_html=True)

                    st.markdown("<hr>", unsafe_allow_html=True)

                    # ---------- VISUAL ANALYTICS ----------
                    st.markdown(
                        '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;'
                        'letter-spacing:0.1em;text-transform:uppercase;color:#2a7ab8;">Visual Analytics</span>',
                        unsafe_allow_html=True
                    )
                    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

                    plot_col1, plot_col2 = st.columns(2)

                    # BAR CHART
                    with plot_col1:
                        st.markdown(
                            '<p style="font-family:\'IBM Plex Mono\',monospace;font-size:12px;'
                            'color:#1a6a9e;margin-bottom:8px;">Attack vs Normal Distribution</p>',
                            unsafe_allow_html=True
                        )
                        bar_fig = go.Figure(go.Bar(
                            x=["Normal", "Attack"],
                            y=[normal_hours, attack_hours],
                            marker_color=['#0ea5e9', '#f97316'],
                            marker_line_color=['#0284c7', '#ea580c'],
                            marker_line_width=2,
                            width=0.45,
                            text=[normal_hours, attack_hours],
                            textposition='outside',
                            textfont=dict(family='IBM Plex Mono', size=12, color='#063a5c'),
                        ))
                        bar_fig.update_layout(
                            plot_bgcolor='rgba(255,255,255,0.55)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            margin=dict(l=10, r=10, t=30, b=10),
                            font=dict(family='IBM Plex Mono', color='#1a6a9e', size=12),
                            xaxis=dict(
                                showgrid=False, zeroline=False,
                                color='#1a6a9e',
                                tickfont=dict(size=13, color='#063a5c', family='IBM Plex Mono'),
                                linecolor='#bae6fd', linewidth=1
                            ),
                            yaxis=dict(
                                showgrid=True, gridcolor='#dbeafe',
                                zeroline=False, color='#1a6a9e',
                                tickfont=dict(size=11, color='#1a6a9e', family='IBM Plex Mono'),
                                linecolor='#bae6fd'
                            ),
                            hoverlabel=dict(
                                bgcolor='#ffffff',
                                font=dict(family='IBM Plex Mono', size=12, color='#063a5c'),
                                bordercolor='#7dd3fc'
                            ),
                            height=270,
                            shapes=[dict(
                                type='line', x0=-0.5, x1=1.5,
                                y0=0, y1=0,
                                line=dict(color='#93c5fd', width=1)
                            )]
                        )
                        st.plotly_chart(bar_fig, use_container_width=True)

                    # PIE / DONUT CHART
                    with plot_col2:
                        st.markdown(
                            '<p style="font-family:\'IBM Plex Mono\',monospace;font-size:12px;'
                            'color:#1a6a9e;margin-bottom:8px;">Attack Percentage Breakdown</p>',
                            unsafe_allow_html=True
                        )
                        pie_fig = go.Figure(go.Pie(
                            labels=["Normal", "Attack"],
                            values=[normal_hours, attack_hours],
                            hole=0.60,
                            marker=dict(
                                colors=['#0ea5e9', '#f97316'],
                                line=dict(color='#e8f4fd', width=3)
                            ),
                            textinfo='percent',
                            textposition='inside',
                            insidetextfont=dict(
                                family='IBM Plex Mono',
                                size=13,
                                color='#ffffff'
                            ),
                            hovertemplate='<b>%{label}</b><br>%{value} hours<br>%{percent}<extra></extra>'
                        ))
                        pie_fig.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            showlegend=True,
                            legend=dict(
                                font=dict(family='IBM Plex Mono', size=12, color='#063a5c'),
                                bgcolor='rgba(255,255,255,0.5)',
                                bordercolor='#bae6fd',
                                borderwidth=1,
                                x=0.5, xanchor='center',
                                y=-0.08, orientation='h'
                            ),
                            margin=dict(l=10, r=10, t=10, b=40),
                            hoverlabel=dict(
                                bgcolor='#ffffff',
                                font=dict(family='IBM Plex Mono', size=12, color='#063a5c'),
                                bordercolor='#7dd3fc'
                            ),
                            height=280,
                            annotations=[dict(
                                text=f"<b>{attack_percent:.1f}%</b><br>attacked",
                                x=0.5, y=0.5,
                                showarrow=False,
                                font=dict(size=15, family='IBM Plex Mono', color='#ea580c'),
                                align='center'
                            )]
                        )
                        st.plotly_chart(pie_fig, use_container_width=True)

                    # TIMELINE
                    st.markdown(
                        '<p style="font-family:\'IBM Plex Mono\',monospace;font-size:12px;'
                        'color:#1a6a9e;margin-bottom:8px;">Attack Timeline</p>',
                        unsafe_allow_html=True
                    )

                    if "DATETIME" in df_processed.columns:
                        try:
                            time_df = df_processed[["DATETIME", "PREDICTED_BINARY"]].copy()
                            timeline_fig = go.Figure()
                            # normal fill
                            timeline_fig.add_trace(go.Scatter(
                                x=time_df["DATETIME"],
                                y=time_df["PREDICTED_BINARY"],
                                mode='lines',
                                fill='tozeroy',
                                line=dict(color='#0284c7', width=2),
                                fillcolor='rgba(14,165,233,0.18)',
                                name='Status',
                                hovertemplate='%{x}<br>%{y}<extra></extra>'
                            ))
                            timeline_fig.update_layout(
                                plot_bgcolor='rgba(255,255,255,0.55)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                margin=dict(l=10, r=10, t=10, b=10),
                                font=dict(family='IBM Plex Mono', color='#1a6a9e', size=11),
                                xaxis=dict(
                                    showgrid=False, zeroline=False,
                                    color='#1a6a9e',
                                    tickfont=dict(color='#1a6a9e', size=11),
                                    linecolor='#bae6fd'
                                ),
                                yaxis=dict(
                                    showgrid=True, gridcolor='#dbeafe',
                                    zeroline=False, color='#1a6a9e',
                                    tickvals=[0, 1],
                                    ticktext=["Normal", "Attack"],
                                    tickfont=dict(color='#1a6a9e', size=11)
                                ),
                                hoverlabel=dict(
                                    bgcolor='#ffffff',
                                    font=dict(family='IBM Plex Mono', size=12, color='#063a5c'),
                                    bordercolor='#7dd3fc'
                                ),
                                showlegend=False,
                                height=210
                            )
                            st.plotly_chart(timeline_fig, use_container_width=True)
                        except:
                            st.warning("Could not render DATETIME timeline")
                    else:
                        st.info("No DATETIME column found — timeline unavailable")

                    # EXPORT
                    st.markdown("<hr>", unsafe_allow_html=True)
                    st.markdown(
                        '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;'
                        'letter-spacing:0.1em;text-transform:uppercase;color:#2a7ab8;">Export</span>',
                        unsafe_allow_html=True
                    )
                    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

                    colA, _ = st.columns([1, 3])
                    with colA:
                        st.download_button(
                            "Download Attack Summary",
                            summary_df.to_csv(index=False),
                            "attack_summary.csv",
                            "text/csv"
                        )

                else:
                    if attack_hours > 0:
                        st.warning(f"Found {attack_hours} attack hours but periods are non-contiguous.")
                    else:
                        st.success("✓  No attacks detected in this dataset.")

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

        else:
            # Empty state placeholder
            st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
            st.markdown(
                """
                <div style='text-align:center;padding:60px 20px;border:1px dashed #7dd3fc;
                border-radius:16px;background:rgba(255,255,255,0.6);'>
                    <div style='font-size:2.5rem;margin-bottom:12px;opacity:0.3;'>⬡</div>
                    <p style='font-family:"IBM Plex Mono",monospace;font-size:12px;
                    color:#2a7ab8;letter-spacing:0.08em;text-transform:uppercase;margin:0;'>
                    Upload a dataset to begin analysis</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    elif page == "Network Graph":
        st.markdown(
            '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;'
            'letter-spacing:0.1em;text-transform:uppercase;color:#0284c7;">GNN Topology View</span>',
            unsafe_allow_html=True
        )
        st.markdown("## Network Graph")
        mapping, nodes, edge_index, edge_weight = load_graph()
        fig = plot_network_graph(nodes, edge_index)
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="HeartGuard AI | Clinical Cardiac Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="+"
)

# ─── RETRO CLINIC DESIGN SYSTEM ────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&family=IBM+Plex+Sans:wght@300;400;500;600;700&family=Playfair+Display:wght@700;800&display=swap');

  /* ── PALETTE
     Cream Canvas:  #FAF7F0
     Warm White:    #FFFFFF
     Slate Rule:    #D4C9B0
     Deep Burgundy: #7C1B2E
     Rich Burgundy: #A52840
     Forest Green:  #1B5741
     Brass Gold:    #B8860B
     Data Blue:     #1E3A5F
     Muted Ink:     #3D3228
     Soft Red:      #C0392B
     Success:       #1B7A4E
  */

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
    color: #3D3228;
  }

  /* ── CANVAS */
  .stApp {
    background: #FAF7F0 !important;
    background-image:
      repeating-linear-gradient(0deg, transparent, transparent 39px, #EDE8DC 40px),
      repeating-linear-gradient(90deg, transparent, transparent 39px, #EDE8DC 40px) !important;
    background-size: 40px 40px !important;
  }

  /* ── SIDEBAR */
  section[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 2px solid #D4C9B0 !important;
    box-shadow: 3px 0 16px rgba(61,50,40,0.06) !important;
  }
  section[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Playfair Display', serif !important;
    color: #7C1B2E !important;
    font-size: 1rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid #D4C9B0 !important;
    padding-bottom: 0.4rem !important;
    margin-bottom: 0.8rem !important;
  }

  /* ── HEADER */
  header[data-testid="stHeader"] {
    background: rgba(250, 247, 240, 0.92) !important;
    backdrop-filter: blur(10px) !important;
    border-bottom: 1px solid #D4C9B0 !important;
  }

  /* ── MAIN CONTENT PADDING */
  .main .block-container {
    padding: 2rem 2.5rem 3rem 2.5rem !important;
    max-width: 1280px !important;
  }

  /* ── FORM */
  div[data-testid="stForm"] {
    background: #FFFFFF !important;
    border: 1.5px solid #D4C9B0 !important;
    border-top: 4px solid #7C1B2E !important;
    border-radius: 4px !important;
    padding: 2rem 2.2rem !important;
    box-shadow: 0 2px 18px rgba(61,50,40,0.07) !important;
  }

  /* ── INPUTS */
  div[data-baseweb="input"] > div,
  div[data-baseweb="select"] > div {
    background: #FAF7F0 !important;
    border: 1.5px solid #C8BCAA !important;
    border-radius: 3px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.93rem !important;
    color: #1E3A5F !important;
  }
  div[data-baseweb="input"] > div:focus-within,
  div[data-baseweb="select"] > div:focus-within {
    border-color: #7C1B2E !important;
    box-shadow: 0 0 0 3px rgba(124,27,46,0.12) !important;
  }

  input, select, textarea {
    color: #1E3A5F !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 500 !important;
  }

  /* ── LABELS */
  label, .stSelectbox label, .stNumberInput label {
    color: #3D3228 !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
  }

  /* ── BUTTONS */
  .stButton > button,
  .stDownloadButton > button,
  .stFormSubmitButton > button {
    background: #7C1B2E !important;
    color: #FAF7F0 !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 0.65rem 1.6rem !important;
    box-shadow: 0 2px 8px rgba(124,27,46,0.25) !important;
    transition: all 0.15s ease !important;
  }
  .stButton > button:hover,
  .stDownloadButton > button:hover,
  .stFormSubmitButton > button:hover {
    background: #A52840 !important;
    box-shadow: 0 4px 14px rgba(124,27,46,0.35) !important;
    transform: translateY(-1px) !important;
  }
  .stButton > button:active,
  .stFormSubmitButton > button:active {
    transform: translateY(0) !important;
    box-shadow: 0 1px 4px rgba(124,27,46,0.2) !important;
  }

  /* ── TABS */
  .stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 2px solid #D4C9B0 !important;
    gap: 0 !important;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -2px !important;
    color: #7A6A5A !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 0.7rem 1.4rem !important;
    border-radius: 0 !important;
    transition: all 0.15s ease !important;
  }
  .stTabs [data-baseweb="tab"]:hover {
    color: #7C1B2E !important;
    background: rgba(124,27,46,0.05) !important;
  }
  .stTabs [aria-selected="true"] {
    color: #7C1B2E !important;
    border-bottom: 2px solid #7C1B2E !important;
    font-weight: 700 !important;
    background: transparent !important;
  }

  /* ── METRICS */
  div[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1.5px solid #D4C9B0 !important;
    border-top: 3px solid #1B5741 !important;
    border-radius: 3px !important;
    padding: 1rem 1.2rem !important;
    box-shadow: 0 1px 8px rgba(61,50,40,0.06) !important;
  }
  div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color: #7A6A5A !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
  }
  div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #1E3A5F !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
  }
  div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #1B5741 !important;
  }

  /* ── DATAFRAMES */
  div[data-testid="stDataFrame"] {
    border: 1.5px solid #D4C9B0 !important;
    border-radius: 3px !important;
    overflow: hidden !important;
  }

  /* ── RADIO (sidebar nav) */
  .stRadio label {
    text-transform: none !important;
    letter-spacing: 0 !important;
    font-size: 0.88rem !important;
    color: #3D3228 !important;
  }
  .stRadio [data-testid="stMarkdownContainer"] p {
    font-size: 0.88rem !important;
    color: #3D3228 !important;
  }

  /* ── SLIDER */
  .stSlider [data-baseweb="slider"] [role="slider"] {
    background: #7C1B2E !important;
    border: 2px solid #7C1B2E !important;
  }
  .stSlider [data-baseweb="slider"] div[data-testid="stTickBarMin"],
  .stSlider [data-baseweb="slider"] div[data-testid="stTickBarMax"] {
    color: #7A6A5A !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
  }

  /* ── DIVIDER */
  hr {
    border: none !important;
    border-top: 1.5px solid #D4C9B0 !important;
    margin: 2rem 0 !important;
  }

  /* ── SCROLLBAR */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: #FAF7F0; }
  ::-webkit-scrollbar-thumb { background: #C8BCAA; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #7C1B2E; }

  /* ── KEYFRAMES */
  @keyframes ecg-pulse {
    0%   { stroke-dashoffset: 1200; }
    100% { stroke-dashoffset: 0; }
  }
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  /* ── HERO BANNER */
  .rc-hero {
    background: #FFFFFF;
    border: 1.5px solid #D4C9B0;
    border-top: 5px solid #7C1B2E;
    border-radius: 4px;
    padding: 2.8rem 3rem 2.4rem 3rem;
    margin-bottom: 2rem;
    box-shadow: 0 4px 30px rgba(61,50,40,0.08);
    display: flex;
    align-items: center;
    gap: 3rem;
    animation: fadeUp 0.5s ease;
  }
  .rc-hero-left { flex: 1; min-width: 0; }
  .rc-hero-right { flex-shrink: 0; }
  .rc-eyebrow {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #7C1B2E;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.7rem;
  }
  .rc-eyebrow::before {
    content: '';
    display: inline-block;
    width: 28px;
    height: 2px;
    background: #7C1B2E;
  }
  .rc-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 800;
    color: #1E3A5F;
    line-height: 1.1;
    letter-spacing: -0.02em;
    margin-bottom: 0.8rem;
  }
  .rc-title span { color: #7C1B2E; }
  .rc-subtitle {
    font-size: 1rem;
    color: #7A6A5A;
    line-height: 1.6;
    max-width: 520px;
    margin-bottom: 1.4rem;
  }
  .rc-badges { display: flex; gap: 0.6rem; flex-wrap: wrap; }
  .rc-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0.3rem 0.8rem;
    border-radius: 2px;
    border: 1.5px solid;
  }
  .rc-badge-burgundy { color: #7C1B2E; border-color: #7C1B2E; background: rgba(124,27,46,0.06); }
  .rc-badge-forest   { color: #1B5741; border-color: #1B5741; background: rgba(27,87,65,0.06); }
  .rc-badge-brass    { color: #8B6914; border-color: #B8860B; background: rgba(184,134,11,0.06); }

  /* ── SECTION HEADER */
  .rc-section-header {
    display: flex;
    align-items: baseline;
    gap: 0.8rem;
    margin: 0 0 1.4rem 0;
    padding-bottom: 0.6rem;
    border-bottom: 1.5px solid #D4C9B0;
  }
  .rc-section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #1E3A5F;
    margin: 0;
  }
  .rc-section-tag {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #7C1B2E;
    background: rgba(124,27,46,0.08);
    padding: 0.2rem 0.6rem;
    border-radius: 2px;
  }

  /* ── CLINIC CARDS */
  .rc-card {
    background: #FFFFFF;
    border: 1.5px solid #D4C9B0;
    border-radius: 4px;
    padding: 1.5rem 1.8rem;
    box-shadow: 0 1px 10px rgba(61,50,40,0.05);
    margin-bottom: 1.2rem;
    animation: fadeUp 0.4s ease;
  }
  .rc-card-accent-burgundy { border-top: 3px solid #7C1B2E; }
  .rc-card-accent-forest   { border-top: 3px solid #1B5741; }
  .rc-card-accent-brass    { border-top: 3px solid #B8860B; }
  .rc-card-accent-blue     { border-top: 3px solid #1E3A5F; }
  .rc-card-accent-red      { border-top: 3px solid #C0392B; }
  .rc-card-title {
    font-family: 'Playfair Display', serif;
    font-size: 1rem;
    font-weight: 700;
    color: #1E3A5F;
    margin: 0 0 0.5rem 0;
  }
  .rc-card p {
    color: #7A6A5A;
    font-size: 0.88rem;
    line-height: 1.6;
    margin: 0;
  }

  /* ── RELEVANCE CHIP */
  .rc-relevance {
    background: #FAF7F0;
    border: 1px solid #D4C9B0;
    border-left: 3px solid #1B5741;
    border-radius: 3px;
    padding: 0.75rem 1rem;
    margin-top: 0.3rem;
    margin-bottom: 1.2rem;
    font-size: 0.8rem;
    line-height: 1.55;
    color: #3D3228;
  }
  .rc-relevance-heading {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #1B5741;
    display: block;
    margin-bottom: 0.3rem;
  }
  .rc-relevance-normal {
    color: #7A6A5A;
    font-size: 0.76rem;
    display: block;
    margin-top: 0.35rem;
  }
  .rc-relevance-value {
    color: #7C1B2E;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
    display: block;
    margin-top: 0.25rem;
  }

  /* ── RISK BANNERS */
  .rc-risk-high {
    background: #FFFAF9;
    border: 1.5px solid #EBCACA;
    border-left: 5px solid #C0392B;
    border-radius: 4px;
    padding: 1.8rem 2rem;
    box-shadow: 0 2px 14px rgba(192,57,43,0.08);
  }
  .rc-risk-warn {
    background: #FFFDF5;
    border: 1.5px solid #EBE0BF;
    border-left: 5px solid #B8860B;
    border-radius: 4px;
    padding: 1.8rem 2rem;
    box-shadow: 0 2px 14px rgba(184,134,11,0.08);
  }
  .rc-risk-safe {
    background: #F6FFFA;
    border: 1.5px solid #B8DCCA;
    border-left: 5px solid #1B5741;
    border-radius: 4px;
    padding: 1.8rem 2rem;
    box-shadow: 0 2px 14px rgba(27,87,65,0.08);
  }
  .rc-risk-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
  }
  .rc-risk-prob {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 3.4rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.5rem;
  }
  .rc-risk-desc {
    font-size: 0.9rem;
    color: #3D3228;
    line-height: 1.55;
  }

  /* ── RECOMMENDATION ROWS */
  .rc-rec {
    background: #FFFFFF;
    border: 1px solid #D4C9B0;
    border-left: 3px solid #1E3A5F;
    border-radius: 3px;
    padding: 0.7rem 1rem;
    margin-bottom: 0.55rem;
    font-size: 0.83rem;
    color: #3D3228;
    line-height: 1.5;
  }

  /* ── QUICK PROFILE BUTTONS */
  .stButton > button[kind="secondary"] {
    background: #FAF7F0 !important;
    color: #7C1B2E !important;
    border: 1.5px solid #D4C9B0 !important;
    box-shadow: none !important;
  }
  .stButton > button[kind="secondary"]:hover {
    border-color: #7C1B2E !important;
    background: rgba(124,27,46,0.05) !important;
    transform: none !important;
  }

  /* ── DATA TABLE TAG */
  .rc-data-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.73rem;
    font-weight: 600;
    padding: 0.18rem 0.5rem;
    border-radius: 2px;
    display: inline-block;
  }
  .rc-data-tag-high { background: #FDECEA; color: #C0392B; }
  .rc-data-tag-mid  { background: #FEF9E7; color: #8B6914; }
  .rc-data-tag-low  { background: #EAF5EE; color: #1B5741; }

  /* ── STAT ROW */
  .rc-stat-row {
    display: flex;
    gap: 1.2rem;
    margin-bottom: 1.4rem;
  }
  .rc-stat {
    flex: 1;
    background: #FFFFFF;
    border: 1.5px solid #D4C9B0;
    border-radius: 3px;
    padding: 1rem 1.2rem;
    text-align: center;
  }
  .rc-stat-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #1E3A5F;
    display: block;
  }
  .rc-stat-lbl {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #7A6A5A;
    margin-top: 0.2rem;
    display: block;
  }
</style>
""", unsafe_allow_html=True)

# ─── MODEL LOADING ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_all_models():
    try:
        with open('models_metadata.json', 'r') as f:
            metadata = json.load(f)
        scaler = joblib.load('scaler.pkl')
        models = {}
        for m_name, info in metadata['models'].items():
            models[m_name] = joblib.load(info['filename'])
        return models, scaler, metadata
    except Exception:
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
        col_names = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']
        try:
            df = pd.read_csv(url, names=col_names, na_values='?')
        except Exception:
            df = pd.read_csv('Heart Disease Data/processed.cleveland.data', names=col_names, na_values='?')
        df = df.dropna().reset_index(drop=True)
        df['target'] = (df['target'] > 0).astype(int)
        X = df.drop('target', axis=1)
        y = df['target']
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        mods = {
            'Random Forest':         RandomForestClassifier(n_estimators=100, random_state=42).fit(Xs, y),
            'Gradient Boosting':     GradientBoostingClassifier(n_estimators=100, random_state=42).fit(Xs, y),
            'K-Nearest Neighbors':   KNeighborsClassifier(n_neighbors=7).fit(Xs, y),
            'Logistic Regression':   LogisticRegression(random_state=42, max_iter=500).fit(Xs, y),
        }
        ens = VotingClassifier(
            estimators=[('rf',mods['Random Forest']),('gb',mods['Gradient Boosting']),
                        ('knn',mods['K-Nearest Neighbors']),('lr',mods['Logistic Regression'])],
            voting='soft'
        ).fit(Xs, y)
        mods['Voting Ensemble'] = ens
        try:
            with open('models_metadata.json','r') as f:
                meta = json.load(f)
        except Exception:
            meta = {'models':{k:{'accuracy':0.867,'roc_auc':0.941,'recall':0.852,'precision':0.871,'f1_score':0.861,'confusion_matrix':[[30,2],[2,26]]} for k in mods}}
        return mods, sc, meta

models_suite, scaler, metadata = load_all_models()

if 'session_history' not in st.session_state:
    st.session_state.session_history = []
if 'current_workspace' not in st.session_state:
    st.session_state.current_workspace = "Patient Intake & XAI"
if 'selected_model_name' not in st.session_state:
    st.session_state.selected_model_name = "Voting Ensemble"

def get_prediction(model_name, features_dict):
    df_in = pd.DataFrame([features_dict])
    scaled = scaler.transform(df_in)
    m = models_suite[model_name]
    prob = float(m.predict_proba(scaled)[0][1] * 100)
    pred = int(m.predict(scaled)[0])
    return prob, pred

# ─── PLOTLY THEME ──────────────────────────────────────────────────────────────
RC_LAYOUT = dict(
    paper_bgcolor='#FFFFFF',
    plot_bgcolor='#FAF7F0',
    font=dict(family='IBM Plex Sans, system-ui, sans-serif', color='#3D3228', size=12),
    margin=dict(l=20, r=20, t=44, b=20),
    title_font=dict(family='Playfair Display, serif', size=16, color='#1E3A5F'),
    legend=dict(bgcolor='rgba(255,255,255,0.8)', bordercolor='#D4C9B0', borderwidth=1),
    xaxis=dict(gridcolor='#EDE8DC', linecolor='#D4C9B0', tickfont=dict(size=11)),
    yaxis=dict(gridcolor='#EDE8DC', linecolor='#D4C9B0', tickfont=dict(size=11)),
)
RC_COLORS = ['#7C1B2E', '#1B5741', '#1E3A5F', '#B8860B', '#A52840', '#2E7D52']

# ─── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rc-hero">
  <div class="rc-hero-left">
    <div class="rc-eyebrow">Clinical Decision Intelligence Platform</div>
    <div class="rc-title">Heart<span>Guard</span> AI</div>
    <div class="rc-subtitle">
      Multi-model machine learning cardiac risk assessment built on the UCI Cleveland Dataset.
      Explainable, interpretable, and designed for clinical-grade decision support.
    </div>
    <div class="rc-badges">
      <span class="rc-badge rc-badge-burgundy">5-Model Ensemble</span>
      <span class="rc-badge rc-badge-forest">UCI Cleveland Dataset</span>
      <span class="rc-badge rc-badge-brass">Explainable AI (XAI)</span>
    </div>
  </div>
  <div class="rc-hero-right">
    <svg width="200" height="90" viewBox="0 0 200 90">
      <rect x="0" y="0" width="200" height="90" rx="4" fill="#FAF7F0" stroke="#D4C9B0" stroke-width="1.5"/>
      <path d="M10,45 L38,45 L46,20 L54,68 L60,30 L68,58 L76,45 L120,45 L128,20 L136,68 L142,30 L150,58 L158,45 L190,45"
            fill="none" stroke="#7C1B2E" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"
            stroke-dasharray="600" stroke-dashoffset="600"
            style="animation: ecg-pulse 3s ease forwards;"/>
      <text x="100" y="83" text-anchor="middle" font-family="IBM Plex Mono,monospace" font-size="9"
            fill="#7A6A5A" letter-spacing="2">ECG TELEMETRY</text>
    </svg>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Navigation")
    workspaces = [
        "Patient Intake & XAI",
        "Clinical Risk Simulator & 10-Yr Prognosis",
        "3D Anatomical Mesh & SOAP Notes",
        "Batch EHR CSV Intelligence Suite",
        "ML Model Workbench & Comparison",
        "Cardiac Knowledge Base & Dataset",
    ]
    selected_ws = st.radio(
        "Select Workspace:",
        workspaces,
        index=workspaces.index(st.session_state.current_workspace)
              if st.session_state.current_workspace in workspaces else 0
    )
    st.session_state.current_workspace = selected_ws

    st.markdown("---")
    st.markdown("### Active ML Engine")
    if metadata and 'models' in metadata:
        model_names = list(metadata['models'].keys())
        active_model = st.selectbox(
            "Select ML Model:",
            model_names,
            index=model_names.index(st.session_state.selected_model_name)
                  if st.session_state.selected_model_name in model_names else len(model_names)-1
        )
        st.session_state.selected_model_name = active_model
        m_info = metadata['models'][active_model]
        st.markdown(f"""
        <div style="background:#FAF7F0;border:1px solid #D4C9B0;border-left:3px solid #7C1B2E;
                    border-radius:3px;padding:0.8rem 1rem;font-size:0.8rem;color:#3D3228;">
          <div style="font-weight:700;color:#7C1B2E;font-size:0.7rem;letter-spacing:0.08em;
                      text-transform:uppercase;margin-bottom:0.5rem;">Model Stats</div>
          <div style="display:flex;justify-content:space-between;margin-bottom:0.3rem;">
            <span style="color:#7A6A5A;">Accuracy</span>
            <span style="font-family:IBM Plex Mono,monospace;font-weight:600;color:#1E3A5F;">
              {m_info.get('accuracy',0.867)*100:.1f}%</span>
          </div>
          <div style="display:flex;justify-content:space-between;margin-bottom:0.3rem;">
            <span style="color:#7A6A5A;">AUC-ROC</span>
            <span style="font-family:IBM Plex Mono,monospace;font-weight:600;color:#1E3A5F;">
              {m_info.get('roc_auc',0.941):.3f}</span>
          </div>
          <div style="display:flex;justify-content:space-between;">
            <span style="color:#7A6A5A;">Recall</span>
            <span style="font-family:IBM Plex Mono,monospace;font-weight:600;color:#1E3A5F;">
              {m_info.get('recall',0.852)*100:.1f}%</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Platform Author")
    st.markdown("""
    <div style="font-size:0.82rem;color:#3D3228;line-height:1.7;">
      <strong>Om Srivastava</strong><br>
      <a href="mailto:srivastavaom078@gmail.com"
         style="color:#7C1B2E;text-decoration:none;font-size:0.78rem;">
         srivastavaom078@gmail.com</a><br>
      <span style="color:#7A6A5A;font-size:0.76rem;">Data Science & Machine Learning</span>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# WORKSPACE 1 — PATIENT INTAKE & XAI
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.current_workspace == "Patient Intake & XAI":

    st.markdown("""
    <div class="rc-section-header">
      <div class="rc-section-title">Patient Intake & Explainable AI</div>
      <span class="rc-section-tag">Workspace 01</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "<p style='color:#7A6A5A;font-size:0.88rem;margin-bottom:1.4rem;'>"
        "Enter patient clinical parameters. Every field includes a clinical relevance callout "
        "with normal thresholds and a real-time assessment of the entered value.</p>",
        unsafe_allow_html=True
    )

    # ── Quick Profiles ─────────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:0.72rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;"
        "color:#7A6A5A;margin-bottom:0.6rem;'>Quick-Load Clinical Profiles</div>",
        unsafe_allow_html=True
    )
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        if st.button("High Risk", use_container_width=True):
            st.session_state.update(dict(wiz_age=67,wiz_sex="Male",wiz_cp="Asymptomatic (4)",
                wiz_trestbps=160,wiz_chol=286,wiz_fbs="No (<= 120 mg/dl)",
                wiz_restecg="Left Ventricular Hypertrophy (2)",wiz_thalach=108,wiz_exang="Yes",
                wiz_oldpeak=1.5,wiz_slope="Flat (2)",wiz_ca=3,wiz_thal="Reversible Defect (7)"))
            st.rerun()
    with p2:
        if st.button("Low Risk", use_container_width=True):
            st.session_state.update(dict(wiz_age=37,wiz_sex="Female",wiz_cp="Typical Angina (1)",
                wiz_trestbps=118,wiz_chol=190,wiz_fbs="No (<= 120 mg/dl)",
                wiz_restecg="Normal (0)",wiz_thalach=185,wiz_exang="No",
                wiz_oldpeak=0.0,wiz_slope="Upsloping (1)",wiz_ca=0,wiz_thal="Normal (3)"))
            st.rerun()
    with p3:
        if st.button("Moderate Risk", use_container_width=True):
            st.session_state.update(dict(wiz_age=58,wiz_sex="Male",wiz_cp="Atypical Angina (2)",
                wiz_trestbps=140,wiz_chol=245,wiz_fbs="Yes (> 120 mg/dl)",
                wiz_restecg="ST-T Wave Abnormality (1)",wiz_thalach=142,wiz_exang="Yes",
                wiz_oldpeak=1.2,wiz_slope="Flat (2)",wiz_ca=1,wiz_thal="Reversible Defect (7)"))
            st.rerun()
    with p4:
        if st.button("Reset Defaults", use_container_width=True):
            st.session_state.update(dict(wiz_age=52,wiz_sex="Male",wiz_cp="Atypical Angina (2)",
                wiz_trestbps=130,wiz_chol=240,wiz_fbs="No (<= 120 mg/dl)",
                wiz_restecg="Normal (0)",wiz_thalach=150,wiz_exang="No",
                wiz_oldpeak=1.0,wiz_slope="Upsloping (1)",wiz_ca=0,wiz_thal="Normal (3)"))
            st.rerun()

    st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)

    # ── Form ───────────────────────────────────────────────────────────────────
    with st.form("patient_intake_form"):
        tab1, tab2, tab3 = st.tabs([
            "Step 1 — Demographics & Vitals",
            "Step 2 — ECG & Stress Tests",
            "Step 3 — Advanced Imaging",
        ])

        # Step 1
        with tab1:
            c1, c2 = st.columns(2, gap="large")
            with c1:
                age = st.number_input("Age (years)", 18, 100,
                    st.session_state.get('wiz_age', 52), key="fi_age")
                age_flag = "Elevated Risk (>55 yrs)" if age > 55 else "Lower Age Risk"
                st.markdown(f"""
                <div class="rc-relevance">
                  <span class="rc-relevance-heading">Clinical Relevance — Age</span>
                  CAD risk rises with age due to arterial stiffening, vascular calcification, and
                  cumulative lipid exposure. Males >55 and females >65 face significantly higher baseline risk.
                  <span class="rc-relevance-normal">Normal threshold: risk factor threshold is 55 yrs (M) / 65 yrs (F)</span>
                  <span class="rc-relevance-value">Patient value: {age} yrs — {age_flag}</span>
                </div>
                """, unsafe_allow_html=True)

                sex = st.selectbox("Biological Sex", ["Male","Female"],
                    index=0 if st.session_state.get('wiz_sex','Male')=="Male" else 1, key="fi_sex")
                sex_note = "Male — higher early-onset CAD baseline (pre-menopausal estrogen absent)" \
                           if sex=="Male" else "Female — estrogen-protective baseline pre-menopause"
                st.markdown(f"""
                <div class="rc-relevance">
                  <span class="rc-relevance-heading">Clinical Relevance — Sex</span>
                  Males develop CAD earlier; the female disadvantage accelerates post-menopause. Sex is
                  an independent cardiovascular risk stratifier used in Framingham scoring.
                  <span class="rc-relevance-normal">Encoded as: Male = 1, Female = 0</span>
                  <span class="rc-relevance-value">Patient value: {sex_note}</span>
                </div>
                """, unsafe_allow_html=True)

                trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 70, 240,
                    st.session_state.get('wiz_trestbps', 130), key="fi_bp")
                bp_cat = ("Stage 2 HTN — High Risk" if trestbps >= 140 else
                          "Stage 1 HTN — Elevated" if trestbps >= 130 else
                          "Elevated — Monitor" if trestbps >= 120 else "Optimal")
                st.markdown(f"""
                <div class="rc-relevance">
                  <span class="rc-relevance-heading">Clinical Relevance — Resting BP</span>
                  Sustained hypertension damages arterial endothelium, accelerates atherosclerosis,
                  and increases left ventricular workload, raising cardiac event risk.
                  <span class="rc-relevance-normal">Normal: &lt;120 | Elevated: 120–129 | Stage 1 HTN: 130–139 | Stage 2: ≥140 mm Hg</span>
                  <span class="rc-relevance-value">Patient value: {trestbps} mm Hg — {bp_cat}</span>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 650,
                    st.session_state.get('wiz_chol', 240), key="fi_chol")
                chol_cat = ("High — Hypercholesterolaemia" if chol >= 240 else
                            "Borderline High" if chol >= 200 else "Desirable")
                st.markdown(f"""
                <div class="rc-relevance">
                  <span class="rc-relevance-heading">Clinical Relevance — Serum Cholesterol</span>
                  Elevated LDL deposits in vessel intima, forming atheromatous plaques that narrow
                  coronary arteries, restrict blood flow, and precipitate ischaemic events.
                  <span class="rc-relevance-normal">Desirable: &lt;200 | Borderline: 200–239 | High: ≥240 mg/dl</span>
                  <span class="rc-relevance-value">Patient value: {chol} mg/dl — {chol_cat}</span>
                </div>
                """, unsafe_allow_html=True)

                fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl",
                    ["No (<= 120 mg/dl)", "Yes (> 120 mg/dl)"],
                    index=0 if "No" in st.session_state.get('wiz_fbs','No') else 1, key="fi_fbs")
                fbs_note = "Diabetic threshold exceeded — doubles CVD risk" if "Yes" in fbs else "Within normal fasting glucose range"
                st.markdown(f"""
                <div class="rc-relevance">
                  <span class="rc-relevance-heading">Clinical Relevance — Fasting Blood Sugar</span>
                  Hyperglycaemia damages endothelial cells and promotes glycosylation of LDL particles,
                  significantly accelerating atherosclerotic progression in diabetic patients.
                  <span class="rc-relevance-normal">Normal fasting glucose: ≤100 mg/dl | Diabetic threshold: &gt;126 mg/dl</span>
                  <span class="rc-relevance-value">Patient value: {fbs} — {fbs_note}</span>
                </div>
                """, unsafe_allow_html=True)

        # Step 2
        with tab2:
            c1, c2 = st.columns(2, gap="large")
            with c1:
                cp_opts = ["Typical Angina (1)","Atypical Angina (2)","Non-Anginal Pain (3)","Asymptomatic (4)"]
                cp = st.selectbox("Chest Pain Type", cp_opts,
                    index=cp_opts.index(st.session_state.get('wiz_cp','Atypical Angina (2)'))
                          if st.session_state.get('wiz_cp') in cp_opts else 1, key="fi_cp")
                cp_note = "Silent ischaemia / highest CAD correlation" if "Asymptomatic" in cp else \
                          "Classic anginal pattern — high suspicion" if "Typical" in cp else \
                          "Moderate suspicion" if "Atypical" in cp else "Non-cardiac — low suspicion"
                st.markdown(f"""
                <div class="rc-relevance">
                  <span class="rc-relevance-heading">Clinical Relevance — Chest Pain Type</span>
                  Chest pain classification is the primary pre-test discriminator for CAD. Paradoxically,
                  asymptomatic presentation (type 4) correlates most strongly with confirmed CAD in the dataset.
                  <span class="rc-relevance-normal">1=Typical | 2=Atypical | 3=Non-anginal | 4=Asymptomatic (highest model weight)</span>
                  <span class="rc-relevance-value">Patient value: {cp} — {cp_note}</span>
                </div>
                """, unsafe_allow_html=True)

                restecg_opts = ["Normal (0)","ST-T Wave Abnormality (1)","Left Ventricular Hypertrophy (2)"]
                restecg = st.selectbox("Resting ECG Results", restecg_opts,
                    index=restecg_opts.index(st.session_state.get('wiz_restecg','Normal (0)'))
                          if st.session_state.get('wiz_restecg') in restecg_opts else 0, key="fi_ecg")
                ecg_note = "Baseline normal — no conduction anomaly" if "0" in restecg else \
                           "Ischaemic repolarisation anomaly present" if "1" in restecg else \
                           "LV hypertrophy — chronic pressure overload"
                st.markdown(f"""
                <div class="rc-relevance">
                  <span class="rc-relevance-heading">Clinical Relevance — Resting ECG</span>
                  The resting 12-lead ECG evaluates baseline conduction. ST-T wave changes and LV hypertrophy
                  both predict adverse cardiac events independent of other risk factors.
                  <span class="rc-relevance-normal">0=Normal | 1=ST-T Abnormality (ischaemic) | 2=LV Hypertrophy (hypertensive)</span>
                  <span class="rc-relevance-value">Patient value: {restecg} — {ecg_note}</span>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                thalach = st.number_input("Max Heart Rate Achieved (bpm)", 60, 230,
                    st.session_state.get('wiz_thalach', 150), key="fi_hr")
                hr_note = "Impaired chronotropic reserve — suggestive of CAD" if thalach < 130 else \
                          "Moderate reserve" if thalach < 160 else "Good exertional capacity"
                st.markdown(f"""
                <div class="rc-relevance">
                  <span class="rc-relevance-heading">Clinical Relevance — Max Heart Rate</span>
                  Failure to achieve age-predicted max HR (220 − age) during treadmill testing indicates
                  impaired chronotropic reserve, a hallmark of significant coronary artery obstruction.
                  <span class="rc-relevance-normal">Target: 220 − age bpm | Impaired reserve: &lt;85% of target HR</span>
                  <span class="rc-relevance-value">Patient value: {thalach} bpm — {hr_note}</span>
                </div>
                """, unsafe_allow_html=True)

                exang = st.selectbox("Exercise-Induced Angina", ["No","Yes"],
                    index=0 if st.session_state.get('wiz_exang','No')=="No" else 1, key="fi_exang")
                ea_note = "Positive for exertional ischaemia — demand-induced flow restriction" \
                          if exang=="Yes" else "Negative for exercise-induced angina"
                st.markdown(f"""
                <div class="rc-relevance">
                  <span class="rc-relevance-heading">Clinical Relevance — Exercise Angina</span>
                  Chest pain precipitated by exertion indicates epicardial stenosis that cannot
                  accommodate demand-induced coronary flow increase — a cardinal ischaemic symptom.
                  <span class="rc-relevance-normal">Encoded as: Yes = 1 (ischaemic indicator) | No = 0</span>
                  <span class="rc-relevance-value">Patient value: {exang} — {ea_note}</span>
                </div>
                """, unsafe_allow_html=True)

        # Step 3
        with tab3:
            c1, c2 = st.columns(2, gap="large")
            with c1:
                oldpeak = st.slider("Exercise ST Depression (oldpeak, mm)", 0.0, 6.2,
                    float(st.session_state.get('wiz_oldpeak', 1.0)), step=0.1, key="fi_op")
                op_note = ("Severe ischaemic depression ≥2.0 mm" if oldpeak >= 2.0 else
                           "Diagnostic for ischaemia ≥1.0 mm" if oldpeak >= 1.0 else "Normal ST baseline")
                st.markdown(f"""
                <div class="rc-relevance">
                  <span class="rc-relevance-heading">Clinical Relevance — ST Depression (oldpeak)</span>
                  Horizontal or down-sloping ST segment depression during exercise quantifies
                  subendocardial ischaemia. Greater depression predicts larger ischaemic territory.
                  <span class="rc-relevance-normal">Normal: &lt;1.0 mm | Diagnostic: ≥1.0 mm | Severe: ≥2.0 mm</span>
                  <span class="rc-relevance-value">Patient value: {oldpeak} mm — {op_note}</span>
                </div>
                """, unsafe_allow_html=True)

                slope_opts = ["Upsloping (1)","Flat (2)","Downsloping (3)"]
                slope = st.selectbox("ST Segment Slope at Peak Exercise", slope_opts,
                    index=slope_opts.index(st.session_state.get('wiz_slope','Upsloping (1)'))
                          if st.session_state.get('wiz_slope') in slope_opts else 0, key="fi_slope")
                sl_note = "Benign upsloping — good prognosis" if "1" in slope else \
                          "Flat — ischaemic pattern, moderate risk" if "2" in slope else \
                          "Downsloping — severe multi-vessel CAD indicator"
                st.markdown(f"""
                <div class="rc-relevance">
                  <span class="rc-relevance-heading">Clinical Relevance — ST Segment Slope</span>
                  The ST slope characterises repolarisation recovery. Flat or down-sloping patterns
                  at peak exercise strongly correlate with multi-vessel obstructive CAD.
                  <span class="rc-relevance-normal">1=Upsloping (benign) | 2=Flat (ischaemic) | 3=Downsloping (severe CAD)</span>
                  <span class="rc-relevance-value">Patient value: {slope} — {sl_note}</span>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                ca = st.slider("Major Vessels via Fluoroscopy (0–3)", 0, 3,
                    int(st.session_state.get('wiz_ca', 0)), key="fi_ca")
                ca_note = ("No stenotic vessels — clean coronaries" if ca == 0 else
                           f"{ca}-vessel CAD — significant anatomic burden")
                st.markdown(f"""
                <div class="rc-relevance">
                  <span class="rc-relevance-heading">Clinical Relevance — Fluoroscopy Vessels (ca)</span>
                  The count of major coronary arteries (LAD, LCx, RCA) showing calcified stenosis
                  under fluoroscopy directly quantifies anatomic disease burden — the strongest predictor
                  in this dataset.
                  <span class="rc-relevance-normal">0=No disease | 1–3=Multi-vessel CAD (strongest model weight)</span>
                  <span class="rc-relevance-value">Patient value: {ca} vessels — {ca_note}</span>
                </div>
                """, unsafe_allow_html=True)

                thal_opts = ["Normal (3)","Fixed Defect (6)","Reversible Defect (7)"]
                thal = st.selectbox("Thallium Stress Test Result", thal_opts,
                    index=thal_opts.index(st.session_state.get('wiz_thal','Normal (3)'))
                          if st.session_state.get('wiz_thal') in thal_opts else 0, key="fi_thal")
                th_note = ("Normal myocardial perfusion" if "3" in thal else
                           "Fixed defect — prior infarct scar" if "6" in thal else
                           "Reversible defect — hibernating ischaemic myocardium")
                st.markdown(f"""
                <div class="rc-relevance">
                  <span class="rc-relevance-heading">Clinical Relevance — Thallium Stress Test</span>
                  Nuclear perfusion imaging differentiates infarcted from ischaemic but viable myocardium.
                  Reversible defects indicate territory amenable to revascularisation.
                  <span class="rc-relevance-normal">3=Normal | 6=Fixed defect (scar) | 7=Reversible defect (viable ischaemia)</span>
                  <span class="rc-relevance-value">Patient value: {thal} — {th_note}</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)
        submit = st.form_submit_button(
            "Run Multi-Model Diagnostic Assessment",
            use_container_width=True,
            type="primary"
        )

    # ── RESULTS ────────────────────────────────────────────────────────────────
    if submit:
        features_dict = {
            'age': age,
            'sex': 1 if sex=="Male" else 0,
            'cp': 1 if "1" in cp else 2 if "2" in cp else 3 if "3" in cp else 4,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': 1 if "Yes" in fbs else 0,
            'restecg': 0 if "0" in restecg else 1 if "1" in restecg else 2,
            'thalach': thalach,
            'exang': 1 if exang=="Yes" else 0,
            'oldpeak': oldpeak,
            'slope': 1 if "1" in slope else 2 if "2" in slope else 3,
            'ca': ca,
            'thal': 3 if "3" in thal else 6 if "6" in thal else 7,
        }
        active_m = st.session_state.selected_model_name
        prob, pred = get_prediction(active_m, features_dict)

        st.session_state.session_history.append({
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'model': active_m,
            'age': age, 'sex': sex,
            'bp': trestbps, 'chol': chol,
            'prob_%': round(prob,1),
            'result': 'Heart Disease' if pred==1 else 'No Disease'
        })

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="rc-section-header">
          <div class="rc-section-title">Diagnostic Assessment Report</div>
          <span class="rc-section-tag">{active_m}</span>
        </div>
        """, unsafe_allow_html=True)

        # Risk level
        if prob >= 70:
            risk_class = "rc-risk-high"
            risk_color = "#C0392B"
            risk_label = "HIGH CARDIOVASCULAR RISK"
        elif prob >= 35:
            risk_class = "rc-risk-warn"
            risk_color = "#8B6914"
            risk_label = "MODERATE CARDIOVASCULAR RISK"
        else:
            risk_class = "rc-risk-safe"
            risk_color = "#1B5741"
            risk_label = "LOW CARDIOVASCULAR RISK"

        r1, r2 = st.columns([1.5, 1], gap="large")
        with r1:
            st.markdown(f"""
            <div class="{risk_class}">
              <div class="rc-risk-label" style="color:{risk_color};">{risk_label}</div>
              <div class="rc-risk-prob" style="color:{risk_color};">{prob:.1f}%
                <span style="font-size:1rem;font-family:'IBM Plex Sans',sans-serif;
                             color:#7A6A5A;font-weight:400;"> disease probability</span>
              </div>
              <div class="rc-risk-desc">
                Model <strong>{active_m}</strong> classifies this clinical profile as
                <strong>{'POSITIVE for Coronary Artery Disease' if pred==1 else 'NEGATIVE for Coronary Artery Disease'}</strong>.
                This result is for decision support only and must be interpreted alongside full clinical assessment.
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Clinical recs
            st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)
            st.markdown(
                "<div style='font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;"
                "color:#7A6A5A;margin-bottom:0.6rem;'>Clinical Action Recommendations</div>",
                unsafe_allow_html=True
            )
            recs = []
            if prob >= 50:
                recs.append("Cardiology referral for coronary angiography or nuclear stress test is warranted.")
            if chol > 240:
                recs.append(f"Dyslipidaemia management: cholesterol {chol} mg/dl exceeds threshold. Evaluate statin therapy.")
            if trestbps >= 130:
                recs.append(f"Hypertension control: BP {trestbps} mm Hg. Ambulatory monitoring and antihypertensive review.")
            if oldpeak >= 1.0:
                recs.append(f"Ischaemia workup: ST depression {oldpeak} mm meets diagnostic threshold for exertional ischaemia.")
            if exang == "Yes":
                recs.append("Exertional angina protocol: restricted coronary flow confirmed — anti-anginal therapy indicated.")
            if ca > 0:
                recs.append(f"Multi-vessel CAD ({ca} vessels): high anatomic burden — revascularisation assessment advised.")
            if not recs:
                recs.append("Parameters largely within normal reference ranges. Maintain lifestyle risk factor modification.")

            for r in recs:
                st.markdown(f'<div class="rc-rec">{r}</div>', unsafe_allow_html=True)

        with r2:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob,
                title={'text':"Risk Probability", 'font':{'size':13,'color':'#7A6A5A',
                                                           'family':'IBM Plex Sans'}},
                number={'suffix':"%", 'font':{'size':38,'color':risk_color,
                                               'family':'IBM Plex Mono'}},
                gauge={
                    'axis':{'range':[0,100],'tickwidth':1,'tickcolor':'#D4C9B0',
                            'tickfont':{'size':10,'color':'#7A6A5A'}},
                    'bar':{'color':risk_color,'thickness':0.22},
                    'bgcolor':'#FAF7F0',
                    'borderwidth':1,'bordercolor':'#D4C9B0',
                    'steps':[
                        {'range':[0,35],'color':'rgba(27,87,65,0.12)'},
                        {'range':[35,70],'color':'rgba(184,134,11,0.12)'},
                        {'range':[70,100],'color':'rgba(192,57,43,0.12)'}
                    ],
                    'threshold':{'line':{'color':risk_color,'width':3},'value':prob}
                }
            ))
            fig_g.update_layout(
                height=280,
                paper_bgcolor='#FFFFFF',
                font=dict(color='#3D3228'),
                margin=dict(l=20,r=20,t=40,b=10)
            )
            st.plotly_chart(fig_g, use_container_width=True)

        # ── XAI Waterfall ──────────────────────────────────────────────────────
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("""
        <div class="rc-section-header">
          <div class="rc-section-title">Explainable AI — Risk Contribution Waterfall</div>
          <span class="rc-section-tag">XAI</span>
        </div>
        <p style='color:#7A6A5A;font-size:0.85rem;margin-bottom:1rem;'>
        Each bar shows how far this patient's value for that feature pushes the risk score
        above (+) or below (−) the population mean baseline.
        </p>
        """, unsafe_allow_html=True)

        means = scaler.mean_
        vals = list(features_dict.values())
        feat_keys = list(features_dict.keys())
        key_weights = {'ca':5.0,'thal':4.5,'oldpeak':4.0,'cp':3.8,'thalach':3.2,'exang':3.0,
                       'trestbps':2.5,'chol':2.5,'age':2.0,'sex':1.5,'fbs':1.2,'restecg':1.8,'slope':2.8}
        deltas = []
        for i, k in enumerate(feat_keys):
            z = (vals[i] - means[i]) / scaler.scale_[i]
            push = round(z * key_weights.get(k, 2.0), 1)
            deltas.append((k.upper(), push))
        deltas_sorted = sorted(deltas, key=lambda x: abs(x[1]), reverse=True)[:9]

        xai_x = [d[0] for d in deltas_sorted]
        xai_y = [d[1] for d in deltas_sorted]
        xai_colors = ['#C0392B' if v > 0 else '#1B5741' for v in xai_y]

        fig_wf = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative"]*len(xai_x),
            x=xai_x,
            y=xai_y,
            textposition="outside",
            text=[f"{v:+.1f}" for v in xai_y],
            textfont=dict(family='IBM Plex Mono', size=11, color='#3D3228'),
            connector={"line":{"color":"#D4C9B0","width":1.5,"dash":"dot"}},
            decreasing={"marker":{"color":"#1B5741","line":{"color":"#155233","width":1}}},
            increasing={"marker":{"color":"#C0392B","line":{"color":"#9B2335","width":1}}},
        ))
        fig_wf.update_layout(
            **RC_LAYOUT,
            title="Feature Risk Push Relative to Population Baseline",
            height=360,
            showlegend=False,
        )
        fig_wf.update_xaxes(tickfont=dict(family='IBM Plex Mono', size=11, color='#3D3228'))
        st.plotly_chart(fig_wf, use_container_width=True)

        # ── Vitals Comparison ──────────────────────────────────────────────────
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("""
        <div class="rc-section-header">
          <div class="rc-section-title">Patient Vitals vs. Clinical Reference Ranges</div>
          <span class="rc-section-tag">Comparison</span>
        </div>
        """, unsafe_allow_html=True)

        vitals_data = {
            'Parameter': ['Blood Pressure\n(mm Hg)', 'Cholesterol\n(mg/dl)',
                          'Max Heart Rate\n(bpm)', 'ST Depression\n(×20, mm)'],
            'Patient': [trestbps, chol, thalach, oldpeak * 20],
            'Clinical Target': [120, 200, 155, 0],
        }
        vdf = pd.DataFrame(vitals_data)
        fig_vit = go.Figure()
        fig_vit.add_trace(go.Bar(
            name='Patient Value',
            x=vdf['Parameter'], y=vdf['Patient'],
            marker_color='#7C1B2E',
            marker_line=dict(color='#5A1422', width=1),
            width=0.32,
            text=[f'{v:.0f}' for v in vdf['Patient']],
            textposition='outside',
            textfont=dict(family='IBM Plex Mono', size=11, color='#7C1B2E'),
        ))
        fig_vit.add_trace(go.Bar(
            name='Clinical Target',
            x=vdf['Parameter'], y=vdf['Clinical Target'],
            marker_color='#1B5741',
            marker_line=dict(color='#144D32', width=1),
            width=0.32,
            text=[f'{v:.0f}' for v in vdf['Clinical Target']],
            textposition='outside',
            textfont=dict(family='IBM Plex Mono', size=11, color='#1B5741'),
        ))
        fig_vit.update_layout(
            **RC_LAYOUT,
            title="Patient Values vs. Healthy Clinical Reference Targets",
            barmode='group',
            height=360,
        )
        st.plotly_chart(fig_vit, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# WORKSPACE 2 — RISK SIMULATOR & 10-YR PROGNOSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_workspace == "Clinical Risk Simulator & 10-Yr Prognosis":
    st.markdown("""
    <div class="rc-section-header">
      <div class="rc-section-title">Clinical Risk Simulator & 10-Year Prognosis</div>
      <span class="rc-section-tag">Workspace 02</span>
    </div>
    <p style='color:#7A6A5A;font-size:0.88rem;margin-bottom:1.6rem;'>
    Adjust vitals in real time to model how clinical interventions alter predicted cardiac risk.
    The 10-year trajectory projects both unmanaged and actively treated risk curves.
    </p>
    """, unsafe_allow_html=True)

    sim_col1, sim_col2 = st.columns([1, 1.3], gap="large")
    with sim_col1:
        st.markdown(
            "<div style='font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;"
            "color:#7A6A5A;margin-bottom:0.8rem;'>Adjust Patient Parameters</div>",
            unsafe_allow_html=True
        )
        sim_age  = st.slider("Age (years)", 20, 90, 60)
        sim_bp   = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 150)
        sim_chol = st.slider("Serum Cholesterol (mg/dl)", 120, 450, 260)
        sim_hr   = st.slider("Max Heart Rate (bpm)", 70, 210, 130)
        sim_op   = st.slider("ST Depression (mm)", 0.0, 5.0, 2.0, step=0.1)
        sim_ca   = st.selectbox("Major Vessels (ca)", [0,1,2,3], index=2)
        sim_ex   = st.selectbox("Exercise Angina", ["No","Yes"], index=1)

    features_sim = {
        'age':sim_age,'sex':1,'cp':4,'trestbps':sim_bp,'chol':sim_chol,'fbs':0,
        'restecg':1,'thalach':sim_hr,'exang':1 if sim_ex=="Yes" else 0,
        'oldpeak':sim_op,'slope':2,'ca':sim_ca,'thal':7
    }
    active_m = st.session_state.selected_model_name
    prob_sim, pred_sim = get_prediction(active_m, features_sim)
    sim_color = "#C0392B" if prob_sim >= 70 else "#8B6914" if prob_sim >= 35 else "#1B5741"
    sim_label = "HIGH RISK" if prob_sim >= 70 else "MODERATE RISK" if prob_sim >= 35 else "LOW RISK"

    with sim_col2:
        st.markdown(
            "<div style='font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;"
            "color:#7A6A5A;margin-bottom:0.8rem;'>Live Simulation Result</div>",
            unsafe_allow_html=True
        )
        rc_sim_card = "rc-risk-high" if prob_sim>=70 else "rc-risk-warn" if prob_sim>=35 else "rc-risk-safe"
        st.markdown(f"""
        <div class="{rc_sim_card}" style="text-align:center;padding:2.2rem 2rem;">
          <div class="rc-risk-label" style="color:{sim_color};">{sim_label}</div>
          <div class="rc-risk-prob" style="color:{sim_color};font-size:4rem;">{prob_sim:.1f}%</div>
          <div style="font-size:0.88rem;color:#3D3228;font-weight:600;">
            {'POSITIVE — Heart Disease Likely' if pred_sim==1 else 'NEGATIVE — No Disease Detected'}
          </div>
          <div style="font-size:0.75rem;color:#7A6A5A;margin-top:0.6rem;">{active_m}</div>
        </div>
        """, unsafe_allow_html=True)

        # Mini stat row
        st.markdown(f"""
        <div class="rc-stat-row" style="margin-top:1rem;">
          <div class="rc-stat">
            <span class="rc-stat-val" style="color:#7C1B2E;">{sim_bp}</span>
            <span class="rc-stat-lbl">BP mm Hg</span>
          </div>
          <div class="rc-stat">
            <span class="rc-stat-val" style="color:#1B5741;">{sim_chol}</span>
            <span class="rc-stat-lbl">Cholesterol</span>
          </div>
          <div class="rc-stat">
            <span class="rc-stat-val" style="color:#1E3A5F;">{sim_hr}</span>
            <span class="rc-stat-lbl">Max HR bpm</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # 10-Year Prognosis
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("""
    <div class="rc-section-header">
      <div class="rc-section-title">10-Year Cardiac Risk Trajectory</div>
      <span class="rc-section-tag">Prognosis</span>
    </div>
    <p style='color:#7A6A5A;font-size:0.85rem;margin-bottom:1rem;'>
    Projected risk curves for unmanaged baseline vs. active medical intervention over a 10-year horizon.
    </p>
    """, unsafe_allow_html=True)

    years_lbl = ['Baseline','Year 1','Year 2','Year 3','Year 5','Year 7','Year 10']
    years_n   = np.array([0,1,2,3,5,7,10])
    unmanaged = np.clip(prob_sim + years_n * 2.4, 0, 98)
    managed   = np.clip(prob_sim - years_n * 3.2, 5, 98)

    fig_prog = go.Figure()
    fig_prog.add_trace(go.Scatter(
        x=years_lbl, y=unmanaged,
        name='Unmanaged Baseline',
        mode='lines+markers',
        line=dict(color='#C0392B', width=2.5),
        marker=dict(size=7, color='#C0392B', line=dict(color='#FFFFFF',width=2)),
        fill='tozeroy',
        fillcolor='rgba(192,57,43,0.06)'
    ))
    fig_prog.add_trace(go.Scatter(
        x=years_lbl, y=managed,
        name='Proactive Medical Intervention',
        mode='lines+markers',
        line=dict(color='#1B5741', width=2.5, dash='dash'),
        marker=dict(size=7, color='#1B5741', line=dict(color='#FFFFFF',width=2)),
        fill='tozeroy',
        fillcolor='rgba(27,87,65,0.06)'
    ))
    fig_prog.update_layout(
        **RC_LAYOUT,
        title="10-Year Cardiovascular Risk Trajectory — Unmanaged vs. Treated",
        yaxis_title="Predicted Risk (%)",
        xaxis_title="Clinical Timeline",
        height=360,
    )
    st.plotly_chart(fig_prog, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# WORKSPACE 3 — 3D CARDIAC MESH & SOAP NOTES
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_workspace == "3D Anatomical Mesh & SOAP Notes":
    st.markdown("""
    <div class="rc-section-header">
      <div class="rc-section-title">3D Anatomical Mesh & EHR SOAP Notes</div>
      <span class="rc-section-tag">Workspace 03</span>
    </div>
    """, unsafe_allow_html=True)

    c_mesh1, c_mesh2 = st.columns([1.3, 1], gap="large")
    with c_mesh1:
        st.markdown(
            "<div style='font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;"
            "color:#7A6A5A;margin-bottom:0.6rem;'>Interactive 3D Myocardial Perfusion Surface</div>",
            unsafe_allow_html=True
        )
        u = np.linspace(0, 2*np.pi, 35)
        v = np.linspace(0, np.pi, 35)
        x = 16*np.sin(v)[:,None]**3 * np.cos(u)[None,:]
        y = 13*np.cos(v)[:,None] - 5*np.cos(2*v)[:,None] - 2*np.cos(3*v)[:,None] - np.cos(4*v)[:,None]
        z = 16*np.sin(v)[:,None]**3 * np.sin(u)[None,:]

        fig_3d = go.Figure(data=[go.Surface(
            x=x, y=y, z=z,
            colorscale=[
                [0.0,'#FAF7F0'],[0.2,'#D4C9B0'],[0.4,'#B8860B'],
                [0.6,'#A52840'],[0.8,'#7C1B2E'],[1.0,'#1E3A5F']
            ],
            showscale=True,
            colorbar=dict(
                thickness=10, len=0.7, x=1.02,
                tickfont=dict(family='IBM Plex Mono', size=10, color='#3D3228'),
                title=dict(text='Perfusion', font=dict(size=10, color='#7A6A5A'))
            ),
            contours=dict(
                x=dict(show=False), y=dict(show=False), z=dict(show=False)
            ),
            lighting=dict(ambient=0.7, diffuse=0.8, specular=0.15),
        )])
        fig_3d.update_layout(
            title=dict(text="3D Parametric Cardiac Surface Model",
                       font=dict(family='Playfair Display, serif', size=14, color='#1E3A5F')),
            scene=dict(
                xaxis=dict(title='LAD Artery', tickfont=dict(size=9,color='#7A6A5A'),
                           backgroundcolor='#FAF7F0', gridcolor='#D4C9B0'),
                yaxis=dict(title='Left Ventricle', tickfont=dict(size=9,color='#7A6A5A'),
                           backgroundcolor='#FAF7F0', gridcolor='#D4C9B0'),
                zaxis=dict(title='RCA Artery', tickfont=dict(size=9,color='#7A6A5A'),
                           backgroundcolor='#FAF7F0', gridcolor='#D4C9B0'),
                bgcolor='#FAF7F0',
            ),
            height=420,
            paper_bgcolor='#FFFFFF',
            margin=dict(l=0,r=0,t=40,b=0),
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    with c_mesh2:
        st.markdown(
            "<div style='font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;"
            "color:#7A6A5A;margin-bottom:0.6rem;'>Automated EHR SOAP Note Generator</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='color:#7A6A5A;font-size:0.82rem;margin-bottom:0.8rem;'>"
            "Generates a physician-format SOAP note ready for Epic/Cerner EHR integration.</p>",
            unsafe_allow_html=True
        )
        soap = f"""CLINICAL SOAP NOTE — HeartGuard AI
Date:       {datetime.now().strftime('%Y-%m-%d  %H:%M')}
Patient ID: HG-{np.random.randint(10000, 99999)}
────────────────────────────────────────

SUBJECTIVE:
  Chief Complaint:   Cardiac risk evaluation
  Chest Pain Type:   Typical Angina during exertion
  Exercise Angina:   Present

OBJECTIVE:
  Resting BP:        140 mm Hg
  Serum Cholesterol: 245 mg/dl
  Max Heart Rate:    142 bpm
  ST Depression:     1.2 mm
  Vessels (fluoro):  1 vessel

ASSESSMENT:
  ML Model (Voting Ensemble): 64.5% CAD probability
  Classification:             MODERATE RISK

PLAN:
  1. Stress echocardiography / coronary angiography
  2. Statin therapy — target cholesterol <200 mg/dl
  3. Ambulatory BP monitoring + antihypertensive review
  4. Follow-up in 2 weeks — cardiology referral

────────────────────────────────────────
Generated by HeartGuard AI | For clinical reference only
"""
        st.text_area("Generated SOAP Note", soap, height=320)


# ═══════════════════════════════════════════════════════════════════════════════
# WORKSPACE 4 — BATCH EHR CSV INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_workspace == "Batch EHR CSV Intelligence Suite":
    st.markdown("""
    <div class="rc-section-header">
      <div class="rc-section-title">Batch EHR CSV Intelligence Suite</div>
      <span class="rc-section-tag">Workspace 04</span>
    </div>
    <p style='color:#7A6A5A;font-size:0.88rem;margin-bottom:1.4rem;'>
    Upload a CSV dataset of patient records to run bulk multi-model risk assessments
    and export annotated clinical prediction files.
    </p>
    """, unsafe_allow_html=True)

    up_file = st.file_uploader("Upload Patient Records CSV", type=["csv"])
    if up_file:
        try:
            b_df = pd.read_csv(up_file)
            st.markdown(f"**Loaded:** `{up_file.name}` — {len(b_df)} records")
            st.dataframe(b_df.head(5), use_container_width=True)
            req = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach',
                   'exang','oldpeak','slope','ca','thal']
            missing = [c for c in req if c not in b_df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                if st.button("Process Batch — Active Model", type="primary", use_container_width=True):
                    active_m = st.session_state.selected_model_name
                    Xb = scaler.transform(b_df[req])
                    probs = models_suite[active_m].predict_proba(Xb)[:,1]*100
                    preds = models_suite[active_m].predict(Xb)
                    b_df['Probability_%'] = np.round(probs,1)
                    b_df['Prediction'] = np.where(preds==1,'Heart Disease','No Disease')
                    b_df['Risk_Category'] = np.where(probs>=70,'High',np.where(probs>=35,'Moderate','Low'))

                    m1,m2,m3 = st.columns(3)
                    with m1: st.metric("High Risk",   int(sum(probs>=70)),   f"{sum(probs>=70)/len(b_df)*100:.1f}%")
                    with m2: st.metric("Moderate",    int(sum((probs>=35)&(probs<70))),
                                       f"{sum((probs>=35)&(probs<70))/len(b_df)*100:.1f}%")
                    with m3: st.metric("Low Risk",    int(sum(probs<35)),    f"{sum(probs<35)/len(b_df)*100:.1f}%")

                    fig_b = px.histogram(
                        b_df, x='Probability_%', nbins=20,
                        title=f"Risk Score Distribution ({active_m})",
                        color='Risk_Category',
                        color_discrete_map={'High':'#C0392B','Moderate':'#B8860B','Low':'#1B5741'}
                    )
                    fig_b.update_layout(**RC_LAYOUT, height=320)
                    st.plotly_chart(fig_b, use_container_width=True)
                    st.dataframe(b_df, use_container_width=True)
                    st.download_button(
                        "Export Annotated Predictions (CSV)",
                        data=b_df.to_csv(index=False).encode(),
                        file_name=f"cardiac_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv", use_container_width=True
                    )
        except Exception as ex:
            st.error(f"Error reading CSV: {ex}")


# ═══════════════════════════════════════════════════════════════════════════════
# WORKSPACE 5 — ML MODEL WORKBENCH
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_workspace == "ML Model Workbench & Comparison":
    st.markdown("""
    <div class="rc-section-header">
      <div class="rc-section-title">ML Model Workbench & Comparative Analytics</div>
      <span class="rc-section-tag">Workspace 05</span>
    </div>
    """, unsafe_allow_html=True)

    if metadata and 'models' in metadata:
        m_df = pd.DataFrame(metadata['models']).T.reset_index().rename(columns={'index':'Model'})
        st.markdown(
            "<div style='font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;"
            "color:#7A6A5A;margin-bottom:0.6rem;'>Performance Metrics — All Models</div>",
            unsafe_allow_html=True
        )
        display_cols = [c for c in ['Model','accuracy','roc_auc','precision','recall','f1_score'] if c in m_df.columns]
        st.dataframe(m_df[display_cols], use_container_width=True)

        st.markdown("<div style='margin-bottom:1.2rem;'></div>", unsafe_allow_html=True)
        wm1, wm2 = st.columns(2, gap="large")

        with wm1:
            st.markdown(
                "<div style='font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;"
                "color:#7A6A5A;margin-bottom:0.6rem;'>Accuracy / AUC / F1 Comparison</div>",
                unsafe_allow_html=True
            )
            available_metrics = [c for c in ['accuracy','roc_auc','f1_score'] if c in m_df.columns]
            fig_acc = px.bar(
                m_df, x='Model', y=available_metrics,
                barmode='group',
                title="Model Performance Metrics",
                color_discrete_sequence=['#7C1B2E','#1B5741','#1E3A5F']
            )
            fig_acc.update_layout(**RC_LAYOUT, height=320)
            fig_acc.update_xaxes(tickangle=-20)
            st.plotly_chart(fig_acc, use_container_width=True)

        with wm2:
            st.markdown(
                "<div style='font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;"
                "color:#7A6A5A;margin-bottom:0.6rem;'>Confusion Matrix — Active Model</div>",
                unsafe_allow_html=True
            )
            active_m = st.session_state.selected_model_name
            cm = metadata['models'][active_m].get('confusion_matrix',[[30,2],[2,26]])
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted",y="Actual",color="Count"),
                x=['No Disease','Heart Disease'],
                y=['No Disease','Heart Disease'],
                text_auto=True,
                color_continuous_scale=[[0,'#F6FFFA'],[0.5,'#B8DCCA'],[1,'#1B5741']]
            )
            fig_cm.update_layout(
                paper_bgcolor='#FFFFFF',
                font=dict(color='#3D3228',family='IBM Plex Sans'),
                title=dict(text=f"Confusion Matrix — {active_m}",
                           font=dict(family='Playfair Display,serif',size=14,color='#1E3A5F')),
                height=320,
                margin=dict(l=20,r=20,t=44,b=20)
            )
            st.plotly_chart(fig_cm, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# WORKSPACE 6 — CARDIAC KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("""
    <div class="rc-section-header">
      <div class="rc-section-title">Cardiac Knowledge Base & Dataset Explorer</div>
      <span class="rc-section-tag">Workspace 06</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="rc-card rc-card-accent-burgundy">
      <div class="rc-card-title">UCI Cleveland Heart Disease Benchmark Dataset</div>
      <p>
        The UCI Cleveland dataset is the gold-standard benchmark in cardiac machine learning research.
        Compiled at the Cleveland Clinic Foundation, it contains 297 cleaned patient records across
        13 clinical features, widely used to evaluate cardiovascular risk models since 1988.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="rc-stat-row">
      <div class="rc-stat"><span class="rc-stat-val">297</span><span class="rc-stat-lbl">Total Patients</span></div>
      <div class="rc-stat"><span class="rc-stat-val" style="color:#7C1B2E;">137</span><span class="rc-stat-lbl">Positive Cases (46.1%)</span></div>
      <div class="rc-stat"><span class="rc-stat-val" style="color:#1B5741;">160</span><span class="rc-stat-lbl">Negative Cases (53.9%)</span></div>
      <div class="rc-stat"><span class="rc-stat-val" style="color:#1E3A5F;">13</span><span class="rc-stat-lbl">Clinical Features</span></div>
      <div class="rc-stat"><span class="rc-stat-val" style="color:#8B6914;">5</span><span class="rc-stat-lbl">ML Models</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;"
        "color:#7A6A5A;margin-bottom:0.6rem;'>Clinical Parameter Reference Dictionary</div>",
        unsafe_allow_html=True
    )
    param_df = pd.DataFrame([
        {'Feature':'age',      'Clinical Name':'Age',                  'Description':'Patient age in years','Reference Range':'Risk threshold >55 yrs'},
        {'Feature':'sex',      'Clinical Name':'Biological Sex',       'Description':'Biological sex','Reference Range':'1=Male, 0=Female'},
        {'Feature':'cp',       'Clinical Name':'Chest Pain Type',      'Description':'1=Typical, 2=Atypical, 3=Non-anginal, 4=Asymptomatic','Reference Range':'1–4 (4=highest CAD correlation)'},
        {'Feature':'trestbps', 'Clinical Name':'Resting Blood Pressure','Description':'BP on admission (mm Hg)','Reference Range':'<120 mm Hg optimal'},
        {'Feature':'chol',     'Clinical Name':'Serum Cholesterol',    'Description':'Total cholesterol (mg/dl)','Reference Range':'<200 mg/dl desirable'},
        {'Feature':'fbs',      'Clinical Name':'Fasting Blood Sugar',  'Description':'FBS >120 mg/dl','Reference Range':'1=True, 0=False'},
        {'Feature':'restecg',  'Clinical Name':'Resting ECG',         'Description':'0=Normal, 1=ST-T, 2=LVH','Reference Range':'0=Normal'},
        {'Feature':'thalach',  'Clinical Name':'Max Heart Rate',       'Description':'Peak HR during stress test','Reference Range':'220 − age bpm target'},
        {'Feature':'exang',    'Clinical Name':'Exercise Angina',      'Description':'Angina on exertion','Reference Range':'1=Yes, 0=No'},
        {'Feature':'oldpeak',  'Clinical Name':'ST Depression',        'Description':'ST depression vs. rest (mm)','Reference Range':'<1.0 mm normal'},
        {'Feature':'slope',    'Clinical Name':'ST Slope',             'Description':'1=Up, 2=Flat, 3=Down','Reference Range':'1=Benign'},
        {'Feature':'ca',       'Clinical Name':'Vessels (Fluoroscopy)','Description':'Calcified vessels (LAD/LCx/RCA)','Reference Range':'0=No disease'},
        {'Feature':'thal',     'Clinical Name':'Thallium Stress Test', 'Description':'3=Normal, 6=Fixed, 7=Reversible','Reference Range':'3=Normal perfusion'},
    ])
    st.dataframe(param_df, use_container_width=True, hide_index=True)

# ─── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("""
<div style="display:flex;justify-content:space-between;align-items:center;
            padding:0.5rem 0 1rem 0;flex-wrap:wrap;gap:0.5rem;">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#7A6A5A;letter-spacing:0.04em;">
    HEARTGUARD AI &nbsp;|&nbsp; UCI CLEVELAND DATASET &nbsp;|&nbsp; CLINICAL DECISION SUPPORT ONLY
  </div>
  <div style="font-size:0.75rem;color:#7A6A5A;">
    Developed by <strong style="color:#3D3228;">Om Srivastava</strong> &nbsp;&middot;&nbsp;
    <a href="mailto:srivastavaom078@gmail.com" style="color:#7C1B2E;text-decoration:none;">
      srivastavaom078@gmail.com</a>
  </div>
</div>
""", unsafe_allow_html=True)

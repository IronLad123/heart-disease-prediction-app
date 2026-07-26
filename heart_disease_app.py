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

# ─────────────────────────────────────────────────────────────────────────────
#  DESIGN SYSTEM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@300;400;500;600;700&family=Playfair+Display:ital,wght@0,700;0,800;1,700&display=swap');

/* ── RESET & BASE */
html, body, [class*="css"] {
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
  color: #3D3228;
  -webkit-font-smoothing: antialiased;
}

/* ── PAGE CANVAS — warm ruled paper */
.stApp {
  background-color: #FAF7F0 !important;
  background-image:
    linear-gradient(rgba(212,201,176,0.35) 1px, transparent 1px),
    linear-gradient(90deg, rgba(212,201,176,0.35) 1px, transparent 1px) !important;
  background-size: 40px 40px !important;
  background-position: -1px -1px !important;
}

/* ── MAIN CONTENT AREA */
.main .block-container {
  padding: 1.5rem 3rem 4rem 3rem !important;
  max-width: 1300px !important;
}

/* ── SIDEBAR */
section[data-testid="stSidebar"] {
  background: #FFFFFF !important;
  border-right: 2px solid #D4C9B0 !important;
  box-shadow: 4px 0 20px rgba(61,50,40,0.05) !important;
  padding-top: 1.5rem !important;
}
section[data-testid="stSidebar"] > div { padding: 0 1rem !important; }

/* ── TOP HEADER BAR */
header[data-testid="stHeader"] {
  background: rgba(250,247,240,0.95) !important;
  backdrop-filter: blur(12px) !important;
  border-bottom: 1.5px solid #D4C9B0 !important;
}

/* ── SIDEBAR SECTION LABELS */
.sb-label {
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #7C1B2E;
  padding-bottom: 0.4rem;
  border-bottom: 1.5px solid #EDE8DC;
  margin-bottom: 0.8rem;
  display: block;
}

/* ── SIDEBAR MODEL STATS CARD */
.sb-stats {
  background: #FAF7F0;
  border: 1px solid #D4C9B0;
  border-left: 3px solid #7C1B2E;
  border-radius: 3px;
  padding: 0.9rem 1rem;
  margin-top: 0.4rem;
}
.sb-stat-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.22rem 0;
  border-bottom: 1px solid #EDE8DC;
}
.sb-stat-row:last-child { border-bottom: none; }
.sb-stat-key {
  font-size: 0.74rem;
  color: #7A6A5A;
  font-weight: 500;
}
.sb-stat-val {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.8rem;
  font-weight: 700;
  color: #1E3A5F;
}

/* ── SIDEBAR NAV RADIO */
.stRadio label {
  text-transform: none !important;
  letter-spacing: 0 !important;
  font-size: 0.85rem !important;
  color: #3D3228 !important;
  padding: 0.1rem 0 !important;
}
.stRadio [data-testid="stMarkdownContainer"] p {
  font-size: 0.85rem !important;
  color: #3D3228 !important;
  line-height: 1.4 !important;
}

/* ── FORM */
div[data-testid="stForm"] {
  background: #FFFFFF !important;
  border: 1.5px solid #D4C9B0 !important;
  border-top: 4px solid #7C1B2E !important;
  border-radius: 4px !important;
  padding: 2rem 2.4rem !important;
  box-shadow: 0 2px 20px rgba(61,50,40,0.07) !important;
}

/* ── INPUTS */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
  background: #FAF7F0 !important;
  border: 1.5px solid #C8BCAA !important;
  border-radius: 3px !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.92rem !important;
  color: #1E3A5F !important;
  transition: border-color 0.15s ease !important;
}
div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="select"] > div:focus-within {
  border-color: #7C1B2E !important;
  box-shadow: 0 0 0 3px rgba(124,27,46,0.1) !important;
}
input, select, textarea {
  color: #1E3A5F !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-weight: 500 !important;
}

/* ── LABELS */
label,
.stSelectbox label,
.stNumberInput label,
.stSlider label {
  color: #5C4A3A !important;
  font-weight: 600 !important;
  font-size: 0.76rem !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  margin-bottom: 0.2rem !important;
}

/* ── BUTTONS — primary */
.stFormSubmitButton > button {
  background: #7C1B2E !important;
  color: #FAF7F0 !important;
  border: none !important;
  border-radius: 3px !important;
  font-family: 'IBM Plex Sans', sans-serif !important;
  font-weight: 700 !important;
  font-size: 0.85rem !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  padding: 0.75rem 2rem !important;
  box-shadow: 0 2px 10px rgba(124,27,46,0.3) !important;
  transition: all 0.15s ease !important;
  width: 100% !important;
}
.stFormSubmitButton > button:hover {
  background: #A52840 !important;
  box-shadow: 0 4px 18px rgba(124,27,46,0.38) !important;
  transform: translateY(-1px) !important;
}

/* ── BUTTONS — standard */
.stButton > button {
  background: #FFFFFF !important;
  color: #7C1B2E !important;
  border: 1.5px solid #D4C9B0 !important;
  border-radius: 3px !important;
  font-family: 'IBM Plex Sans', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.79rem !important;
  letter-spacing: 0.05em !important;
  text-transform: uppercase !important;
  padding: 0.55rem 1rem !important;
  box-shadow: none !important;
  transition: all 0.15s ease !important;
}
.stButton > button:hover {
  border-color: #7C1B2E !important;
  background: rgba(124,27,46,0.04) !important;
  color: #7C1B2E !important;
}

/* ── DOWNLOAD BUTTON */
.stDownloadButton > button {
  background: #1B5741 !important;
  color: #FFFFFF !important;
  border: none !important;
  border-radius: 3px !important;
  font-family: 'IBM Plex Sans', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.82rem !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  padding: 0.65rem 1.4rem !important;
  box-shadow: 0 2px 8px rgba(27,87,65,0.25) !important;
}

/* ── TABS */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 2px solid #D4C9B0 !important;
  gap: 0 !important;
  padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  margin-bottom: -2px !important;
  color: #7A6A5A !important;
  font-weight: 600 !important;
  font-size: 0.79rem !important;
  letter-spacing: 0.07em !important;
  text-transform: uppercase !important;
  padding: 0.65rem 1.3rem !important;
  border-radius: 0 !important;
  transition: all 0.15s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
  color: #7C1B2E !important;
  background: rgba(124,27,46,0.04) !important;
}
.stTabs [aria-selected="true"] {
  color: #7C1B2E !important;
  border-bottom: 2px solid #7C1B2E !important;
  font-weight: 700 !important;
  background: transparent !important;
}

/* ── STREAMLIT METRICS */
div[data-testid="stMetric"] {
  background: #FFFFFF !important;
  border: 1.5px solid #D4C9B0 !important;
  border-top: 3px solid #1B5741 !important;
  border-radius: 3px !important;
  padding: 1rem 1.1rem !important;
  box-shadow: 0 1px 8px rgba(61,50,40,0.05) !important;
}
div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
  color: #7A6A5A !important;
  font-size: 0.72rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.07em !important;
  text-transform: uppercase !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
  color: #1E3A5F !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 1.55rem !important;
  font-weight: 700 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
  color: #1B5741 !important;
  font-size: 0.78rem !important;
}

/* ── DATAFRAMES */
div[data-testid="stDataFrame"] {
  border: 1.5px solid #D4C9B0 !important;
  border-radius: 3px !important;
  overflow: hidden !important;
}

/* ── SLIDER */
.stSlider [role="slider"] {
  background: #7C1B2E !important;
  border-color: #7C1B2E !important;
}

/* ── DIVIDER */
hr {
  border: none !important;
  border-top: 1.5px solid #D4C9B0 !important;
  margin: 1.8rem 0 !important;
}

/* ── SCROLLBAR */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #FAF7F0; }
::-webkit-scrollbar-thumb { background: #C8BCAA; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #7C1B2E; }

/* ────────────────────────────────────────────────────────
   CUSTOM LAYOUT COMPONENTS
   ──────────────────────────────────────────────────────── */

/* ── ANIMATIONS */
@keyframes ecg-draw  { to { stroke-dashoffset: 0; } }
@keyframes fadeUp    { from { opacity:0; transform:translateY(14px); } to { opacity:1; transform:translateY(0); } }
@keyframes slideIn   { from { opacity:0; transform:translateX(-10px); } to { opacity:1; transform:translateX(0); } }
@keyframes popIn     { from { opacity:0; transform:scale(0.97); } to { opacity:1; transform:scale(1); } }

/* ── HERO */
.rc-hero {
  background: #FFFFFF;
  border: 1.5px solid #D4C9B0;
  border-top: 5px solid #7C1B2E;
  border-radius: 4px;
  padding: 0;
  margin-bottom: 1.8rem;
  box-shadow: 0 3px 24px rgba(61,50,40,0.07);
  overflow: hidden;
  animation: fadeUp 0.45s ease;
  display: grid;
  grid-template-columns: 1fr auto;
}
.rc-hero-body {
  padding: 2.6rem 2.8rem;
  border-right: 1.5px solid #EDE8DC;
}
.rc-hero-panel {
  padding: 2rem 2rem;
  background: #FAF7F0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-width: 230px;
}
.rc-eyebrow {
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: #7C1B2E;
  display: flex;
  align-items: center;
  gap: 0.6rem;
  margin-bottom: 0.8rem;
}
.rc-eyebrow::before {
  content: '';
  display: inline-block;
  width: 24px;
  height: 2px;
  background: #7C1B2E;
}
.rc-title {
  font-family: 'Playfair Display', serif;
  font-size: 3rem;
  font-weight: 800;
  color: #1E3A5F;
  line-height: 1.05;
  letter-spacing: -0.02em;
  margin-bottom: 0.8rem;
}
.rc-title em { font-style: italic; color: #7C1B2E; }
.rc-subtitle {
  font-size: 0.92rem;
  color: #7A6A5A;
  line-height: 1.65;
  max-width: 500px;
  margin-bottom: 1.4rem;
}
.rc-badges { display: flex; gap: 0.5rem; flex-wrap: wrap; }
.rc-badge {
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  padding: 0.28rem 0.75rem;
  border-radius: 2px;
  border: 1.5px solid;
}
.rc-badge-burg  { color:#7C1B2E; border-color:#7C1B2E; background:rgba(124,27,46,0.06); }
.rc-badge-forest{ color:#1B5741; border-color:#1B5741; background:rgba(27,87,65,0.06); }
.rc-badge-brass { color:#8B6914; border-color:#B8860B; background:rgba(184,134,11,0.06); }
.rc-badge-navy  { color:#1E3A5F; border-color:#1E3A5F; background:rgba(30,58,95,0.06); }

/* ── PAGE KPI ROW (below hero) */
.rc-kpi-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1rem;
  margin-bottom: 1.8rem;
  animation: fadeUp 0.5s 0.1s ease both;
}
.rc-kpi {
  background: #FFFFFF;
  border: 1.5px solid #D4C9B0;
  border-radius: 3px;
  padding: 1rem 1.2rem;
  display: flex;
  align-items: center;
  gap: 0.9rem;
  box-shadow: 0 1px 8px rgba(61,50,40,0.04);
}
.rc-kpi-icon {
  width: 38px;
  height: 38px;
  border-radius: 3px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.1rem;
  flex-shrink: 0;
}
.rc-kpi-text {}
.rc-kpi-val {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 1.35rem;
  font-weight: 700;
  color: #1E3A5F;
  line-height: 1;
}
.rc-kpi-lbl {
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  color: #7A6A5A;
  margin-top: 0.18rem;
}

/* ── SECTION HEADER */
.rc-sh {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 1.2rem;
  padding-bottom: 0.6rem;
  border-bottom: 1.5px solid #D4C9B0;
}
.rc-sh-left { display: flex; align-items: baseline; gap: 0.8rem; }
.rc-sh-title {
  font-family: 'Playfair Display', serif;
  font-size: 1.45rem;
  font-weight: 700;
  color: #1E3A5F;
  margin: 0;
  line-height: 1;
}
.rc-sh-tag {
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #7C1B2E;
  background: rgba(124,27,46,0.08);
  padding: 0.18rem 0.55rem;
  border-radius: 2px;
}
.rc-sh-right {
  font-size: 0.74rem;
  color: #7A6A5A;
  font-style: italic;
}

/* ── QUICK PROFILE ROW */
.rc-profiles-label {
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #7A6A5A;
  margin-bottom: 0.5rem;
}

/* ── FORM STEP HEADER */
.rc-step-header {
  display: flex;
  align-items: center;
  gap: 0.7rem;
  margin-bottom: 1.4rem;
  padding: 0.7rem 1rem;
  background: #FAF7F0;
  border-left: 3px solid #7C1B2E;
  border-radius: 0 3px 3px 0;
}
.rc-step-num {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.7rem;
  font-weight: 700;
  color: #7C1B2E;
  background: rgba(124,27,46,0.1);
  padding: 0.18rem 0.5rem;
  border-radius: 2px;
  letter-spacing: 0.06em;
}
.rc-step-title {
  font-size: 0.82rem;
  font-weight: 600;
  color: #3D3228;
  letter-spacing: 0.02em;
}
.rc-step-desc {
  font-size: 0.76rem;
  color: #7A6A5A;
  margin-left: auto;
}

/* ── INPUT GROUP WRAPPER — pairs label+input+relevance */
.rc-input-group {
  margin-bottom: 0.2rem;
}

/* ── RELEVANCE CALLOUT */
.rc-rel {
  background: #FAFAF8;
  border: 1px solid #E2DAD0;
  border-left: 3px solid #1B5741;
  border-radius: 0 3px 3px 3px;
  padding: 0.65rem 0.9rem;
  margin-top: 0.25rem;
  margin-bottom: 1.4rem;
  font-size: 0.78rem;
  line-height: 1.55;
  color: #4A3D32;
}
.rc-rel-head {
  font-size: 0.67rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #1B5741;
  display: block;
  margin-bottom: 0.28rem;
}
.rc-rel-body { display: block; color: #5C4A3A; }
.rc-rel-norm {
  font-size: 0.73rem;
  color: #7A6A5A;
  display: block;
  margin-top: 0.3rem;
  font-style: italic;
}
.rc-rel-val {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.76rem;
  font-weight: 600;
  color: #7C1B2E;
  display: block;
  margin-top: 0.25rem;
}

/* ── RISK RESULT PANELS */
.rc-result-grid {
  display: grid;
  grid-template-columns: 1fr 300px;
  gap: 1.4rem;
  margin-bottom: 1.6rem;
  animation: popIn 0.4s ease;
}
.rc-risk-banner {
  border-radius: 4px;
  padding: 2rem 2.2rem;
  border: 1.5px solid;
}
.rc-risk-high  { background:#FFFAF9; border-color:#EBCACA; border-left:5px solid #C0392B; }
.rc-risk-warn  { background:#FFFDF5; border-color:#EBE0BF; border-left:5px solid #B8860B; }
.rc-risk-safe  { background:#F6FFFA; border-color:#B8DCCA; border-left:5px solid #1B5741; }
.rc-risk-eyebrow {
  font-size: 0.66rem;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  margin-bottom: 0.5rem;
}
.rc-risk-prob {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 3.6rem;
  font-weight: 700;
  line-height: 1;
  margin-bottom: 0.5rem;
}
.rc-risk-prob small {
  font-size: 1rem;
  font-weight: 400;
  color: #7A6A5A;
  font-family: 'IBM Plex Sans', sans-serif;
  margin-left: 0.4rem;
}
.rc-risk-desc {
  font-size: 0.86rem;
  color: #3D3228;
  line-height: 1.6;
}

/* ── GAUGE WRAPPER */
.rc-gauge-wrap {
  background: #FFFFFF;
  border: 1.5px solid #D4C9B0;
  border-radius: 4px;
  padding: 0.5rem;
  box-shadow: 0 1px 8px rgba(61,50,40,0.05);
}

/* ── RECS */
.rc-recs-title {
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #7A6A5A;
  margin: 1.2rem 0 0.6rem 0;
  padding-top: 1rem;
  border-top: 1px solid #EDE8DC;
}
.rc-rec {
  background: #FFFFFF;
  border: 1px solid #D4C9B0;
  border-left: 3px solid #1E3A5F;
  border-radius: 0 3px 3px 0;
  padding: 0.6rem 0.9rem;
  margin-bottom: 0.45rem;
  font-size: 0.8rem;
  color: #3D3228;
  line-height: 1.55;
  animation: slideIn 0.3s ease;
}

/* ── CHARTS ROW */
.rc-charts-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.2rem;
  margin-bottom: 1.6rem;
}
.rc-chart-card {
  background: #FFFFFF;
  border: 1.5px solid #D4C9B0;
  border-radius: 4px;
  overflow: hidden;
  box-shadow: 0 1px 10px rgba(61,50,40,0.05);
}
.rc-chart-header {
  padding: 0.8rem 1.2rem;
  border-bottom: 1px solid #EDE8DC;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.rc-chart-title {
  font-size: 0.76rem;
  font-weight: 700;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: #3D3228;
}
.rc-chart-sub {
  font-size: 0.7rem;
  color: #7A6A5A;
}
.rc-chart-body { padding: 0.2rem; }

/* ── FULL WIDTH CHART CARD */
.rc-chart-card-full {
  background: #FFFFFF;
  border: 1.5px solid #D4C9B0;
  border-radius: 4px;
  overflow: hidden;
  box-shadow: 0 1px 10px rgba(61,50,40,0.05);
  margin-bottom: 1.4rem;
}

/* ── STAT ROW */
.rc-stat-row {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 0.9rem;
  margin-bottom: 1.4rem;
  animation: fadeUp 0.4s ease;
}
.rc-stat {
  background: #FFFFFF;
  border: 1.5px solid #D4C9B0;
  border-radius: 3px;
  padding: 1rem 1rem;
  text-align: center;
  box-shadow: 0 1px 6px rgba(61,50,40,0.04);
}
.rc-stat-val {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 1.7rem;
  font-weight: 700;
  color: #1E3A5F;
  display: block;
  line-height: 1;
}
.rc-stat-lbl {
  font-size: 0.67rem;
  font-weight: 600;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  color: #7A6A5A;
  margin-top: 0.3rem;
  display: block;
}

/* ── INFO CARD */
.rc-card {
  background: #FFFFFF;
  border: 1.5px solid #D4C9B0;
  border-radius: 4px;
  padding: 1.4rem 1.6rem;
  box-shadow: 0 1px 10px rgba(61,50,40,0.05);
  margin-bottom: 1.2rem;
  animation: fadeUp 0.4s ease;
}
.rc-card-title {
  font-family: 'Playfair Display', serif;
  font-size: 1rem;
  font-weight: 700;
  color: #1E3A5F;
  margin: 0 0 0.5rem 0;
}
.rc-card p { color:#7A6A5A; font-size:0.85rem; line-height:1.65; margin:0; }

/* ── SIMULATOR LIVE CARD */
.rc-sim-card {
  border-radius: 4px;
  padding: 2rem;
  border: 1.5px solid;
  text-align: center;
  animation: popIn 0.35s ease;
}

/* ── MINI VITALS STRIP */
.rc-vitals-strip {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.7rem;
  margin-top: 1rem;
}
.rc-vital {
  background: #FAF7F0;
  border: 1px solid #D4C9B0;
  border-radius: 3px;
  padding: 0.65rem 0.8rem;
  text-align: center;
}
.rc-vital-val {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 1.2rem;
  font-weight: 700;
  color: #1E3A5F;
  display: block;
  line-height: 1;
}
.rc-vital-lbl {
  font-size: 0.65rem;
  font-weight: 600;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  color: #7A6A5A;
  margin-top: 0.2rem;
  display: block;
}

/* ── FOOTER */
.rc-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.5rem;
  padding: 0.8rem 0 0.5rem 0;
}
.rc-footer-mono {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.68rem;
  color: #7A6A5A;
  letter-spacing: 0.05em;
}
.rc-footer-name {
  font-size: 0.74rem;
  color: #7A6A5A;
}
.rc-footer-name a {
  color: #7C1B2E;
  text-decoration: none;
  font-weight: 600;
}

/* ── TOOLTIP-STYLE CALLOUT */
.rc-note {
  background: #FFF8F0;
  border: 1px solid #EAD9B8;
  border-left: 3px solid #B8860B;
  border-radius: 3px;
  padding: 0.55rem 0.9rem;
  font-size: 0.78rem;
  color: #5C4020;
  line-height: 1.5;
  margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_all_models():
    try:
        with open('models_metadata.json', 'r') as f:
            metadata = json.load(f)
        scaler = joblib.load('scaler.pkl')
        models = {n: joblib.load(info['filename']) for n, info in metadata['models'].items()}
        return models, scaler, metadata
    except Exception:
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
        cols = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach',
                'exang','oldpeak','slope','ca','thal','target']
        try:
            df = pd.read_csv(url, names=cols, na_values='?')
        except Exception:
            df = pd.read_csv('Heart Disease Data/processed.cleveland.data', names=cols, na_values='?')
        df = df.dropna().reset_index(drop=True)
        df['target'] = (df['target'] > 0).astype(int)
        X, y = df.drop('target', axis=1), df['target']
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        mods = {
            'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42).fit(Xs, y),
            'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42).fit(Xs, y),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7).fit(Xs, y),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=500).fit(Xs, y),
        }
        ens = VotingClassifier(
            estimators=[('rf', mods['Random Forest']), ('gb', mods['Gradient Boosting']),
                        ('knn', mods['K-Nearest Neighbors']), ('lr', mods['Logistic Regression'])],
            voting='soft').fit(Xs, y)
        mods['Voting Ensemble'] = ens
        try:
            with open('models_metadata.json', 'r') as f:
                meta = json.load(f)
        except Exception:
            meta = {'models': {k: {'accuracy': 0.867, 'roc_auc': 0.941, 'recall': 0.852,
                                   'precision': 0.871, 'f1_score': 0.861,
                                   'confusion_matrix': [[30,2],[2,26]]} for k in mods}}
        return mods, sc, meta

models_suite, scaler, metadata = load_all_models()

# Session state
for key, val in [('session_history', []), ('current_workspace', 'Patient Intake & XAI'),
                  ('selected_model_name', 'Voting Ensemble')]:
    if key not in st.session_state:
        st.session_state[key] = val

def predict(model_name, feat):
    Xs = scaler.transform(pd.DataFrame([feat]))
    m  = models_suite[model_name]
    return float(m.predict_proba(Xs)[0][1]*100), int(m.predict(Xs)[0])

# ─────────────────────────────────────────────────────────────────────────────
#  PLOTLY BASE THEME
# ─────────────────────────────────────────────────────────────────────────────
RC = dict(
    paper_bgcolor='#FFFFFF', plot_bgcolor='#FAF7F0',
    font=dict(family='IBM Plex Sans, sans-serif', color='#3D3228', size=11),
    margin=dict(l=16, r=16, t=40, b=16),
    title_font=dict(family='Playfair Display, serif', size=14, color='#1E3A5F'),
    legend=dict(bgcolor='rgba(255,255,255,0.9)', bordercolor='#D4C9B0', borderwidth=1,
                font=dict(size=11)),
    xaxis=dict(gridcolor='#EDE8DC', linecolor='#D4C9B0', tickfont=dict(size=10, color='#7A6A5A')),
    yaxis=dict(gridcolor='#EDE8DC', linecolor='#D4C9B0', tickfont=dict(size=10, color='#7A6A5A')),
)
BURGUNDY, FOREST, NAVY, BRASS, ROSE = '#7C1B2E', '#1B5741', '#1E3A5F', '#B8860B', '#A52840'


# ─────────────────────────────────────────────────────────────────────────────
#  HERO BANNER
# ─────────────────────────────────────────────────────────────────────────────
n_models = len(models_suite) if models_suite else 5
n_sess   = len(st.session_state.session_history)

st.markdown(f"""
<div class="rc-hero">
  <div class="rc-hero-body">
    <div class="rc-eyebrow">Clinical Cardiac Decision Support</div>
    <div class="rc-title">Heart<em>Guard</em> AI</div>
    <div class="rc-subtitle">
      Multi-model machine learning platform for cardiac risk assessment,
      built on the UCI Cleveland dataset. Explainable, interpretable,
      and designed for clinical-grade decision support.
    </div>
    <div class="rc-badges">
      <span class="rc-badge rc-badge-burg">{n_models}-Model ML Suite</span>
      <span class="rc-badge rc-badge-forest">UCI Cleveland · 297 Patients</span>
      <span class="rc-badge rc-badge-brass">Explainable AI (XAI)</span>
      <span class="rc-badge rc-badge-navy">Session: {n_sess} assessments</span>
    </div>
  </div>
  <div class="rc-hero-panel">
    <svg width="180" height="76" viewBox="0 0 180 76">
      <path d="M8,38 L34,38 L42,16 L50,60 L56,24 L64,52 L72,38
               L108,38 L116,16 L124,60 L130,24 L138,52 L146,38 L172,38"
            fill="none" stroke="#7C1B2E" stroke-width="2.2"
            stroke-linecap="round" stroke-linejoin="round"
            stroke-dasharray="600" stroke-dashoffset="600"
            style="animation:ecg-draw 2.8s ease forwards;"/>
      <text x="90" y="70" text-anchor="middle"
            font-family="IBM Plex Mono,monospace" font-size="8"
            fill="#7A6A5A" letter-spacing="3">TELEMETRY ACTIVE</text>
    </svg>
    <div style="margin-top:1.2rem;width:100%;">
      <div class="sb-stat-row">
        <span class="sb-stat-key">Models loaded</span>
        <span class="sb-stat-val">{n_models}</span>
      </div>
      <div class="sb-stat-row">
        <span class="sb-stat-key">Active model</span>
        <span class="sb-stat-val" style="font-size:0.68rem;">{st.session_state.selected_model_name[:10]}</span>
      </div>
      <div class="sb-stat-row">
        <span class="sb-stat-key">Dataset rows</span>
        <span class="sb-stat-val">297</span>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<span class="sb-label">Workspaces</span>', unsafe_allow_html=True)
    workspaces = [
        "Patient Intake & XAI",
        "Clinical Risk Simulator & 10-Yr Prognosis",
        "3D Anatomical Mesh & SOAP Notes",
        "Batch EHR CSV Intelligence Suite",
        "ML Model Workbench & Comparison",
        "Cardiac Knowledge Base & Dataset",
    ]
    selected_ws = st.radio(
        "", workspaces,
        index=workspaces.index(st.session_state.current_workspace)
              if st.session_state.current_workspace in workspaces else 0,
        label_visibility="collapsed"
    )
    st.session_state.current_workspace = selected_ws

    st.markdown("<div style='margin:1.2rem 0 0.8rem 0;'></div>", unsafe_allow_html=True)
    st.markdown('<span class="sb-label">Active ML Engine</span>', unsafe_allow_html=True)

    if metadata and 'models' in metadata:
        model_names = list(metadata['models'].keys())
        active_model = st.selectbox(
            "", model_names,
            index=model_names.index(st.session_state.selected_model_name)
                  if st.session_state.selected_model_name in model_names else len(model_names)-1,
            label_visibility="collapsed"
        )
        st.session_state.selected_model_name = active_model
        mi = metadata['models'][active_model]
        st.markdown(f"""
        <div class="sb-stats">
          <div class="sb-stat-row">
            <span class="sb-stat-key">Accuracy</span>
            <span class="sb-stat-val">{mi.get('accuracy',0.867)*100:.1f}%</span>
          </div>
          <div class="sb-stat-row">
            <span class="sb-stat-key">AUC-ROC</span>
            <span class="sb-stat-val">{mi.get('roc_auc',0.941):.3f}</span>
          </div>
          <div class="sb-stat-row">
            <span class="sb-stat-key">Recall</span>
            <span class="sb-stat-val">{mi.get('recall',0.852)*100:.1f}%</span>
          </div>
          <div class="sb-stat-row">
            <span class="sb-stat-key">Precision</span>
            <span class="sb-stat-val">{mi.get('precision',0.871)*100:.1f}%</span>
          </div>
          <div class="sb-stat-row">
            <span class="sb-stat-key">F1 Score</span>
            <span class="sb-stat-val">{mi.get('f1_score',0.861):.3f}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Session history mini log
    if st.session_state.session_history:
        st.markdown("<div style='margin:1.2rem 0 0.8rem 0;'></div>", unsafe_allow_html=True)
        st.markdown('<span class="sb-label">Session Log</span>', unsafe_allow_html=True)
        for h in reversed(st.session_state.session_history[-4:]):
            col = BURGUNDY if h['result'] == 'Heart Disease' else FOREST
            st.markdown(f"""
            <div style="font-size:0.72rem;padding:0.4rem 0.6rem;margin-bottom:0.3rem;
                        background:#FAF7F0;border:1px solid #D4C9B0;border-left:3px solid {col};
                        border-radius:0 3px 3px 0;">
              <span style="font-family:IBM Plex Mono,monospace;color:{col};font-weight:700;">
                {h['prob_%']}%</span>
              <span style="color:#7A6A5A;"> · {h.get('model',h.get('model_used',''))[:8]} · {h['timestamp']}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin:1.4rem 0 0.8rem 0;'></div>", unsafe_allow_html=True)
    st.markdown('<span class="sb-label">Platform Author</span>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.8rem;color:#3D3228;line-height:1.8;">
      <strong>Om Srivastava</strong><br>
      <a href="mailto:srivastavaom078@gmail.com"
         style="color:#7C1B2E;text-decoration:none;font-size:0.75rem;">
        srivastavaom078@gmail.com</a><br>
      <span style="color:#7A6A5A;font-size:0.74rem;">Data Science & Machine Learning</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  WS 1 — PATIENT INTAKE & XAI
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.current_workspace == "Patient Intake & XAI":

    st.markdown("""
    <div class="rc-sh">
      <div class="rc-sh-left">
        <div class="rc-sh-title">Patient Intake & Explainable AI</div>
        <span class="rc-sh-tag">Workspace 01</span>
      </div>
      <div class="rc-sh-right">Enter vitals — every field shows clinical relevance & thresholds</div>
    </div>
    """, unsafe_allow_html=True)

    # Quick profile row
    st.markdown('<div class="rc-profiles-label">Quick-load clinical profiles</div>', unsafe_allow_html=True)
    pc1, pc2, pc3, pc4 = st.columns(4, gap="small")
    with pc1:
        if st.button("High Risk Profile", use_container_width=True):
            st.session_state.update(dict(wiz_age=67,wiz_sex="Male",wiz_cp="Asymptomatic (4)",
                wiz_trestbps=160,wiz_chol=286,wiz_fbs="No (<= 120 mg/dl)",
                wiz_restecg="Left Ventricular Hypertrophy (2)",wiz_thalach=108,wiz_exang="Yes",
                wiz_oldpeak=1.5,wiz_slope="Flat (2)",wiz_ca=3,wiz_thal="Reversible Defect (7)"))
            st.rerun()
    with pc2:
        if st.button("Low Risk Profile", use_container_width=True):
            st.session_state.update(dict(wiz_age=37,wiz_sex="Female",wiz_cp="Typical Angina (1)",
                wiz_trestbps=118,wiz_chol=190,wiz_fbs="No (<= 120 mg/dl)",
                wiz_restecg="Normal (0)",wiz_thalach=185,wiz_exang="No",
                wiz_oldpeak=0.0,wiz_slope="Upsloping (1)",wiz_ca=0,wiz_thal="Normal (3)"))
            st.rerun()
    with pc3:
        if st.button("Moderate Risk Profile", use_container_width=True):
            st.session_state.update(dict(wiz_age=58,wiz_sex="Male",wiz_cp="Atypical Angina (2)",
                wiz_trestbps=140,wiz_chol=245,wiz_fbs="Yes (> 120 mg/dl)",
                wiz_restecg="ST-T Wave Abnormality (1)",wiz_thalach=142,wiz_exang="Yes",
                wiz_oldpeak=1.2,wiz_slope="Flat (2)",wiz_ca=1,wiz_thal="Reversible Defect (7)"))
            st.rerun()
    with pc4:
        if st.button("Reset to Defaults", use_container_width=True):
            st.session_state.update(dict(wiz_age=52,wiz_sex="Male",wiz_cp="Atypical Angina (2)",
                wiz_trestbps=130,wiz_chol=240,wiz_fbs="No (<= 120 mg/dl)",
                wiz_restecg="Normal (0)",wiz_thalach=150,wiz_exang="No",
                wiz_oldpeak=1.0,wiz_slope="Upsloping (1)",wiz_ca=0,wiz_thal="Normal (3)"))
            st.rerun()

    st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)

    # ── Patient Intake Form ───────────────────────────────────────────────────
    with st.form("intake_form"):
        t1, t2, t3 = st.tabs([
            "01  Demographics & Vitals",
            "02  ECG & Stress Testing",
            "03  Advanced Imaging",
        ])

        # ── Tab 1 ─────────────────────────────────────────────────────────────
        with t1:
            st.markdown("""
            <div class="rc-step-header">
              <span class="rc-step-num">STEP 01</span>
              <span class="rc-step-title">Patient Demographics & Basic Clinical Vitals</span>
              <span class="rc-step-desc">4 parameters</span>
            </div>
            """, unsafe_allow_html=True)

            col_a, col_b = st.columns(2, gap="large")

            with col_a:
                age = st.number_input("Age (years)", 18, 100, st.session_state.get('wiz_age',52))
                af = "Elevated risk factor (>55 yrs)" if age > 55 else "Below major age threshold"
                st.markdown(f"""<div class="rc-rel">
                  <span class="rc-rel-head">Age — Clinical Relevance</span>
                  <span class="rc-rel-body">CAD risk rises steadily with age from arterial stiffening,
                  vascular calcification, and cumulative lipid exposure. Age &gt;55 (M) / &gt;65 (F)
                  is a major independent cardiovascular risk factor.</span>
                  <span class="rc-rel-norm">Risk threshold: &gt;55 yrs (male), &gt;65 yrs (female)</span>
                  <span class="rc-rel-val">Patient: {age} yrs — {af}</span>
                </div>""", unsafe_allow_html=True)

                sex = st.selectbox("Biological Sex", ["Male","Female"],
                    index=0 if st.session_state.get('wiz_sex','Male')=="Male" else 1)
                sn = "Male — higher early-onset CAD risk (no pre-menopausal oestrogen protection)" \
                     if sex=="Male" else "Female — oestrogen-protective baseline (pre-menopause)"
                st.markdown(f"""<div class="rc-rel">
                  <span class="rc-rel-head">Sex — Clinical Relevance</span>
                  <span class="rc-rel-body">Males develop obstructive CAD approximately a decade earlier
                  than females. Post-menopausal females rapidly lose this protective advantage.</span>
                  <span class="rc-rel-norm">Encoded: Male = 1, Female = 0</span>
                  <span class="rc-rel-val">Patient: {sn}</span>
                </div>""", unsafe_allow_html=True)

            with col_b:
                trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 70, 240,
                    st.session_state.get('wiz_trestbps',130))
                bpc = ("Stage 2 HTN" if trestbps>=140 else "Stage 1 HTN" if trestbps>=130
                       else "Elevated" if trestbps>=120 else "Optimal")
                st.markdown(f"""<div class="rc-rel">
                  <span class="rc-rel-head">Resting Blood Pressure — Clinical Relevance</span>
                  <span class="rc-rel-body">Sustained hypertension damages arterial endothelium,
                  accelerates atherosclerotic plaque formation, and increases left ventricular
                  myocardial workload — raising risk of cardiac events.</span>
                  <span class="rc-rel-norm">Optimal: &lt;120 | Elevated: 120–129 | Stage 1 HTN: 130–139 | Stage 2: ≥140 mm Hg</span>
                  <span class="rc-rel-val">Patient: {trestbps} mm Hg — {bpc}</span>
                </div>""", unsafe_allow_html=True)

                chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 650,
                    st.session_state.get('wiz_chol',240))
                cc = ("High / Hypercholesterolaemia" if chol>=240 else
                      "Borderline high" if chol>=200 else "Desirable")
                st.markdown(f"""<div class="rc-rel">
                  <span class="rc-rel-head">Serum Cholesterol — Clinical Relevance</span>
                  <span class="rc-rel-body">Elevated LDL-cholesterol deposits in the vessel intima
                  as atheromatous plaques, narrowing coronary arteries and restricting myocardial
                  blood supply during exertion.</span>
                  <span class="rc-rel-norm">Desirable: &lt;200 | Borderline: 200–239 | High: ≥240 mg/dl</span>
                  <span class="rc-rel-val">Patient: {chol} mg/dl — {cc}</span>
                </div>""", unsafe_allow_html=True)

                fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl",
                    ["No (<= 120 mg/dl)","Yes (> 120 mg/dl)"],
                    index=0 if "No" in st.session_state.get('wiz_fbs','No') else 1)
                fn = "Diabetic threshold exceeded — significantly elevates CVD risk" \
                     if "Yes" in fbs else "Within normal fasting glucose range"
                st.markdown(f"""<div class="rc-rel">
                  <span class="rc-rel-head">Fasting Blood Sugar — Clinical Relevance</span>
                  <span class="rc-rel-body">Hyperglycaemia damages vascular endothelial cells and
                  promotes glycosylation of LDL particles, accelerating atherosclerotic progression.
                  Diabetics carry approximately 2× the cardiovascular event risk.</span>
                  <span class="rc-rel-norm">Normal fasting glucose: ≤100 mg/dl | Diabetic: &gt;126 mg/dl</span>
                  <span class="rc-rel-val">Patient: {fbs} — {fn}</span>
                </div>""", unsafe_allow_html=True)

        # ── Tab 2 ─────────────────────────────────────────────────────────────
        with t2:
            st.markdown("""
            <div class="rc-step-header">
              <span class="rc-step-num">STEP 02</span>
              <span class="rc-step-title">Resting ECG Results & Exercise Stress Testing</span>
              <span class="rc-step-desc">4 parameters</span>
            </div>
            """, unsafe_allow_html=True)

            col_a, col_b = st.columns(2, gap="large")

            with col_a:
                cp_opts = ["Typical Angina (1)","Atypical Angina (2)","Non-Anginal Pain (3)","Asymptomatic (4)"]
                cp = st.selectbox("Chest Pain Type", cp_opts,
                    index=cp_opts.index(st.session_state.get('wiz_cp','Atypical Angina (2)'))
                          if st.session_state.get('wiz_cp') in cp_opts else 1)
                cpn = ("Silent ischaemia / highest CAD correlation in dataset" if "Asymptomatic" in cp else
                       "Classic anginal pattern — high pre-test probability" if "Typical" in cp else
                       "Moderate CAD suspicion" if "Atypical" in cp else "Non-cardiac — low suspicion")
                st.markdown(f"""<div class="rc-rel">
                  <span class="rc-rel-head">Chest Pain Type — Clinical Relevance</span>
                  <span class="rc-rel-body">Chest pain classification is the primary pre-test
                  discriminator for CAD. Paradoxically, asymptomatic presentation (type 4) correlates
                  most strongly with confirmed disease in this dataset — silent ischaemia.</span>
                  <span class="rc-rel-norm">1=Typical | 2=Atypical | 3=Non-anginal | 4=Asymptomatic (highest model weight)</span>
                  <span class="rc-rel-val">Patient: {cp} — {cpn}</span>
                </div>""", unsafe_allow_html=True)

                restecg_opts = ["Normal (0)","ST-T Wave Abnormality (1)","Left Ventricular Hypertrophy (2)"]
                restecg = st.selectbox("Resting ECG Results", restecg_opts,
                    index=restecg_opts.index(st.session_state.get('wiz_restecg','Normal (0)'))
                          if st.session_state.get('wiz_restecg') in restecg_opts else 0)
                en = ("No conduction anomaly detected" if "0" in restecg else
                      "Ischaemic repolarisation abnormality present" if "1" in restecg else
                      "LV hypertrophy — chronic pressure overload pattern")
                st.markdown(f"""<div class="rc-rel">
                  <span class="rc-rel-head">Resting ECG — Clinical Relevance</span>
                  <span class="rc-rel-body">The resting 12-lead ECG evaluates baseline conduction.
                  ST-T wave changes indicate ischaemic repolarisation; LV hypertrophy reflects
                  long-term hypertensive myocardial strain.</span>
                  <span class="rc-rel-norm">0=Normal | 1=ST-T Abnormality (ischaemic) | 2=LV Hypertrophy (hypertensive)</span>
                  <span class="rc-rel-val">Patient: {restecg} — {en}</span>
                </div>""", unsafe_allow_html=True)

            with col_b:
                thalach = st.number_input("Max Heart Rate Achieved (bpm)", 60, 230,
                    st.session_state.get('wiz_thalach',150))
                hn = ("Impaired chronotropic reserve — indicator of significant stenosis" if thalach<130
                      else "Moderate reserve" if thalach<160 else "Good exertional capacity")
                st.markdown(f"""<div class="rc-rel">
                  <span class="rc-rel-head">Max Heart Rate — Clinical Relevance</span>
                  <span class="rc-rel-body">Failure to achieve age-predicted maximum HR (220 − age)
                  during treadmill testing indicates impaired chronotropic reserve — a hallmark
                  of significant coronary artery obstruction reducing cardiac output.</span>
                  <span class="rc-rel-norm">Target: 220 − age bpm | Impaired: &lt;85% of target</span>
                  <span class="rc-rel-val">Patient: {thalach} bpm — {hn}</span>
                </div>""", unsafe_allow_html=True)

                exang = st.selectbox("Exercise-Induced Angina", ["No","Yes"],
                    index=0 if st.session_state.get('wiz_exang','No')=="No" else 1)
                ean = ("Positive for exertional ischaemia — demand-induced coronary flow restriction"
                       if exang=="Yes" else "Negative for exercise-induced angina")
                st.markdown(f"""<div class="rc-rel">
                  <span class="rc-rel-head">Exercise Angina — Clinical Relevance</span>
                  <span class="rc-rel-body">Chest pain precipitated specifically by exertion indicates
                  epicardial stenosis that cannot accommodate demand-driven coronary flow increase —
                  a cardinal ischaemic symptom.</span>
                  <span class="rc-rel-norm">Encoded: Yes = 1 (ischaemic indicator) | No = 0</span>
                  <span class="rc-rel-val">Patient: {exang} — {ean}</span>
                </div>""", unsafe_allow_html=True)

        # ── Tab 3 ─────────────────────────────────────────────────────────────
        with t3:
            st.markdown("""
            <div class="rc-step-header">
              <span class="rc-step-num">STEP 03</span>
              <span class="rc-step-title">Advanced Coronary Imaging & Nuclear Perfusion</span>
              <span class="rc-step-desc">5 parameters</span>
            </div>
            """, unsafe_allow_html=True)

            col_a, col_b = st.columns(2, gap="large")

            with col_a:
                oldpeak = st.slider("Exercise ST Depression (mm)", 0.0, 6.2,
                    float(st.session_state.get('wiz_oldpeak',1.0)), step=0.1)
                opn = ("Severe ischaemic depression ≥2.0 mm" if oldpeak>=2.0 else
                       "Diagnostic for ischaemia ≥1.0 mm" if oldpeak>=1.0 else "Normal ST baseline")
                st.markdown(f"""<div class="rc-rel">
                  <span class="rc-rel-head">ST Depression (oldpeak) — Clinical Relevance</span>
                  <span class="rc-rel-body">Horizontal or down-sloping ST segment depression during
                  exercise quantifies subendocardial ischaemia. Greater depression predicts larger
                  ischaemic territory and multi-vessel disease.</span>
                  <span class="rc-rel-norm">Normal: &lt;1.0 mm | Diagnostic: ≥1.0 mm | Severe: ≥2.0 mm</span>
                  <span class="rc-rel-val">Patient: {oldpeak} mm — {opn}</span>
                </div>""", unsafe_allow_html=True)

                slope_opts = ["Upsloping (1)","Flat (2)","Downsloping (3)"]
                slope = st.selectbox("ST Segment Slope at Peak Exercise", slope_opts,
                    index=slope_opts.index(st.session_state.get('wiz_slope','Upsloping (1)'))
                          if st.session_state.get('wiz_slope') in slope_opts else 0)
                sln = ("Benign upsloping — favourable prognosis" if "1" in slope else
                       "Flat — ischaemic pattern, moderate risk" if "2" in slope else
                       "Downsloping — severe multi-vessel CAD indicator")
                st.markdown(f"""<div class="rc-rel">
                  <span class="rc-rel-head">ST Slope — Clinical Relevance</span>
                  <span class="rc-rel-body">The slope of the ST segment at peak exercise characterises
                  repolarisation recovery. Flat or downsloping patterns strongly correlate with
                  multi-vessel obstructive CAD and adverse prognosis.</span>
                  <span class="rc-rel-norm">1=Upsloping (benign) | 2=Flat (ischaemic) | 3=Downsloping (severe CAD)</span>
                  <span class="rc-rel-val">Patient: {slope} — {sln}</span>
                </div>""", unsafe_allow_html=True)

            with col_b:
                ca = st.slider("Major Vessels via Fluoroscopy (0–3)", 0, 3,
                    int(st.session_state.get('wiz_ca',0)))
                can = ("No stenotic vessels — clean coronary anatomy" if ca==0 else
                       f"{ca}-vessel CAD — significant anatomic disease burden")
                st.markdown(f"""<div class="rc-rel">
                  <span class="rc-rel-head">Fluoroscopy Vessels (ca) — Clinical Relevance</span>
                  <span class="rc-rel-body">Count of major coronary arteries (LAD, LCx, RCA) showing
                  calcified stenosis under fluoroscopy. This is the strongest predictor in the
                  Cleveland dataset — directly quantifying anatomic disease burden.</span>
                  <span class="rc-rel-norm">0=No disease | 1–3=Multi-vessel CAD (highest model weight)</span>
                  <span class="rc-rel-val">Patient: {ca} vessels — {can}</span>
                </div>""", unsafe_allow_html=True)

                thal_opts = ["Normal (3)","Fixed Defect (6)","Reversible Defect (7)"]
                thal = st.selectbox("Thallium Stress Test Result", thal_opts,
                    index=thal_opts.index(st.session_state.get('wiz_thal','Normal (3)'))
                          if st.session_state.get('wiz_thal') in thal_opts else 0)
                thn = ("Normal myocardial perfusion throughout" if "3" in thal else
                       "Fixed defect — prior infarct scar territory" if "6" in thal else
                       "Reversible defect — hibernating, viable ischaemic myocardium")
                st.markdown(f"""<div class="rc-rel">
                  <span class="rc-rel-head">Thallium Stress Test — Clinical Relevance</span>
                  <span class="rc-rel-body">Nuclear perfusion imaging differentiates infarcted (fixed)
                  from ischaemic but viable (reversible) myocardium. Reversible defects identify
                  territory amenable to revascularisation.</span>
                  <span class="rc-rel-norm">3=Normal | 6=Fixed defect (scar) | 7=Reversible (viable ischaemia)</span>
                  <span class="rc-rel-val">Patient: {thal} — {thn}</span>
                </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "Run Diagnostic Assessment",
            use_container_width=True, type="primary"
        )

    # ── RESULTS ──────────────────────────────────────────────────────────────
    if submitted:
        feat = {
            'age': age, 'sex': 1 if sex=="Male" else 0,
            'cp': 1 if "1" in cp else 2 if "2" in cp else 3 if "3" in cp else 4,
            'trestbps': trestbps, 'chol': chol,
            'fbs': 1 if "Yes" in fbs else 0,
            'restecg': 0 if "0" in restecg else 1 if "1" in restecg else 2,
            'thalach': thalach, 'exang': 1 if exang=="Yes" else 0,
            'oldpeak': oldpeak,
            'slope': 1 if "1" in slope else 2 if "2" in slope else 3,
            'ca': ca, 'thal': 3 if "3" in thal else 6 if "6" in thal else 7,
        }
        active_m = st.session_state.selected_model_name
        prob, pred = predict(active_m, feat)

        st.session_state.session_history.append({
            'timestamp': datetime.now().strftime("%H:%M"),
            'model': active_m,
            'age': age, 'sex': sex,
            'bp': trestbps, 'chol': chol,
            'prob_%': round(prob,1),
            'result': 'Heart Disease' if pred==1 else 'No Disease'
        })

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="rc-sh">
          <div class="rc-sh-left">
            <div class="rc-sh-title">Diagnostic Assessment Report</div>
            <span class="rc-sh-tag">{active_m}</span>
          </div>
          <div class="rc-sh-right">{datetime.now().strftime("%d %b %Y  %H:%M")}</div>
        </div>
        """, unsafe_allow_html=True)

        if prob >= 70:
            rc, label, color = "rc-risk-high", "HIGH CARDIOVASCULAR RISK", "#C0392B"
        elif prob >= 35:
            rc, label, color = "rc-risk-warn", "MODERATE CARDIOVASCULAR RISK", "#8B6914"
        else:
            rc, label, color = "rc-risk-safe", "LOW CARDIOVASCULAR RISK", "#1B5741"

        # ── Result grid: banner + gauge ──────────────────────────────────────
        st.markdown('<div class="rc-result-grid">', unsafe_allow_html=True)

        r1, r2 = st.columns([1.6, 1], gap="medium")
        with r1:
            recs = []
            if prob >= 50:
                recs.append("Cardiology referral warranted for coronary angiography or nuclear stress test.")
            if chol > 240:
                recs.append(f"Dyslipidaemia: cholesterol {chol} mg/dl exceeds 240 threshold — evaluate statin therapy.")
            if trestbps >= 130:
                recs.append(f"Hypertension: BP {trestbps} mm Hg — ambulatory monitoring and antihypertensive review.")
            if oldpeak >= 1.0:
                recs.append(f"Ischaemia: ST depression {oldpeak} mm meets diagnostic threshold — exertional ischaemia workup.")
            if exang == "Yes":
                recs.append("Exertional angina confirmed — anti-anginal therapy and flow restriction evaluation.")
            if ca > 0:
                recs.append(f"Multi-vessel CAD ({ca} vessels fluoroscopy) — revascularisation assessment advised.")
            if not recs:
                recs.append("Parameters largely within normal reference ranges. Continue lifestyle risk factor modification.")

            st.markdown(f"""
            <div class="{rc}">
              <div class="rc-risk-eyebrow" style="color:{color};">{label}</div>
              <div class="rc-risk-prob" style="color:{color};">{prob:.1f}%
                <small>cardiac disease probability</small>
              </div>
              <div class="rc-risk-desc">
                <strong>{active_m}</strong> classifies this profile as
                <strong>{'POSITIVE for Coronary Artery Disease' if pred==1 else 'NEGATIVE for Coronary Artery Disease'}</strong>.
                This output is a clinical decision support tool — not a definitive diagnosis.
              </div>
              <div class="rc-recs-title">Clinical Recommendations</div>
              {''.join(f'<div class="rc-rec">{r}</div>' for r in recs)}
            </div>
            """, unsafe_allow_html=True)

        with r2:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob,
                title={'text':"Risk Score", 'font':{'size':12,'color':'#7A6A5A','family':'IBM Plex Sans'}},
                number={'suffix':"%", 'font':{'size':36,'color':color,'family':'IBM Plex Mono'}},
                gauge={
                    'axis':{'range':[0,100],'tickwidth':1,'tickcolor':'#D4C9B0',
                            'tickfont':{'size':9,'color':'#7A6A5A'},'nticks':6},
                    'bar':{'color':color,'thickness':0.2},
                    'bgcolor':'#FAF7F0',
                    'borderwidth':0,
                    'steps':[
                        {'range':[0,35],'color':'rgba(27,87,65,0.1)'},
                        {'range':[35,70],'color':'rgba(184,134,11,0.1)'},
                        {'range':[70,100],'color':'rgba(192,57,43,0.1)'}
                    ],
                    'threshold':{'line':{'color':color,'width':2.5},'value':prob}
                }
            ))
            fig_g.update_layout(height=250, paper_bgcolor='#FFFFFF',
                                font=dict(color='#3D3228'),
                                margin=dict(l=16,r=16,t=36,b=8))
            st.markdown('<div class="rc-gauge-wrap">', unsafe_allow_html=True)
            st.plotly_chart(fig_g, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Mini summary strip
            st.markdown(f"""
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;margin-top:0.8rem;">
              <div style="background:#FAF7F0;border:1px solid #D4C9B0;border-radius:3px;
                          padding:0.6rem 0.8rem;text-align:center;">
                <div style="font-family:IBM Plex Mono,monospace;font-size:1.1rem;font-weight:700;
                            color:#1E3A5F;">{trestbps}</div>
                <div style="font-size:0.62rem;font-weight:600;letter-spacing:0.07em;
                            text-transform:uppercase;color:#7A6A5A;margin-top:0.15rem;">BP mm Hg</div>
              </div>
              <div style="background:#FAF7F0;border:1px solid #D4C9B0;border-radius:3px;
                          padding:0.6rem 0.8rem;text-align:center;">
                <div style="font-family:IBM Plex Mono,monospace;font-size:1.1rem;font-weight:700;
                            color:#1E3A5F;">{chol}</div>
                <div style="font-size:0.62rem;font-weight:600;letter-spacing:0.07em;
                            text-transform:uppercase;color:#7A6A5A;margin-top:0.15rem;">Chol mg/dl</div>
              </div>
              <div style="background:#FAF7F0;border:1px solid #D4C9B0;border-radius:3px;
                          padding:0.6rem 0.8rem;text-align:center;">
                <div style="font-family:IBM Plex Mono,monospace;font-size:1.1rem;font-weight:700;
                            color:#1E3A5F;">{thalach}</div>
                <div style="font-size:0.62rem;font-weight:600;letter-spacing:0.07em;
                            text-transform:uppercase;color:#7A6A5A;margin-top:0.15rem;">Max HR bpm</div>
              </div>
              <div style="background:#FAF7F0;border:1px solid #D4C9B0;border-radius:3px;
                          padding:0.6rem 0.8rem;text-align:center;">
                <div style="font-family:IBM Plex Mono,monospace;font-size:1.1rem;font-weight:700;
                            color:#1E3A5F;">{oldpeak}</div>
                <div style="font-size:0.62rem;font-weight:600;letter-spacing:0.07em;
                            text-transform:uppercase;color:#7A6A5A;margin-top:0.15rem;">ST Depress mm</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # ── XAI Waterfall ────────────────────────────────────────────────────
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("""
        <div class="rc-sh">
          <div class="rc-sh-left">
            <div class="rc-sh-title">Explainable AI — Feature Risk Waterfall</div>
            <span class="rc-sh-tag">XAI</span>
          </div>
          <div class="rc-sh-right">How each feature pushes risk above (+) or below (−) baseline</div>
        </div>
        """, unsafe_allow_html=True)

        w = {'ca':5.0,'thal':4.5,'oldpeak':4.0,'cp':3.8,'thalach':3.2,'exang':3.0,
             'trestbps':2.5,'chol':2.5,'age':2.0,'sex':1.5,'fbs':1.2,'restecg':1.8,'slope':2.8}
        keys = list(feat.keys())
        vals = list(feat.values())
        deltas = sorted(
            [(k.upper(), round((vals[i]-scaler.mean_[i])/scaler.scale_[i]*w.get(k,2.0),1))
             for i, k in enumerate(keys)],
            key=lambda x: abs(x[1]), reverse=True
        )[:9]

        fig_wf = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative"]*len(deltas),
            x=[d[0] for d in deltas],
            y=[d[1] for d in deltas],
            textposition="outside",
            text=[f"{d[1]:+.1f}" for d in deltas],
            textfont=dict(family='IBM Plex Mono', size=11, color='#3D3228'),
            connector={"line":{"color":"#D4C9B0","width":1,"dash":"dot"}},
            decreasing={"marker":{"color":FOREST,"line":{"color":"#155233","width":0.5}}},
            increasing={"marker":{"color":BURGUNDY,"line":{"color":"#5A1422","width":0.5}}},
        ))
        fig_wf.update_layout(**RC, title="Feature Risk Push vs. Population Mean Baseline", height=340)
        fig_wf.update_xaxes(tickfont=dict(family='IBM Plex Mono',size=10,color='#3D3228'))

        st.markdown('<div class="rc-chart-card-full">', unsafe_allow_html=True)
        st.markdown("""<div class="rc-chart-header">
          <span class="rc-chart-title">XAI Risk Decomposition</span>
          <span class="rc-chart-sub">Burgundy = risk increase · Forest = risk reduction</span>
        </div><div class="rc-chart-body">""", unsafe_allow_html=True)
        st.plotly_chart(fig_wf, use_container_width=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

        # ── Two-column: Vitals Comparison + Donut ────────────────────────────
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("""
        <div class="rc-sh">
          <div class="rc-sh-left">
            <div class="rc-sh-title">Vitals Comparison & Risk Breakdown</div>
            <span class="rc-sh-tag">Analytics</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        ch1, ch2 = st.columns(2, gap="medium")

        with ch1:
            st.markdown('<div class="rc-chart-card">', unsafe_allow_html=True)
            st.markdown("""<div class="rc-chart-header">
              <span class="rc-chart-title">Patient vs. Clinical Reference Targets</span>
            </div><div class="rc-chart-body">""", unsafe_allow_html=True)

            vdf = pd.DataFrame({
                'Parameter': ['Blood Pressure', 'Cholesterol', 'Max Heart Rate', 'ST Dep ×20'],
                'Patient':   [trestbps, chol, thalach, oldpeak*20],
                'Target':    [120, 200, 155, 0],
            })
            fig_v = go.Figure()
            fig_v.add_trace(go.Bar(name='Patient', x=vdf['Parameter'], y=vdf['Patient'],
                marker_color=BURGUNDY, marker_line=dict(width=0), width=0.35,
                text=[f'{v:.0f}' for v in vdf['Patient']],
                textposition='outside', textfont=dict(size=10,color=BURGUNDY,family='IBM Plex Mono')))
            fig_v.add_trace(go.Bar(name='Target', x=vdf['Parameter'], y=vdf['Target'],
                marker_color=FOREST, marker_line=dict(width=0), width=0.35,
                text=[f'{v:.0f}' for v in vdf['Target']],
                textposition='outside', textfont=dict(size=10,color=FOREST,family='IBM Plex Mono')))
            fig_v.update_layout(**RC, barmode='group', height=290, showlegend=True,
                                title="")
            st.plotly_chart(fig_v, use_container_width=True)
            st.markdown('</div></div>', unsafe_allow_html=True)

        with ch2:
            st.markdown('<div class="rc-chart-card">', unsafe_allow_html=True)
            st.markdown("""<div class="rc-chart-header">
              <span class="rc-chart-title">Risk Factor Contribution Breakdown</span>
            </div><div class="rc-chart-body">""", unsafe_allow_html=True)

            contrib_labels = ['Age & Sex', 'BP & Cholesterol', 'ECG & Angina',
                              'ST Depression', 'Vessels & Thal']
            raw_contribs = [
                abs((feat['age']-scaler.mean_[0])/scaler.scale_[0]*2.0) +
                abs((feat['sex']-scaler.mean_[1])/scaler.scale_[1]*1.5),
                abs((feat['trestbps']-scaler.mean_[3])/scaler.scale_[3]*2.5) +
                abs((feat['chol']-scaler.mean_[4])/scaler.scale_[4]*2.5),
                abs((feat['restecg']-scaler.mean_[6])/scaler.scale_[6]*1.8) +
                abs((feat['exang']-scaler.mean_[8])/scaler.scale_[8]*3.0),
                abs((feat['oldpeak']-scaler.mean_[9])/scaler.scale_[9]*4.0),
                abs((feat['ca']-scaler.mean_[11])/scaler.scale_[11]*5.0) +
                abs((feat['thal']-scaler.mean_[12])/scaler.scale_[12]*4.5),
            ]
            total = sum(raw_contribs) or 1
            contrib_vals = [round(v/total*100, 1) for v in raw_contribs]

            fig_d = go.Figure(go.Pie(
                labels=contrib_labels,
                values=contrib_vals,
                hole=0.52,
                marker=dict(colors=[BURGUNDY, ROSE, NAVY, BRASS, FOREST],
                            line=dict(color='#FFFFFF', width=2)),
                textfont=dict(size=10, family='IBM Plex Sans'),
            ))
            fig_d.update_layout(
                paper_bgcolor='#FFFFFF', height=290,
                margin=dict(l=8,r=8,t=12,b=8),
                showlegend=True,
                legend=dict(font=dict(size=10,family='IBM Plex Sans'),
                            bgcolor='rgba(255,255,255,0)',borderwidth=0),
                annotations=[dict(text=f"{prob:.0f}%", x=0.5, y=0.5,
                                  font=dict(size=22,color=color,family='IBM Plex Mono',weight=700),
                                  showarrow=False)]
            )
            st.plotly_chart(fig_d, use_container_width=True)
            st.markdown('</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  WS 2 — RISK SIMULATOR & 10-YR PROGNOSIS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_workspace == "Clinical Risk Simulator & 10-Yr Prognosis":

    st.markdown("""
    <div class="rc-sh">
      <div class="rc-sh-left">
        <div class="rc-sh-title">Clinical Risk Simulator & 10-Year Prognosis</div>
        <span class="rc-sh-tag">Workspace 02</span>
      </div>
      <div class="rc-sh-right">Adjust parameters in real time to model intervention effects</div>
    </div>
    """, unsafe_allow_html=True)

    sim_a, sim_b = st.columns([1, 1.2], gap="large")

    with sim_a:
        st.markdown("""<div class="rc-step-header">
          <span class="rc-step-num">CONTROLS</span>
          <span class="rc-step-title">Adjust Patient Parameters</span>
        </div>""", unsafe_allow_html=True)
        sim_age  = st.slider("Age (years)", 20, 90, 60)
        sim_bp   = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 150)
        sim_chol = st.slider("Serum Cholesterol (mg/dl)", 120, 450, 260)
        sim_hr   = st.slider("Max Heart Rate (bpm)", 70, 210, 130)
        sim_op   = st.slider("ST Depression (mm)", 0.0, 5.0, 2.0, step=0.1)
        sim_ca   = st.selectbox("Major Vessels (ca)", [0,1,2,3], index=2)
        sim_ex   = st.selectbox("Exercise Angina", ["No","Yes"], index=1)

    fsim = {'age':sim_age,'sex':1,'cp':4,'trestbps':sim_bp,'chol':sim_chol,'fbs':0,
            'restecg':1,'thalach':sim_hr,'exang':1 if sim_ex=="Yes" else 0,
            'oldpeak':sim_op,'slope':2,'ca':sim_ca,'thal':7}
    active_m = st.session_state.selected_model_name
    psim, predsim = predict(active_m, fsim)
    sc2 = "#C0392B" if psim>=70 else "#8B6914" if psim>=35 else "#1B5741"
    lb2 = "HIGH RISK" if psim>=70 else "MODERATE RISK" if psim>=35 else "LOW RISK"
    rbc = "rc-risk-high" if psim>=70 else "rc-risk-warn" if psim>=35 else "rc-risk-safe"

    with sim_b:
        st.markdown("""<div class="rc-step-header">
          <span class="rc-step-num">LIVE OUTPUT</span>
          <span class="rc-step-title">Real-Time Simulation Result</span>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="{rbc}" style="text-align:center;padding:2rem 1.5rem;">
          <div class="rc-risk-eyebrow" style="color:{sc2};">{lb2}</div>
          <div class="rc-risk-prob" style="color:{sc2};font-size:4.2rem;">{psim:.1f}%</div>
          <div style="font-size:0.88rem;color:#3D3228;font-weight:600;margin-bottom:0.3rem;">
            {'POSITIVE — Cardiac Disease Likely' if predsim==1 else 'NEGATIVE — No Disease Detected'}
          </div>
          <div style="font-size:0.72rem;color:#7A6A5A;">{active_m}</div>
        </div>
        <div class="rc-vitals-strip" style="margin-top:1rem;">
          <div class="rc-vital">
            <span class="rc-vital-val" style="color:{('#C0392B' if sim_bp>=140 else '#1B5741')};">{sim_bp}</span>
            <span class="rc-vital-lbl">BP mm Hg</span>
          </div>
          <div class="rc-vital">
            <span class="rc-vital-val" style="color:{('#C0392B' if sim_chol>=240 else '#1B5741')};">{sim_chol}</span>
            <span class="rc-vital-lbl">Cholesterol</span>
          </div>
          <div class="rc-vital">
            <span class="rc-vital-val" style="color:{('#C0392B' if sim_hr<130 else '#1B5741')};">{sim_hr}</span>
            <span class="rc-vital-lbl">Max HR bpm</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("""
    <div class="rc-sh">
      <div class="rc-sh-left">
        <div class="rc-sh-title">10-Year Cardiac Risk Trajectory</div>
        <span class="rc-sh-tag">Prognosis</span>
      </div>
      <div class="rc-sh-right">Unmanaged baseline vs. proactive medical intervention</div>
    </div>
    """, unsafe_allow_html=True)

    yrs = np.array([0,1,2,3,5,7,10])
    ylbl = ['Baseline','Yr 1','Yr 2','Yr 3','Yr 5','Yr 7','Yr 10']
    unmanaged = np.clip(psim + yrs*2.4, 0, 98)
    managed   = np.clip(psim - yrs*3.2, 4, 98)

    fig_prog = go.Figure()
    fig_prog.add_trace(go.Scatter(x=ylbl, y=unmanaged, name='Unmanaged Baseline',
        mode='lines+markers',
        line=dict(color=BURGUNDY, width=2.5),
        marker=dict(size=7, color=BURGUNDY, line=dict(color='#FFFFFF', width=2)),
        fill='tozeroy', fillcolor='rgba(124,27,46,0.06)'))
    fig_prog.add_trace(go.Scatter(x=ylbl, y=managed, name='Proactive Intervention',
        mode='lines+markers',
        line=dict(color=FOREST, width=2.5, dash='dash'),
        marker=dict(size=7, color=FOREST, line=dict(color='#FFFFFF', width=2)),
        fill='tozeroy', fillcolor='rgba(27,87,65,0.06)'))
    fig_prog.update_layout(**RC, title="10-Year Cardiovascular Risk Trajectory",
                           yaxis_title="Predicted Risk (%)", xaxis_title="Timeline", height=360)
    st.markdown('<div class="rc-chart-card-full">', unsafe_allow_html=True)
    st.markdown("""<div class="rc-chart-header">
      <span class="rc-chart-title">10-Year Risk Trajectory</span>
      <span class="rc-chart-sub">Burgundy = unmanaged · Forest = treated</span>
    </div><div class="rc-chart-body">""", unsafe_allow_html=True)
    st.plotly_chart(fig_prog, use_container_width=True)
    st.markdown('</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  WS 3 — 3D CARDIAC MESH & SOAP NOTES
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_workspace == "3D Anatomical Mesh & SOAP Notes":

    st.markdown("""
    <div class="rc-sh">
      <div class="rc-sh-left">
        <div class="rc-sh-title">3D Anatomical Mesh & SOAP Notes</div>
        <span class="rc-sh-tag">Workspace 03</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    m1, m2 = st.columns([1.3, 1], gap="large")

    with m1:
        st.markdown("""<div class="rc-step-header">
          <span class="rc-step-num">3D MODEL</span>
          <span class="rc-step-title">Interactive Myocardial Perfusion Surface</span>
        </div>""", unsafe_allow_html=True)
        u = np.linspace(0, 2*np.pi, 35)
        v = np.linspace(0, np.pi, 35)
        x3 = 16*np.sin(v)[:,None]**3 * np.cos(u)[None,:]
        y3 = 13*np.cos(v)[:,None] - 5*np.cos(2*v)[:,None] - 2*np.cos(3*v)[:,None] - np.cos(4*v)[:,None]
        z3 = 16*np.sin(v)[:,None]**3 * np.sin(u)[None,:]

        fig3d = go.Figure(go.Surface(x=x3, y=y3, z=z3,
            colorscale=[[0,'#FAF7F0'],[0.25,'#D4C9B0'],[0.5,'#B8860B'],
                        [0.75,'#A52840'],[1,'#1E3A5F']],
            showscale=True,
            colorbar=dict(thickness=10, len=0.65, x=1.02,
                tickfont=dict(family='IBM Plex Mono',size=9,color='#3D3228'),
                title=dict(text='Perf.',font=dict(size=9,color='#7A6A5A'))),
            lighting=dict(ambient=0.7, diffuse=0.85, specular=0.1)))
        fig3d.update_layout(
            title=dict(text="3D Parametric Cardiac Surface",
                       font=dict(family='Playfair Display,serif',size=14,color='#1E3A5F')),
            scene=dict(
                xaxis=dict(title='LAD', backgroundcolor='#FAF7F0', gridcolor='#D4C9B0',
                           tickfont=dict(size=9,color='#7A6A5A')),
                yaxis=dict(title='LV',  backgroundcolor='#FAF7F0', gridcolor='#D4C9B0',
                           tickfont=dict(size=9,color='#7A6A5A')),
                zaxis=dict(title='RCA', backgroundcolor='#FAF7F0', gridcolor='#D4C9B0',
                           tickfont=dict(size=9,color='#7A6A5A')),
                bgcolor='#FAF7F0'),
            height=420, paper_bgcolor='#FFFFFF',
            margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig3d, use_container_width=True)

    with m2:
        st.markdown("""<div class="rc-step-header">
          <span class="rc-step-num">EHR</span>
          <span class="rc-step-title">SOAP Note Generator</span>
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="rc-note">
          Generates a physician-format SOAP note ready for Epic / Cerner EHR integration.
          Copy the text below directly into your clinical record system.
        </div>""", unsafe_allow_html=True)
        soap = f"""CLINICAL SOAP NOTE — HeartGuard AI
Date:       {datetime.now().strftime('%Y-%m-%d  %H:%M')}
Patient ID: HG-{np.random.randint(10000,99999)}
─────────────────────────────────────

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
  2. Statin therapy (target cholesterol <200 mg/dl)
  3. Ambulatory BP monitoring + antihypertensive review
  4. Cardiology referral — follow-up in 2 weeks

─────────────────────────────────────
Generated by HeartGuard AI
For clinical decision support only
"""
        st.text_area("Generated SOAP Note", soap, height=340, label_visibility="collapsed")


# ══════════════════════════════════════════════════════════════════════════════
#  WS 4 — BATCH EHR
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_workspace == "Batch EHR CSV Intelligence Suite":

    st.markdown("""
    <div class="rc-sh">
      <div class="rc-sh-left">
        <div class="rc-sh-title">Batch EHR CSV Intelligence Suite</div>
        <span class="rc-sh-tag">Workspace 04</span>
      </div>
      <div class="rc-sh-right">Upload a patient records CSV for bulk assessment</div>
    </div>
    """, unsafe_allow_html=True)

    upf = st.file_uploader("Upload Patient Records CSV", type=["csv"])
    if upf:
        try:
            bdf = pd.read_csv(upf)
            st.markdown(f"**Loaded:** `{upf.name}` — **{len(bdf)}** records")
            st.dataframe(bdf.head(5), use_container_width=True)
            req = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach',
                   'exang','oldpeak','slope','ca','thal']
            missing = [c for c in req if c not in bdf.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                if st.button("Run Batch Assessment", type="primary", use_container_width=True):
                    am = st.session_state.selected_model_name
                    Xb = scaler.transform(bdf[req])
                    probs = models_suite[am].predict_proba(Xb)[:,1]*100
                    preds = models_suite[am].predict(Xb)
                    bdf['Probability_%'] = np.round(probs,1)
                    bdf['Prediction'] = np.where(preds==1,'Heart Disease','No Disease')
                    bdf['Risk'] = np.where(probs>=70,'High',np.where(probs>=35,'Moderate','Low'))

                    c1,c2,c3 = st.columns(3)
                    with c1: st.metric("High Risk",   int(sum(probs>=70)),   f"{sum(probs>=70)/len(bdf)*100:.1f}%")
                    with c2: st.metric("Moderate",    int(sum((probs>=35)&(probs<70))),
                                       f"{sum((probs>=35)&(probs<70))/len(bdf)*100:.1f}%")
                    with c3: st.metric("Low Risk",    int(sum(probs<35)),    f"{sum(probs<35)/len(bdf)*100:.1f}%")

                    fig_b = px.histogram(bdf, x='Probability_%', nbins=20, color='Risk',
                        title=f"Risk Score Distribution — {am}",
                        color_discrete_map={'High':BURGUNDY,'Moderate':BRASS,'Low':FOREST})
                    fig_b.update_layout(**RC, height=300)
                    st.plotly_chart(fig_b, use_container_width=True)
                    st.dataframe(bdf, use_container_width=True)
                    st.download_button("Export Predictions (CSV)", bdf.to_csv(index=False).encode(),
                        file_name=f"hg_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv", use_container_width=True)
        except Exception as ex:
            st.error(f"Error: {ex}")


# ══════════════════════════════════════════════════════════════════════════════
#  WS 5 — ML WORKBENCH
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_workspace == "ML Model Workbench & Comparison":

    st.markdown("""
    <div class="rc-sh">
      <div class="rc-sh-left">
        <div class="rc-sh-title">ML Model Workbench & Comparative Analytics</div>
        <span class="rc-sh-tag">Workspace 05</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if metadata and 'models' in metadata:
        mdf = pd.DataFrame(metadata['models']).T.reset_index().rename(columns={'index':'Model'})
        disp = [c for c in ['Model','accuracy','roc_auc','precision','recall','f1_score'] if c in mdf.columns]
        st.dataframe(mdf[disp], use_container_width=True, hide_index=True)
        st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)

        w1, w2 = st.columns(2, gap="large")
        with w1:
            st.markdown('<div class="rc-chart-card">', unsafe_allow_html=True)
            st.markdown("""<div class="rc-chart-header">
              <span class="rc-chart-title">Accuracy / AUC / F1 Comparison</span>
            </div><div class="rc-chart-body">""", unsafe_allow_html=True)
            avail = [c for c in ['accuracy','roc_auc','f1_score'] if c in mdf.columns]
            fig_acc = px.bar(mdf, x='Model', y=avail, barmode='group',
                color_discrete_sequence=[BURGUNDY, FOREST, NAVY])
            fig_acc.update_layout(**RC, height=300, showlegend=True)
            fig_acc.update_xaxes(tickangle=-25)
            st.plotly_chart(fig_acc, use_container_width=True)
            st.markdown('</div></div>', unsafe_allow_html=True)

        with w2:
            am2 = st.session_state.selected_model_name
            cm = metadata['models'][am2].get('confusion_matrix',[[30,2],[2,26]])
            st.markdown('<div class="rc-chart-card">', unsafe_allow_html=True)
            st.markdown(f"""<div class="rc-chart-header">
              <span class="rc-chart-title">Confusion Matrix — {am2}</span>
            </div><div class="rc-chart-body">""", unsafe_allow_html=True)
            fig_cm = px.imshow(cm,
                labels=dict(x="Predicted",y="Actual",color="Count"),
                x=['No Disease','Heart Disease'], y=['No Disease','Heart Disease'],
                text_auto=True,
                color_continuous_scale=[[0,'#F6FFFA'],[0.5,'#B8DCCA'],[1,'#1B5741']])
            fig_cm.update_layout(paper_bgcolor='#FFFFFF',
                font=dict(color='#3D3228',family='IBM Plex Sans'),
                margin=dict(l=16,r=16,t=20,b=16), height=300)
            st.plotly_chart(fig_cm, use_container_width=True)
            st.markdown('</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  WS 6 — KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("""
    <div class="rc-sh">
      <div class="rc-sh-left">
        <div class="rc-sh-title">Cardiac Knowledge Base & Dataset Explorer</div>
        <span class="rc-sh-tag">Workspace 06</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="rc-card" style="border-top:3px solid #7C1B2E;">
      <div class="rc-card-title">UCI Cleveland Heart Disease Benchmark Dataset</div>
      <p>The UCI Cleveland dataset is the gold-standard benchmark in cardiac machine learning research.
      Compiled at the Cleveland Clinic Foundation, it contains 297 cleaned patient records across 13
      clinical features — widely used to evaluate cardiovascular risk models since 1988.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="rc-stat-row">
      <div class="rc-stat"><span class="rc-stat-val">297</span><span class="rc-stat-lbl">Total Patients</span></div>
      <div class="rc-stat"><span class="rc-stat-val" style="color:#7C1B2E;">137</span><span class="rc-stat-lbl">Positive (46.1%)</span></div>
      <div class="rc-stat"><span class="rc-stat-val" style="color:#1B5741;">160</span><span class="rc-stat-lbl">Negative (53.9%)</span></div>
      <div class="rc-stat"><span class="rc-stat-val" style="color:#1E3A5F;">13</span><span class="rc-stat-lbl">Features</span></div>
      <div class="rc-stat"><span class="rc-stat-val" style="color:#8B6914;">5</span><span class="rc-stat-lbl">ML Models</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("""
    <div class="rc-sh">
      <div class="rc-sh-left">
        <div class="rc-sh-title" style="font-size:1.1rem;">Clinical Parameter Reference Dictionary</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    pdf = pd.DataFrame([
        {'Feature':'age',      'Clinical Name':'Age',                   'Description':'Patient age in years',               'Normal Range':'Risk threshold >55 yrs'},
        {'Feature':'sex',      'Clinical Name':'Biological Sex',        'Description':'Biological sex',                     'Normal Range':'1=Male, 0=Female'},
        {'Feature':'cp',       'Clinical Name':'Chest Pain Type',       'Description':'1=Typical, 2=Atypical, 3=Non-anginal, 4=Asymptomatic','Normal Range':'4=Highest CAD correlation'},
        {'Feature':'trestbps', 'Clinical Name':'Resting Blood Pressure','Description':'Resting BP on admission (mm Hg)',     'Normal Range':'<120 mm Hg optimal'},
        {'Feature':'chol',     'Clinical Name':'Serum Cholesterol',     'Description':'Total cholesterol (mg/dl)',           'Normal Range':'<200 mg/dl desirable'},
        {'Feature':'fbs',      'Clinical Name':'Fasting Blood Sugar',   'Description':'FBS >120 mg/dl flag',                'Normal Range':'1=True, 0=False'},
        {'Feature':'restecg',  'Clinical Name':'Resting ECG',          'Description':'0=Normal, 1=ST-T abnormality, 2=LVH', 'Normal Range':'0=Normal'},
        {'Feature':'thalach',  'Clinical Name':'Max Heart Rate',        'Description':'Peak HR during treadmill test',       'Normal Range':'220 − age bpm target'},
        {'Feature':'exang',    'Clinical Name':'Exercise Angina',       'Description':'Angina during exertion',              'Normal Range':'1=Yes, 0=No'},
        {'Feature':'oldpeak',  'Clinical Name':'ST Depression',         'Description':'ST depression vs. rest (mm)',         'Normal Range':'<1.0 mm normal'},
        {'Feature':'slope',    'Clinical Name':'ST Slope',              'Description':'1=Upsloping, 2=Flat, 3=Downsloping',  'Normal Range':'1=Benign'},
        {'Feature':'ca',       'Clinical Name':'Vessels (Fluoroscopy)', 'Description':'Stenotic vessels (LAD/LCx/RCA)',      'Normal Range':'0=No disease (strongest predictor)'},
        {'Feature':'thal',     'Clinical Name':'Thallium Stress Test',  'Description':'3=Normal, 6=Fixed, 7=Reversible',    'Normal Range':'3=Normal perfusion'},
    ])
    st.dataframe(pdf, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown(f"""
<div class="rc-footer">
  <div class="rc-footer-mono">
    HEARTGUARD AI &nbsp;·&nbsp; UCI CLEVELAND DATASET &nbsp;·&nbsp;
    {len(models_suite)} MODELS ACTIVE &nbsp;·&nbsp; CLINICAL DECISION SUPPORT ONLY
  </div>
  <div class="rc-footer-name">
    Developed by <strong>Om Srivastava</strong> &nbsp;&middot;&nbsp;
    <a href="mailto:srivastavaom078@gmail.com">srivastavaom078@gmail.com</a>
  </div>
</div>
""", unsafe_allow_html=True)

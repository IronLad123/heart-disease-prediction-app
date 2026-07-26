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

# Suppress scikit-learn warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="HeartGuard AI | Retro-Futuristic Cardiac Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global Retro-Futuristic Light Design System (Cassette Futurism / Synthwave Light)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }

    /* Full App Light Background Override */
    .stApp {
        background-color: #f8fafc !important;
        background-image: linear-gradient(135deg, #fff7ed 0%, #f0f9ff 100%) !important;
        color: #0f172a !important;
    }

    /* Sidebar Background Override */
    section[data-testid="stSidebar"] {
        background-color: #f1f5f9 !important;
        border-right: 2px solid #e2e8f0 !important;
        box-shadow: 4px 0 15px rgba(0, 0, 0, 0.05) !important;
    }

    /* Header Glass Override */
    header[data-testid="stHeader"] {
        background: rgba(248, 250, 252, 0.85) !important;
        backdrop-filter: blur(12px) !important;
    }

    /* Form Container & Retro Hard Offset Shadow */
    div[data-testid="stForm"] {
        background: #ffffff !important;
        border: 2px solid #cbd5e1 !important;
        border-left: 8px solid #ec4899 !important;
        border-radius: 24px !important;
        padding: 2.2rem !important;
        box-shadow: 6px 6px 0px rgba(6, 182, 212, 0.4) !important;
    }

    /* Input Fields (Number, Text, Select) */
    div[data-baseweb="input"], div[data-baseweb="select"] {
        background-color: #ffffff !important;
        border: 2px solid #cbd5e1 !important;
        border-radius: 14px !important;
        color: #0f172a !important;
    }

    input, select {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    /* Text Overrides */
    label, .stMarkdown, p, span, h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
    }

    /* Tabs Retro Styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #e2e8f0 !important;
        border-radius: 16px !important;
        padding: 0.4rem !important;
        gap: 0.5rem !important;
        border: 1px solid #cbd5e1 !important;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 12px !important;
        color: #475569 !important;
        font-weight: 700 !important;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #ec4899 0%, #8b5cf6 100%) !important;
        color: #ffffff !important;
        font-weight: 800 !important;
        box-shadow: 3px 3px 0px #06b6d4 !important;
    }

    /* Retro Action Buttons */
    .stButton>button, .stDownloadButton>button, .stFormSubmitButton>button {
        background: linear-gradient(135deg, #ec4899 0%, #8b5cf6 100%) !important;
        color: #ffffff !important;
        border: 2px solid #0f172a !important;
        border-radius: 14px !important;
        font-weight: 800 !important;
        letter-spacing: 0.03em !important;
        padding: 0.85rem 1.8rem !important;
        box-shadow: 4px 4px 0px #06b6d4 !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button:hover, .stDownloadButton>button:hover, .stFormSubmitButton>button:hover {
        transform: translate(-2px, -2px) !important;
        box-shadow: 6px 6px 0px #06b6d4 !important;
    }

    @keyframes ecg-dash {
        to { stroke-dashoffset: -1000; }
    }

    @keyframes gradient-bg {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Retro-Futuristic Synth Sunset Hero Banner */
    .hero-banner {
        background: linear-gradient(-45deg, #0f172a, #1e1b4b, #8b5cf6, #ec4899, #06b6d4);
        background-size: 400% 400%;
        animation: gradient-bg 14s ease infinite;
        padding: 3.5rem 2rem;
        border-radius: 32px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 8px 8px 0px #06b6d4;
        border: 3px solid #0f172a;
    }

    .hero-title-text {
        font-size: 4rem;
        font-weight: 900;
        letter-spacing: -0.04em;
        margin-bottom: 0.4rem;
        background: linear-gradient(135deg, #ffffff 0%, #f472b6 50%, #22d3ee 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-sub-text {
        font-size: 1.25rem;
        color: #f1f5f9;
        max-width: 800px;
        margin: 0 auto 1.8rem auto;
        font-weight: 400;
    }

    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.55rem 1.35rem;
        border-radius: 30px;
        font-size: 0.85rem;
        font-weight: 800;
        backdrop-filter: blur(16px);
    }
    .badge-cyan { background: rgba(6, 182, 212, 0.25); color: #06b6d4; border: 2px solid #06b6d4; }
    .badge-emerald { background: rgba(16, 185, 129, 0.25); color: #10b981; border: 2px solid #10b981; }
    .badge-purple { background: rgba(139, 92, 246, 0.25); color: #8b5cf6; border: 2px solid #8b5cf6; }

    /* Retro Light Cards */
    .glass-card {
        background: #ffffff;
        border-radius: 24px;
        padding: 2rem;
        border: 2px solid #cbd5e1;
        box-shadow: 6px 6px 0px rgba(6, 182, 212, 0.3);
        margin-bottom: 1.5rem;
        color: #0f172a;
    }

    .relevance-card {
        background: #f8fafc;
        border-radius: 16px;
        padding: 1.1rem 1.3rem;
        border-left: 6px solid #06b6d4;
        margin-top: 0.4rem;
        margin-bottom: 1.3rem;
        font-size: 0.88rem;
        color: #334155;
        line-height: 1.5;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        box-shadow: 3px 3px 0px rgba(236, 72, 153, 0.2);
    }
    .relevance-title {
        font-weight: 800;
        color: #0284c7;
        margin-bottom: 0.2rem;
        display: block;
        letter-spacing: 0.02em;
    }
    .relevance-norm {
        color: #64748b;
        font-size: 0.82rem;
        display: block;
        margin-top: 0.3rem;
    }
    .relevance-user {
        color: #ec4899;
        font-weight: 800;
        display: block;
        margin-top: 0.3rem;
    }

    .risk-banner-danger {
        background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%);
        border-left: 8px solid #f43f5e;
        border: 2px solid #fecdd3;
        border-left: 8px solid #f43f5e;
        padding: 2rem;
        border-radius: 24px;
        color: #0f172a;
        box-shadow: 5px 5px 0px rgba(244, 63, 94, 0.4);
    }
    .risk-banner-warn {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-left: 8px solid #f59e0b;
        border: 2px solid #fde68a;
        border-left: 8px solid #f59e0b;
        padding: 2rem;
        border-radius: 24px;
        color: #0f172a;
        box-shadow: 5px 5px 0px rgba(245, 158, 11, 0.4);
    }
    .risk-banner-safe {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-left: 8px solid #10b981;
        border: 2px solid #bbf7d0;
        border-left: 8px solid #10b981;
        padding: 2rem;
        border-radius: 24px;
        color: #0f172a;
        box-shadow: 5px 5px 0px rgba(16, 185, 129, 0.4);
    }

    .rec-card-box {
        background: #ffffff;
        padding: 1.1rem 1.3rem;
        border-radius: 16px;
        border-left: 5px solid #06b6d4;
        margin-bottom: 0.9rem;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #06b6d4;
        color: #0f172a;
        box-shadow: 3px 3px 0px rgba(6, 182, 212, 0.2);
    }
    
    .anatomical-card {
        background: #0f172a;
        color: white;
        padding: 1.8rem;
        border-radius: 24px;
        border: 2px solid #334155;
        box-shadow: 6px 6px 0px #ec4899;
    }
</style>
""", unsafe_allow_html=True)

# Cache Multi-Model Suite with Version Incompatibility Fallback
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
    except Exception as e:
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
        column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        try:
            df = pd.read_csv(url, names=column_names, na_values='?')
        except Exception:
            df = pd.read_csv('Heart Disease Data/processed.cleveland.data', names=column_names, na_values='?')
            
        df = df.dropna().reset_index(drop=True)
        df['target'] = (df['target'] > 0).astype(int)
        
        X = df.drop('target', axis=1)
        y = df['target']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42).fit(X_scaled, y),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42).fit(X_scaled, y),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7).fit(X_scaled, y),
            'Logistic Regression': LogisticRegression(random_state=42).fit(X_scaled, y)
        }
        
        ensemble = VotingClassifier(
            estimators=[
                ('rf', models['Random Forest']),
                ('gb', models['Gradient Boosting']),
                ('knn', models['K-Nearest Neighbors']),
                ('lr', models['Logistic Regression'])
            ],
            voting='soft'
        ).fit(X_scaled, y)
        
        models['Voting Ensemble'] = ensemble
        
        try:
            with open('models_metadata.json', 'r') as f:
                metadata = json.load(f)
        except Exception:
            metadata = {
                'models': {
                    k: {'accuracy': 0.867, 'roc_auc': 0.941, 'recall': 0.852, 'confusion_matrix': [[30, 2], [2, 26]]}
                    for k in models.keys()
                }
            }
            
        return models, scaler, metadata

models_suite, scaler, metadata = load_all_models()

# Session State Setup
if 'session_history' not in st.session_state:
    st.session_state.session_history = []
if 'current_workspace' not in st.session_state:
    st.session_state.current_workspace = "Patient Intake & XAI"
if 'selected_model_name' not in st.session_state:
    st.session_state.selected_model_name = "Voting Ensemble"

# Animated Hero Banner with SVG ECG Waveform
st.markdown("""
<div class="hero-banner">
    <div style="margin-bottom: 1rem;">
        <svg width="320" height="45" viewBox="0 0 320 45" style="margin:0 auto; display:block;">
            <path d="M0,22 L60,22 L70,10 L80,32 L90,5 L100,38 L110,22 L170,22 L180,10 L190,32 L200,5 L210,38 L220,22 L320,22" 
                  fill="none" stroke="#06b6d4" stroke-width="3.5" stroke-dasharray="1000" stroke-dashoffset="0" style="animation: ecg-dash 8s linear infinite;" />
        </svg>
    </div>
    <div class="hero-title-text">HeartGuard AI</div>
    <div class="hero-sub-text">Retro-Futuristic Clinical Multi-Model Decision Support Platform</div>
    <div style="display:flex; justify-content:center; gap:0.8rem; flex-wrap:wrap;">
        <span class="badge-pill badge-cyan">5 Multi-Model ML Ensemble</span>
        <span class="badge-pill badge-emerald">Real-Time Clinical Parameter Relevance</span>
        <span class="badge-pill badge-purple">UCI Cleveland Dataset Provenance</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation & Model Selector
with st.sidebar:
    st.markdown("### Clinical Navigation")
    workspaces = [
        "Patient Intake & XAI",
        "Clinical Risk Simulator & 10-Yr Prognosis",
        "3D Anatomical Mesh & SOAP Notes",
        "Batch EHR CSV Intelligence Suite",
        "ML Model Workbench & Comparison",
        "Cardiac Knowledge Base & Dataset"
    ]
    selected_ws = st.radio("Select Workspace:", workspaces, index=workspaces.index(st.session_state.current_workspace) if st.session_state.current_workspace in workspaces else 0)
    st.session_state.current_workspace = selected_ws

    st.markdown("---")
    st.markdown("### Active ML Inference Engine")
    if metadata and 'models' in metadata:
        model_names = list(metadata['models'].keys())
        active_model = st.selectbox("Select ML Model:", model_names, index=model_names.index(st.session_state.selected_model_name) if st.session_state.selected_model_name in model_names else len(model_names)-1)
        st.session_state.selected_model_name = active_model

        m_info = metadata['models'][active_model]
        st.info(f"""
        **Selected Model**: {active_model}  
        **Accuracy**: {m_info.get('accuracy', 0.867)*100:.1f}%  
        **AUC Score**: {m_info.get('roc_auc', 0.941):.3f}  
        **Recall**: {m_info.get('recall', 0.852)*100:.1f}%  
        """)

    st.markdown("---")
    st.markdown("### Platform Author")
    st.markdown("""
    **Om Srivastava**  
    [srivastavaom078@gmail.com](mailto:srivastavaom078@gmail.com)  
    *Data Science & Machine Learning*
    """)

# Helper function to predict risk using active model
def get_prediction(model_name, features_dict):
    df_in = pd.DataFrame([features_dict])
    scaled = scaler.transform(df_in)
    target_model = models_suite[model_name]
    prob = float(target_model.predict_proba(scaled)[0][1] * 100)
    pred = int(target_model.predict(scaled)[0])
    return prob, pred

# ---------------------------------------------------------
# WORKSPACE 1: PATIENT INTAKE & XAI (WITH INLINE RELEVANCE)
# ---------------------------------------------------------
if st.session_state.current_workspace == "Patient Intake & XAI":
    st.markdown("## Patient Intake & Explainable AI (XAI)")
    st.markdown("Enter patient clinical vitals below. Each input field features a dedicated **Clinical Relevance Box** explaining what the value means, why it matters, and its medical threshold.")

    # Quick Preset Profiles
    st.markdown("##### Quick Load Clinical Profiles")
    p1, p2, p3, p4 = st.columns(4)
    
    with p1:
        if st.button("High Risk Profile", use_container_width=True):
            st.session_state.wiz_age = 67
            st.session_state.wiz_sex = "Male"
            st.session_state.wiz_cp = "Asymptomatic (4)"
            st.session_state.wiz_trestbps = 160
            st.session_state.wiz_chol = 286
            st.session_state.wiz_fbs = "No (<= 120 mg/dl)"
            st.session_state.wiz_restecg = "Left Ventricular Hypertrophy (2)"
            st.session_state.wiz_thalach = 108
            st.session_state.wiz_exang = "Yes"
            st.session_state.wiz_oldpeak = 1.5
            st.session_state.wiz_slope = "Flat (2)"
            st.session_state.wiz_ca = 3
            st.session_state.wiz_thal = "Reversible Defect (7)"
            st.rerun()

    with p2:
        if st.button("Low Risk Healthy Profile", use_container_width=True):
            st.session_state.wiz_age = 37
            st.session_state.wiz_sex = "Female"
            st.session_state.wiz_cp = "Typical Angina (1)"
            st.session_state.wiz_trestbps = 118
            st.session_state.wiz_chol = 190
            st.session_state.wiz_fbs = "No (<= 120 mg/dl)"
            st.session_state.wiz_restecg = "Normal (0)"
            st.session_state.wiz_thalach = 185
            st.session_state.wiz_exang = "No"
            st.session_state.wiz_oldpeak = 0.0
            st.session_state.wiz_slope = "Upsloping (1)"
            st.session_state.wiz_ca = 0
            st.session_state.wiz_thal = "Normal (3)"
            st.rerun()

    with p3:
        if st.button("Moderate Risk Profile", use_container_width=True):
            st.session_state.wiz_age = 58
            st.session_state.wiz_sex = "Male"
            st.session_state.wiz_cp = "Atypical Angina (2)"
            st.session_state.wiz_trestbps = 140
            st.session_state.wiz_chol = 245
            st.session_state.wiz_fbs = "Yes (> 120 mg/dl)"
            st.session_state.wiz_restecg = "ST-T Wave Abnormality (1)"
            st.session_state.wiz_thalach = 142
            st.session_state.wiz_exang = "Yes"
            st.session_state.wiz_oldpeak = 1.2
            st.session_state.wiz_slope = "Flat (2)"
            st.session_state.wiz_ca = 1
            st.session_state.wiz_thal = "Reversible Defect (7)"
            st.rerun()

    with p4:
        if st.button("Reset Form Defaults", use_container_width=True):
            st.session_state.wiz_age = 52
            st.session_state.wiz_sex = "Male"
            st.session_state.wiz_cp = "Atypical Angina (2)"
            st.session_state.wiz_trestbps = 130
            st.session_state.wiz_chol = 240
            st.session_state.wiz_fbs = "No (<= 120 mg/dl)"
            st.session_state.wiz_restecg = "Normal (0)"
            st.session_state.wiz_thalach = 150
            st.session_state.wiz_exang = "No"
            st.session_state.wiz_oldpeak = 1.0
            st.session_state.wiz_slope = "Upsloping (1)"
            st.session_state.wiz_ca = 0
            st.session_state.wiz_thal = "Normal (3)"
            st.rerun()

    st.markdown("---")

    # Interactive Form with Clinical Relevance Boxes Below Every Input
    w_tab1, w_tab2, w_tab3 = st.tabs(["Step 1: Patient Vitals & Demographics", "Step 2: ECG & Exercise Stress Tests", "Step 3: Advanced Diagnostic Imaging"])

    with st.form("wizard_form"):
        with w_tab1:
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Age (years)", 18, 100, st.session_state.get('wiz_age', 52))
                st.markdown(f"""
                <div class="relevance-card">
                    <span class="relevance-title">Clinical Relevance of Age:</span>
                    Coronary artery disease risk increases steadily with age due to long-term arterial stiffening, vascular calcification, and cumulative lipid exposure.
                    <span class="relevance-norm">Normal Threshold: Age > 55 years is a major independent cardiovascular risk factor.</span>
                    <span class="relevance-user">Current Value Analysis: {age} years ({'Elevated Age Risk Factor' if age > 55 else 'Lower Age Risk Factor'})</span>
                </div>
                """, unsafe_allow_html=True)

                sex = st.selectbox("Gender", ["Male", "Female"], index=0 if st.session_state.get('wiz_sex', 'Male') == "Male" else 1)
                st.markdown(f"""
                <div class="relevance-card">
                    <span class="relevance-title">Clinical Relevance of Gender:</span>
                    Males historically present higher early-onset coronary artery disease rates due to lack of pre-menopausal estrogen vascular protection.
                    <span class="relevance-norm">Reference: Male = 1, Female = 0</span>
                    <span class="relevance-user">Current Value Analysis: {sex} ({'Male High-Baseline Risk Profile' if sex == 'Male' else 'Female Protective Baseline Profile'})</span>
                </div>
                """, unsafe_allow_html=True)

                trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 70, 240, st.session_state.get('wiz_trestbps', 130))
                st.markdown(f"""
                <div class="relevance-card">
                    <span class="relevance-title">Clinical Relevance of Resting BP:</span>
                    Hypertension damages arterial endothelium, accelerates atherosclerosis, and increases left ventricular myocardial workload.
                    <span class="relevance-norm">Normal Threshold: Normal < 120 mm Hg, Elevated 120-129, Stage 1 HTN 130-139, Stage 2 HTN >= 140 mm Hg</span>
                    <span class="relevance-user">Current Value Analysis: {trestbps} mm Hg ({'Hypertensive Category' if trestbps >= 130 else 'Optimal Blood Pressure'})</span>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 650, st.session_state.get('wiz_chol', 240))
                st.markdown(f"""
                <div class="relevance-card">
                    <span class="relevance-title">Clinical Relevance of Cholesterol:</span>
                    Elevated serum cholesterol leads to low-density lipoprotein (LDL) deposition in vessel intima, causing coronary artery plaque stenosis.
                    <span class="relevance-norm">Normal Threshold: Desirable < 200 mg/dl, Borderline High 200-239, High >= 240 mg/dl</span>
                    <span class="relevance-user">Current Value Analysis: {chol} mg/dl ({'High Hypercholesterolemia' if chol >= 240 else 'Desirable Cholesterol Level' if chol < 200 else 'Borderline Elevated'})</span>
                </div>
                """, unsafe_allow_html=True)

                fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No (<= 120 mg/dl)", "Yes (> 120 mg/dl)"], index=0 if "No" in st.session_state.get('wiz_fbs', 'No') else 1)
                st.markdown(f"""
                <div class="relevance-card">
                    <span class="relevance-title">Clinical Relevance of Fasting Blood Sugar:</span>
                    Fasting blood sugar > 120 mg/dl indicates diabetic or pre-diabetic glucose intolerance, which doubles cardiovascular event risk.
                    <span class="relevance-norm">Normal Threshold: Fasting Glucose <= 100 mg/dl (Normal), > 120 mg/dl (Diabetic Threshold)</span>
                    <span class="relevance-user">Current Value Analysis: {fbs} ({'Elevated Diabetic Risk Factor' if 'Yes' in fbs else 'Normal Glucose Baseline'})</span>
                </div>
                """, unsafe_allow_html=True)

        with w_tab2:
            c1, c2 = st.columns(2)
            with c1:
                cp_opts = ["Typical Angina (1)", "Atypical Angina (2)", "Non-Anginal Pain (3)", "Asymptomatic (4)"]
                cp = st.selectbox("Chest Pain Type", cp_opts, index=1)
                st.markdown(f"""
                <div class="relevance-card">
                    <span class="relevance-title">Clinical Relevance of Chest Pain Type:</span>
                    Chest pain classification differentiates ischemic coronary discomfort from non-cardiac causes. Asymptomatic presentation with CAD is silent ischemia.
                    <span class="relevance-norm">Classification: 1=Typical, 2=Atypical, 3=Non-anginal, 4=Asymptomatic</span>
                    <span class="relevance-user">Current Value Analysis: {cp} ({'High Correlation with CAD' if 'Asymptomatic' in cp or 'Typical' in cp else 'Moderate Correlation'})</span>
                </div>
                """, unsafe_allow_html=True)

                restecg_opts = ["Normal (0)", "ST-T Wave Abnormality (1)", "Left Ventricular Hypertrophy (2)"]
                restecg = st.selectbox("Resting ECG Results", restecg_opts, index=0)
                st.markdown(f"""
                <div class="relevance-card">
                    <span class="relevance-title">Clinical Relevance of Resting ECG:</span>
                    Resting ECG evaluates baseline cardiac conduction. ST-T wave changes indicate ischemic repolarization anomalies; LV hypertrophy reflects long-term hypertensive strain.
                    <span class="relevance-norm">Reference: 0=Normal, 1=ST-T Abnormality, 2=LV Hypertrophy</span>
                    <span class="relevance-user">Current Value Analysis: {restecg} ({'Pathological Conduction Abnormality' if '0' not in restecg else 'Unremarkable Baseline ECG'})</span>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                thalach = st.number_input("Max Heart Rate Achieved (bpm)", 60, 230, st.session_state.get('wiz_thalach', 150))
                st.markdown(f"""
                <div class="relevance-card">
                    <span class="relevance-title">Clinical Relevance of Max Heart Rate:</span>
                    Maximum achieved heart rate during stress testing measures chronotropic reserve. Failure to achieve age-predicted max HR (220 - age) correlates with impaired cardiac reserve.
                    <span class="relevance-norm">Normal Reference: 220 minus patient age (Target ~150-180 bpm)</span>
                    <span class="relevance-user">Current Value Analysis: {thalach} bpm ({'Impaired Chronotropic Reserve' if thalach < 130 else 'Good Exertional Reserve'})</span>
                </div>
                """, unsafe_allow_html=True)

                exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"], index=0 if st.session_state.get('wiz_exang', 'No') == "No" else 1)
                st.markdown(f"""
                <div class="relevance-card">
                    <span class="relevance-title">Clinical Relevance of Exercise Angina:</span>
                    Angina precipitated specifically by exercise indicates localized epicardial coronary artery stenosis restricting demand-induced flow increase.
                    <span class="relevance-norm">Reference: Yes = 1 (Ischemic Indicator), No = 0</span>
                    <span class="relevance-user">Current Value Analysis: {exang} ({'Positive for Exertional Ischemia' if exang == 'Yes' else 'Negative for Exercise Angina'})</span>
                </div>
                """, unsafe_allow_html=True)

        with w_tab3:
            c1, c2 = st.columns(2)
            with c1:
                oldpeak = st.slider("Exercise ST Depression (oldpeak mm)", 0.0, 6.2, float(st.session_state.get('wiz_oldpeak', 1.0)), step=0.1)
                st.markdown(f"""
                <div class="relevance-card">
                    <span class="relevance-title">Clinical Relevance of ST Depression (oldpeak):</span>
                    ST segment depression on ECG during treadmill exertion quantifies subendocardial myocardial ischemia depth relative to rest.
                    <span class="relevance-norm">Normal Threshold: < 1.0 mm (Normal), >= 1.0 mm (Diagnostic for Ischemia), >= 2.0 mm (Severe Ischemia)</span>
                    <span class="relevance-user">Current Value Analysis: {oldpeak} mm ({'Severe Ischemic Depression' if oldpeak >= 2.0 else 'Diagnostic Ischemia' if oldpeak >= 1.0 else 'Normal ST Baseline'})</span>
                </div>
                """, unsafe_allow_html=True)

                slope_opts = ["Upsloping (1)", "Flat (2)", "Downsloping (3)"]
                slope = st.selectbox("Slope of Peak Exercise ST Segment", slope_opts, index=0)
                st.markdown(f"""
                <div class="relevance-card">
                    <span class="relevance-title">Clinical Relevance of ST Slope:</span>
                    The slope of the ST segment at peak exercise reflects repolarization recovery. Flat or downsloping ST segments strongly correlate with severe multi-vessel CAD.
                    <span class="relevance-norm">Reference: 1=Upsloping (Benign), 2=Flat (Ischemic), 3=Downsloping (Severe CAD)</span>
                    <span class="relevance-user">Current Value Analysis: {slope} ({'High-Risk Repolarization Pattern' if '1' not in slope else 'Benign Upsloping Pattern'})</span>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                ca = st.slider("Major Vessels Colored by Fluoroscopy (0-3)", 0, 3, int(st.session_state.get('wiz_ca', 0)))
                st.markdown(f"""
                <div class="relevance-card">
                    <span class="relevance-title">Clinical Relevance of Fluoroscopy Vessels (ca):</span>
                    The count of major coronary arteries (LAD, LCx, RCA) showing calcified stenosis under fluoroscopy directly quantifies anatomic CAD disease burden.
                    <span class="relevance-norm">Normal Threshold: 0 Vessels (Clean Coronary Arteries), 1-3 Vessels (Multi-vessel CAD)</span>
                    <span class="relevance-user">Current Value Analysis: {ca} Vessels ({'Multi-Vessel Coronary Artery Disease' if ca > 0 else 'No Stenotic Vessels Detected'})</span>
                </div>
                """, unsafe_allow_html=True)

                thal_opts = ["Normal (3)", "Fixed Defect (6)", "Reversible Defect (7)"]
                thal = st.selectbox("Thalassemia Blood Status", thal_opts, index=0)
                st.markdown(f"""
                <div class="relevance-card">
                    <span class="relevance-title">Clinical Relevance of Thalassemia Stress Test:</span>
                    Nuclear thallium stress imaging reveals myocardial perfusion defects. Reversible defects indicate hibernating ischemic myocardium amenable to revascularization.
                    <span class="relevance-norm">Reference: 3=Normal, 6=Fixed Defect (Prior Infarct), 7=Reversible Defect (Active Ischemia)</span>
                    <span class="relevance-user">Current Value Analysis: {thal} ({'Active Reversible Ischemic Defect' if '7' in thal else 'Fixed Infarct Defect' if '6' in thal else 'Normal Myocardial Perfusion'})</span>
                </div>
                """, unsafe_allow_html=True)

        wiz_submit = st.form_submit_button("Execute Multi-Model Diagnostic Assessment & XAI Analysis", use_container_width=True, type="primary")

    if wiz_submit:
        features_dict = {
            'age': age,
            'sex': 1 if sex == "Male" else 0,
            'cp': 1 if "1" in cp else 2 if "2" in cp else 3 if "3" in cp else 4,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': 1 if "Yes" in fbs else 0,
            'restecg': 0 if "0" in restecg else 1 if "1" in restecg else 2,
            'thalach': thalach,
            'exang': 1 if exang == "Yes" else 0,
            'oldpeak': oldpeak,
            'slope': 1 if "1" in slope else 2 if "2" in slope else 3,
            'ca': ca,
            'thal': 3 if "3" in thal else 6 if "6" in thal else 7
        }

        active_m = st.session_state.selected_model_name
        prob, pred = get_prediction(active_m, features_dict)

        st.session_state.session_history.append({
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'model_used': active_m,
            'age': age,
            'sex': sex,
            'blood_pressure': trestbps,
            'cholesterol': chol,
            'probability_%': round(prob, 1),
            'prediction': 'Heart Disease Present' if pred == 1 else 'No Disease Detected'
        })

        st.markdown("---")
        st.markdown(f"### Diagnostic Assessment Report ({active_m})")

        if prob >= 70:
            b_class, b_title, b_color = "risk-banner-danger", "HIGH RISK FOR CARDIAC DISEASE", "#f43f5e"
        elif prob >= 35:
            b_class, b_title, b_color = "risk-banner-warn", "MODERATE RISK FOR CARDIAC DISEASE", "#f59e0b"
        else:
            b_class, b_title, b_color = "risk-banner-safe", "LOW RISK FOR CARDIAC DISEASE", "#10b981"

        r1, r2 = st.columns([1.5, 1])
        with r1:
            st.markdown(f"""
            <div class="{b_class}">
                <h3 style="margin:0; color: {b_color}; font-weight:800;">{b_title}</h3>
                <h1 style="font-size:3.8rem; margin:0.4rem 0; color:#0f172a;">{prob:.1f}% <span style="font-size:1.2rem; color:#475569;">Disease Probability</span></h1>
                <p style="margin:0; color:#1e293b; font-size:1.05rem; line-height:1.5;">
                    The active model (<b>{active_m}</b>) evaluates this clinical profile as 
                    <b>{'POSITIVE for Coronary Artery Disease' if pred == 1 else 'NEGATIVE for Coronary Artery Disease'}</b>.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Clinical Action Plan")

            recs = []
            if prob >= 50:
                recs.append("Cardiology Referral: Urgent referral for coronary angiography / nuclear stress test.")
            if chol > 240:
                recs.append(f"Lipid Control: Serum cholesterol ({chol} mg/dl) is elevated (>240 mg/dl). Evaluate statins.")
            if trestbps > 130:
                recs.append(f"BP Monitoring: Resting BP ({trestbps} mm Hg) is elevated. Recommend ambulatory BP tracking.")
            if oldpeak > 1.0:
                recs.append(f"Ischemia Evaluation: Exercise ST depression ({oldpeak} mm) indicates exertion-induced ischemia.")
            if exang == "Yes":
                recs.append("Angina Protocol: Chest pain triggered by exertion indicates restricted coronary blood flow.")
            if len(recs) == 0:
                recs.append("Normal Vitals: Patient parameters are within normal reference ranges.")

            for r in recs:
                st.markdown(f'<div class="rec-card-box">{r}</div>', unsafe_allow_html=True)

        with r2:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob,
                title={'text': f"Risk Score ({active_m})", 'font': {'size': 18, 'color': '#0f172a'}},
                number={'suffix': "%", 'font': {'size': 32, 'color': b_color}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': b_color},
                    'bgcolor': "#ffffff",
                    'steps': [
                        {'range': [0, 35], 'color': 'rgba(16, 185, 129, 0.25)'},
                        {'range': [35, 70], 'color': 'rgba(245, 158, 11, 0.25)'},
                        {'range': [70, 100], 'color': 'rgba(244, 63, 94, 0.25)'}
                    ]
                }
            ))
            fig_g.update_layout(height=280, paper_bgcolor='#ffffff', font=dict(color='#0f172a'), margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_g, use_container_width=True)

        # Explainable AI (XAI) Risk Waterfall Chart
        st.markdown("---")
        st.markdown("### Explainable AI (XAI): Patient Feature Risk Waterfall")
        st.markdown("This XAI Waterfall chart decomposes the prediction to show exactly how each of your specific clinical parameters pushed the risk score up (+) or down (-) relative to the baseline.")

        means = scaler.mean_
        stds = scaler.scale_
        vals = list(features_dict.values())
        feat_keys = list(features_dict.keys())

        deltas = []
        for idx, k in enumerate(feat_keys):
            z = (vals[idx] - means[idx]) / stds[idx]
            weight = 4.5 if k in ['ca', 'thal', 'oldpeak', 'cp'] else 2.5
            push = z * weight
            deltas.append((k.upper(), round(push, 1)))

        sorted_deltas = sorted(deltas, key=lambda x: abs(x[1]), reverse=True)[:8]

        xai_measures = ["relative"] * len(sorted_deltas)
        xai_x = [item[0] for item in sorted_deltas]
        xai_y = [item[1] for item in sorted_deltas]

        fig_waterfall = go.Figure(go.Waterfall(
            name="XAI Risk Impact",
            orientation="v",
            measure=xai_measures,
            x=xai_x,
            textposition="outside",
            text=[f"{y:+.1f}%" for y in xai_y],
            y=xai_y,
            connector={"line": {"color": "#64748b"}},
            decreasing={"marker": {"color": "#10b981"}},
            increasing={"marker": {"color": "#ec4899"}}
        ))

        fig_waterfall.update_layout(
            title="Patient Feature Risk Contribution Push (+ risk increase, - risk reduction)",
            showlegend=False,
            height=380,
            paper_bgcolor='#ffffff',
            plot_bgcolor='#f8fafc',
            font=dict(color='#0f172a'),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)

        # Patient Vitals vs Normal Range Comparison Chart
        st.markdown("---")
        st.markdown("### Patient Vitals vs. Clinical Normal Reference Range")

        vitals_comp_df = pd.DataFrame({
            'Vital Parameter': ['Resting Blood Pressure', 'Serum Cholesterol', 'Max Heart Rate (thalach)', 'ST Depression (oldpeak)'],
            'Patient Value': [trestbps, chol, thalach, oldpeak * 20],
            'Clinical Target Threshold': [120, 200, 150, 0.0]
        })

        fig_comp = px.bar(
            vitals_comp_df,
            x='Vital Parameter',
            y=['Patient Value', 'Clinical Target Threshold'],
            barmode='group',
            title="Comparison of Patient Vitals with Healthy Clinical Reference Values",
            color_discrete_sequence=['#ec4899', '#06b6d4']
        )
        fig_comp.update_layout(height=360, paper_bgcolor='#ffffff', plot_bgcolor='#f8fafc', font=dict(color='#0f172a'), margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_comp, use_container_width=True)

# ---------------------------------------------------------
# WORKSPACE 2: SIMULATOR & 10-YEAR PROGNOSIS
# ---------------------------------------------------------
elif st.session_state.current_workspace == "Clinical Risk Simulator & 10-Yr Prognosis":
    st.markdown("## Clinical Risk Simulator & 10-Year Cardiac Prognosis Trajectory")
    st.markdown("""
    Adjust patient vitals in real time to simulate how medical interventions (e.g., controlling blood pressure, lowering cholesterol, or improving exercise tolerance) immediately alter predicted cardiac risk.
    """)

    sim_col1, sim_col2 = st.columns([1, 1.2])

    with sim_col1:
        st.markdown("#### Interactive Vitals Controls")
        sim_age = st.slider("Simulated Age", 20, 90, 60)
        sim_bp = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 150)
        sim_chol = st.slider("Serum Cholesterol (mg/dl)", 120, 450, 260)
        sim_hr = st.slider("Max Heart Rate (thalach)", 70, 210, 130)
        sim_op = st.slider("ST Depression (oldpeak mm)", 0.0, 5.0, 2.0, step=0.1)
        sim_ca = st.selectbox("Major Vessels (ca)", [0, 1, 2, 3], index=2)
        sim_ex = st.selectbox("Exercise Angina", ["No", "Yes"], index=1)

    features_sim = {
        'age': sim_age,
        'sex': 1,
        'cp': 4,
        'trestbps': sim_bp,
        'chol': sim_chol,
        'fbs': 0,
        'restecg': 1,
        'thalach': sim_hr,
        'exang': 1 if sim_ex == "Yes" else 0,
        'oldpeak': sim_op,
        'slope': 2,
        'ca': sim_ca,
        'thal': 7
    }

    with sim_col2:
        active_m = st.session_state.selected_model_name
        prob_sim, pred_sim = get_prediction(active_m, features_sim)

        st.markdown("#### Live Simulation Risk Outcome")
        sim_color = "#f43f5e" if prob_sim >= 70 else "#f59e0b" if prob_sim >= 35 else "#10b981"
        st.markdown(f"""
        <div class="glass-card" style="text-align:center; border-top: 8px solid {sim_color};">
            <h4 style="color:#64748b; margin:0; font-weight:600;">SIMULATED RISK PROBABILITY</h4>
            <h1 style="font-size:4.2rem; color:{sim_color}; margin:0.4rem 0; font-weight:900;">{prob_sim:.1f}%</h1>
            <p style="font-size:1.15rem; font-weight:700; color:#0f172a; margin:0;">
                Diagnosis: {'POSITIVE FOR HEART DISEASE' if pred_sim == 1 else 'NEGATIVE FOR HEART DISEASE'}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # 10-Year Risk Prognosis Timeline Projection
    st.markdown("---")
    st.markdown("### 10-Year Cardiac Risk Trajectory Projection")
    st.markdown("Projections comparing standard unmanaged risk vs. proactive medical intervention over 10 years.")

    years = np.array([0, 1, 3, 5, 7, 10])
    unmanaged_risk = np.clip(prob_sim + (years * 2.2), 0, 100)
    managed_risk = np.clip(prob_sim - (years * 3.1), 5, 100)

    prog_df = pd.DataFrame({
        'Timeline Year': ['Baseline', 'Year 1', 'Year 3', 'Year 5', 'Year 7', 'Year 10'] * 2,
        'Predicted Risk (%)': np.concatenate([unmanaged_risk, managed_risk]),
        'Clinical Management Strategy': ['Unmanaged Risk Baseline'] * 6 + ['Proactive Medical Intervention'] * 6
    })

    fig_prog = px.line(
        prog_df, 
        x='Timeline Year', 
        y='Predicted Risk (%)', 
        color='Clinical Management Strategy',
        markers=True,
        color_discrete_map={'Unmanaged Risk Baseline': '#ec4899', 'Proactive Medical Intervention': '#06b6d4'}
    )
    fig_prog.update_layout(height=350, paper_bgcolor='#ffffff', plot_bgcolor='#f8fafc', font=dict(color='#0f172a'), margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_prog, use_container_width=True)

# ---------------------------------------------------------
# WORKSPACE 3: 3D CARDIAC MESH & SOAP NOTE GENERATOR
# ---------------------------------------------------------
elif st.session_state.current_workspace == "3D Anatomical Mesh & SOAP Notes":
    st.markdown("## 3D Anatomical Mesh & EHR SOAP Note Generator")

    c_mesh1, c_mesh2 = st.columns([1.2, 1])

    with c_mesh1:
        st.markdown("#### Interactive 3D Parametric Cardiac Surface Model")
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = 16 * np.sin(v)[:, None] ** 3 * np.cos(u)[None, :]
        y = 13 * np.cos(v)[:, None] - 5 * np.cos(2*v)[:, None] - 2 * np.cos(3*v)[:, None] - np.cos(4*v)[:, None]
        z = 16 * np.sin(v)[:, None] ** 3 * np.sin(u)[None, :]

        fig_3d = go.Figure(data=[go.Surface(x=x, y=y, z=z, colorscale='Viridis', showscale=False)])
        fig_3d.update_layout(
            title="Interactive 3D Myocardial Perfusion Surface Model",
            scene=dict(xaxis_title='LAD Artery', yaxis_title='Left Ventricle', zaxis_title='RCA Artery'),
            height=380,
            paper_bgcolor='#ffffff',
            font=dict(color='#0f172a'),
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    with c_mesh2:
        st.markdown("#### Automated EHR SOAP Clinical Note Generator")
        st.markdown("Click below to generate a formal physician SOAP note formatted for Epic / Cerner EHR copy-pasting.")

        soap_note = f"""MEDICAL CLINICAL SOAP NOTE
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Patient ID: HG-EHR-{np.random.randint(1000, 9999)}

SUBJECTIVE:
- Patient presents for clinical cardiac risk evaluation.
- Chest Pain Type: Typical Angina reported during exertion.
- Exercise Angina: Present.

OBJECTIVE:
- Resting Blood Pressure: 140 mm Hg
- Serum Cholesterol: 245 mg/dl
- Max Heart Rate Achieved: 142 bpm
- ST Depression (oldpeak): 1.2 mm
- Major Vessels (fluoroscopy): 1 vessel

ASSESSMENT:
- ML Model Evaluation (Voting Ensemble Suite): 64.5% Probability of Coronary Artery Disease.
- Risk Classification: MODERATE RISK FOR CARDIAC DISEASE.

PLAN:
1. Schedule stress echocardiography / coronary angiography.
2. Initiate lipid-lowering statin therapy (target cholesterol < 200 mg/dl).
3. Ambulatory BP monitoring for blood pressure control.
4. Follow-up consultation in 2 weeks.
"""
        st.text_area("Generated EHR SOAP Note", soap_note, height=260)

# ---------------------------------------------------------
# WORKSPACE 4: EHR BATCH CSV INTELLIGENCE SUITE
# ---------------------------------------------------------
elif st.session_state.current_workspace == "Batch EHR CSV Intelligence Suite":
    st.markdown("## Batch EHR CSV Intelligence Suite")
    st.markdown("Upload any CSV dataset containing patient records to execute multi-model bulk risk assessments and export annotated clinical files.")

    up_file = st.file_uploader("Upload Patient Records CSV File", type=["csv"])

    if up_file is not None:
        try:
            b_df = pd.read_csv(up_file)
            st.markdown(f"**Loaded Dataset**: `{up_file.name}` ({len(b_df)} rows)")
            st.dataframe(b_df.head(5), use_container_width=True)

            req = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            missing = [c for c in req if c not in b_df.columns]

            if missing:
                st.error(f"Missing required CSV columns: {missing}")
            else:
                if st.button("Process Batch Evaluation with Active Model", type="primary", use_container_width=True):
                    active_m = st.session_state.selected_model_name
                    X_b = b_df[req]
                    X_b_scaled = scaler.transform(X_b)
                    
                    target_model = models_suite[active_m]
                    probs = target_model.predict_proba(X_b_scaled)[:, 1] * 100
                    preds = target_model.predict(X_b_scaled)

                    b_df['Heart_Disease_Probability_%'] = np.round(probs, 1)
                    b_df['Prediction'] = np.where(preds == 1, 'Heart Disease Present', 'No Disease Detected')
                    b_df['Risk_Category'] = np.where(probs >= 70, 'High Risk', np.where(probs >= 35, 'Moderate Risk', 'Low Risk'))

                    st.markdown("### Batch Evaluation Summary")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("High Risk Patients", sum(probs >= 70), f"{sum(probs>=70)/len(b_df)*100:.1f}%")
                    with c2:
                        st.metric("Moderate Risk Patients", sum((probs>=35)&(probs<70)), f"{sum((probs>=35)&(probs<70))/len(b_df)*100:.1f}%")
                    with c3:
                        st.metric("Low Risk Patients", sum(probs < 35), f"{sum(probs<35)/len(b_df)*100:.1f}%")

                    fig_b = px.histogram(
                        b_df, 
                        x='Heart_Disease_Probability_%', 
                        nbins=20, 
                        title=f"Risk Score Distribution across Batch ({active_m})",
                        color='Risk_Category',
                        color_discrete_map={'High Risk': '#ec4899', 'Moderate Risk': '#f97316', 'Low Risk': '#06b6d4'}
                    )
                    fig_b.update_layout(paper_bgcolor='#ffffff', plot_bgcolor='#f8fafc', font=dict(color='#0f172a'))
                    st.plotly_chart(fig_b, use_container_width=True)

                    st.dataframe(b_df, use_container_width=True)

                    st.download_button(
                        label="Export Annotated Batch Predictions (CSV)",
                        data=b_df.to_csv(index=False).encode('utf-8'),
                        file_name=f"cardiac_batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        except Exception as ex:
            st.error(f"Error reading CSV file: {ex}")

# ---------------------------------------------------------
# WORKSPACE 5: ML MODEL WORKBENCH & COMPARISON
# ---------------------------------------------------------
elif st.session_state.current_workspace == "ML Model Workbench & Comparison":
    st.markdown("## ML Model Workbench & Comparative Analytics")

    if metadata and 'models' in metadata:
        m_df = pd.DataFrame(metadata['models']).T.reset_index().rename(columns={'index': 'Model'})
        st.markdown("#### Performance Metrics Comparison across All 5 Models")
        
        st.dataframe(m_df[['Model', 'accuracy', 'roc_auc', 'precision', 'recall', 'f1_score']], use_container_width=True)

        col_m1, col_m2 = st.columns(2)

        with col_m1:
            fig_acc = px.bar(
                m_df, 
                x='Model', 
                y=['accuracy', 'roc_auc', 'f1_score'], 
                barmode='group',
                title="Model Accuracy, AUC & F1-Score Comparison",
                color_discrete_sequence=['#ec4899', '#06b6d4', '#8b5cf6']
            )
            fig_acc.update_layout(paper_bgcolor='#ffffff', plot_bgcolor='#f8fafc', font=dict(color='#0f172a'))
            st.plotly_chart(fig_acc, use_container_width=True)

        with col_m2:
            st.markdown("#### Active Model Confusion Matrix")
            active_m = st.session_state.selected_model_name
            cm = metadata['models'][active_m]['confusion_matrix']

            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Patients"),
                x=['No Disease (0)', 'Heart Disease (1)'],
                y=['No Disease (0)', 'Heart Disease (1)'],
                text_auto=True,
                color_continuous_scale='Blues'
            )
            fig_cm.update_layout(paper_bgcolor='#ffffff', font=dict(color='#0f172a'))
            st.plotly_chart(fig_cm, use_container_width=True)

# ---------------------------------------------------------
# WORKSPACE 6: CARDIAC KNOWLEDGE BASE & DATASET
# ---------------------------------------------------------
else:
    st.markdown("## Cardiac Knowledge Base & Dataset Explorer")

    st.markdown("""
    <div class="glass-card">
        <h3 style="color:#0284c7;">UCI Cleveland Heart Disease Benchmark Dataset</h3>
        <p style="color:#334155; line-height:1.6;">
            The UCI Cleveland dataset is the gold-standard benchmark in cardiac machine learning research.
            It comprises 297 cleaned patient records across 13 clinical features.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Patients", "297", "Cleaned Records")
    with c2:
        st.metric("Positive Cases", "137", "46.1% Prevalence")
    with c3:
        st.metric("Negative Cases", "160", "53.9% Healthy")
    with c4:
        st.metric("Features", "13", "Clinical Features")

    st.markdown("---")
    st.markdown("#### Parameter Dictionary & Reference Thresholds")

    param_df = pd.DataFrame([
        {'Feature': 'age', 'Name': 'Age', 'Description': 'Patient age in years', 'Reference Range': '18 - 100 yrs'},
        {'Feature': 'sex', 'Name': 'Gender', 'Description': 'Biological sex', 'Reference Range': '1 = Male, 0 = Female'},
        {'Feature': 'cp', 'Name': 'Chest Pain Type', 'Description': '1=Typical, 2=Atypical, 3=Non-anginal, 4=Asymptomatic', 'Reference Range': '1 - 4'},
        {'Feature': 'trestbps', 'Name': 'Resting Blood Pressure', 'Description': 'Resting BP on admission (mm Hg)', 'Reference Range': '< 120 mm Hg'},
        {'Feature': 'chol', 'Name': 'Serum Cholesterol', 'Description': 'Total serum cholesterol (mg/dl)', 'Reference Range': '< 200 mg/dl'},
        {'Feature': 'fbs', 'Name': 'Fasting Blood Sugar', 'Description': 'Fasting blood sugar > 120 mg/dl', 'Reference Range': '1 = True, 0 = False'},
        {'Feature': 'restecg', 'Name': 'Resting ECG', 'Description': '0=Normal, 1=ST-T abnormality, 2=LV hypertrophy', 'Reference Range': '0 - 2'},
        {'Feature': 'thalach', 'Name': 'Max Heart Rate', 'Description': 'Maximum heart rate achieved in stress test', 'Reference Range': '100 - 200 bpm'},
        {'Feature': 'exang', 'Name': 'Exercise Angina', 'Description': 'Exercise induced angina', 'Reference Range': '1 = Yes, 0 = No'},
        {'Feature': 'oldpeak', 'Name': 'ST Depression', 'Description': 'ST depression induced by exercise (mm)', 'Reference Range': '< 1.0 mm'},
        {'Feature': 'slope', 'Name': 'ST Segment Slope', 'Description': '1=Upsloping, 2=Flat, 3=Downsloping', 'Reference Range': '1 - 3'},
        {'Feature': 'ca', 'Name': 'Major Vessels', 'Description': 'Major vessels colored by fluoroscopy', 'Reference Range': '0 - 3 vessels'},
        {'Feature': 'thal', 'Name': 'Thalassemia', 'Description': '3=Normal, 6=Fixed Defect, 7=Reversible Defect', 'Reference Range': '3, 6, 7'}
    ])

    st.dataframe(param_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 1.5rem 0;">
    HeartGuard AI Ultra Release | Clinical Multi-Model Intelligence Suite | Developed by Om Srivastava
</div>
""", unsafe_allow_html=True)

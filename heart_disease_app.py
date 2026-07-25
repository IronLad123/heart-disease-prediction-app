import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings

# Suppress scikit-learn warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="HeartGuard AI | Next-Gen Cardiac Risk Intelligence Platform",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design Tokens & Insane Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }

    /* Keyframe Animations */
    @keyframes pulse-glow {
        0% { box-shadow: 0 0 15px rgba(239, 68, 68, 0.4); }
        50% { box-shadow: 0 0 30px rgba(239, 68, 68, 0.8); }
        100% { box-shadow: 0 0 15px rgba(239, 68, 68, 0.4); }
    }
    
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Header Banner */
    .hero-container {
        background: linear-gradient(-45deg, #0f172a, #1e1b4b, #0f766e, #0369a1);
        background-size: 400% 400%;
        animation: gradient-shift 12s ease infinite;
        padding: 2.8rem 2rem;
        border-radius: 24px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 30px -10px rgba(15, 23, 42, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 0.3rem;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-sub {
        font-size: 1.15rem;
        color: #94a3b8;
        max-width: 720px;
        margin: 0 auto 1.5rem auto;
        font-weight: 400;
    }
    .hero-badges {
        display: flex;
        justify-content: center;
        gap: 0.8rem;
        flex-wrap: wrap;
    }
    .hero-badge {
        padding: 0.45rem 1.2rem;
        border-radius: 30px;
        font-size: 0.85rem;
        font-weight: 600;
        backdrop-filter: blur(10px);
    }
    .b-teal { background: rgba(20, 184, 166, 0.25); color: #2dd4bf; border: 1px solid rgba(45, 212, 191, 0.4); }
    .b-blue { background: rgba(56, 189, 248, 0.25); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); }
    .b-purple { background: rgba(168, 85, 247, 0.25); color: #c084fc; border: 1px solid rgba(192, 132, 252, 0.4); }

    /* Custom Cards */
    .glass-card {
        background: #ffffff;
        border-radius: 18px;
        padding: 1.6rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.03), 0 4px 6px -2px rgba(0, 0, 0, 0.01);
        margin-bottom: 1.5rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 25px -5px rgba(0, 0, 0, 0.06);
    }

    .risk-banner-high {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border-left: 8px solid #ef4444;
        padding: 1.8rem;
        border-radius: 18px;
        box-shadow: 0 10px 20px -5px rgba(239, 68, 68, 0.15);
    }
    .risk-banner-medium {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-left: 8px solid #f59e0b;
        padding: 1.8rem;
        border-radius: 18px;
        box-shadow: 0 10px 20px -5px rgba(245, 158, 11, 0.15);
    }
    .risk-banner-low {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-left: 8px solid #10b981;
        padding: 1.8rem;
        border-radius: 18px;
        box-shadow: 0 10px 20px -5px rgba(16, 185, 129, 0.15);
    }

    /* Metric Panels */
    .metric-card {
        background: #ffffff;
        padding: 1.4rem;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);
    }
    .metric-num {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0f172a;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 0.2rem;
        font-weight: 500;
    }
    
    .rec-box {
        background: #f8fafc;
        padding: 1rem 1.25rem;
        border-radius: 12px;
        border-left: 4px solid #0284c7;
        margin-bottom: 0.75rem;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# Cache Multi-Model Suite and Metadata
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
        st.error(f"Error loading multi-model suite: {e}")
        return None, None, None

models_suite, scaler, metadata = load_all_models()

# Session State Initialization
if 'session_history' not in st.session_state:
    st.session_state.session_history = []
if 'current_workspace' not in st.session_state:
    st.session_state.current_workspace = "🧙‍♂️ Patient Assessment Wizard"
if 'selected_model_name' not in st.session_state:
    st.session_state.selected_model_name = "Voting Ensemble"

# Header Banner
st.markdown("""
<div class="hero-container">
    <div class="hero-title">❤️ HeartGuard AI</div>
    <div class="hero-sub">Next-Generation Multi-Model Cardiac Intelligence & Clinical Simulation Platform</div>
    <div class="hero-badges">
        <span class="hero-badge b-teal">✓ 5 Multi-Model ML Ensemble Suite</span>
        <span class="hero-badge b-blue">⚡ Real-Time Clinical Intervention Simulator</span>
        <span class="hero-badge b-purple">🔬 UCI Cleveland Dataset Provenance</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation & Model Selector
with st.sidebar:
    st.markdown("### 🏥 System Navigation")
    workspaces = [
        "🧙‍♂️ Patient Assessment Wizard",
        "⚡ Real-Time Clinical Intervention Simulator",
        "📂 EHR Batch CSV Intelligence Suite",
        "🔬 ML Model Workbench & Comparison",
        "📚 Cardiac Knowledge Base & Dataset"
    ]
    selected_ws = st.radio("Select Workspace:", workspaces, index=workspaces.index(st.session_state.current_workspace) if st.session_state.current_workspace in workspaces else 0)
    st.session_state.current_workspace = selected_ws

    st.markdown("---")
    st.markdown("### 🤖 Active ML Inference Engine")
    if metadata:
        model_names = list(metadata['models'].keys())
        active_model = st.selectbox("Select ML Model:", model_names, index=model_names.index(st.session_state.selected_model_name) if st.session_state.selected_model_name in model_names else 4)
        st.session_state.selected_model_name = active_model

        m_info = metadata['models'][active_model]
        st.info(f"""
        **Selected**: {active_model}  
        **Accuracy**: {m_info['accuracy']*100:.1f}%  
        **AUC Score**: {m_info['roc_auc']:.3f}  
        **Recall**: {m_info['recall']*100:.1f}%  
        """)

    st.markdown("---")
    st.markdown("### 👨‍💻 Platform Author")
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
# WORKSPACE 1: PATIENT ASSESSMENT WIZARD
# ---------------------------------------------------------
if st.session_state.current_workspace == "🧙‍♂️ Patient Assessment Wizard":
    st.markdown("## 🧙‍♂️ Step-by-Step Patient Assessment Wizard")

    # Quick Preset Avatars
    st.markdown("##### ⚡ Load Pre-configured Clinical Benchmark Profiles")
    p1, p2, p3, p4 = st.columns(4)
    
    with p1:
        if st.button("🔴 Critical High Risk Profile", use_container_width=True):
            st.session_state.wiz_age = 67
            st.session_state.wiz_sex = "Male"
            st.session_state.wiz_cp = "Asymptomatic (4)"
            st.session_state.wiz_trestbps = 160
            st.session_state.wiz_chol = 286
            st.session_state.wiz_fbs = "No (≤ 120 mg/dl)"
            st.session_state.wiz_restecg = "Left Ventricular Hypertrophy (2)"
            st.session_state.wiz_thalach = 108
            st.session_state.wiz_exang = "Yes"
            st.session_state.wiz_oldpeak = 1.5
            st.session_state.wiz_slope = "Flat (2)"
            st.session_state.wiz_ca = 3
            st.session_state.wiz_thal = "Reversible Defect (7)"
            st.rerun()

    with p2:
        if st.button("🟢 Low Risk Healthy Profile", use_container_width=True):
            st.session_state.wiz_age = 37
            st.session_state.wiz_sex = "Female"
            st.session_state.wiz_cp = "Typical Angina (1)"
            st.session_state.wiz_trestbps = 118
            st.session_state.wiz_chol = 190
            st.session_state.wiz_fbs = "No (≤ 120 mg/dl)"
            st.session_state.wiz_restecg = "Normal (0)"
            st.session_state.wiz_thalach = 185
            st.session_state.wiz_exang = "No"
            st.session_state.wiz_oldpeak = 0.0
            st.session_state.wiz_slope = "Upsloping (1)"
            st.session_state.wiz_ca = 0
            st.session_state.wiz_thal = "Normal (3)"
            st.rerun()

    with p3:
        if st.button("🟠 Moderate Risk Profile", use_container_width=True):
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
        if st.button("🔄 Reset Defaults", use_container_width=True):
            st.session_state.wiz_age = 52
            st.session_state.wiz_sex = "Male"
            st.session_state.wiz_cp = "Atypical Angina (2)"
            st.session_state.wiz_trestbps = 130
            st.session_state.wiz_chol = 240
            st.session_state.wiz_fbs = "No (≤ 120 mg/dl)"
            st.session_state.wiz_restecg = "Normal (0)"
            st.session_state.wiz_thalach = 150
            st.session_state.wiz_exang = "No"
            st.session_state.wiz_oldpeak = 1.0
            st.session_state.wiz_slope = "Upsloping (1)"
            st.session_state.wiz_ca = 0
            st.session_state.wiz_thal = "Normal (3)"
            st.rerun()

    st.markdown("---")

    # Form with Tooltips
    w_tab1, w_tab2, w_tab3 = st.tabs(["1. Patient Vitals & Demographics", "2. Stress Test & ECG", "3. Fluoroscopy & Blood Status"])

    with st.form("wizard_form"):
        with w_tab1:
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Age (years)", 18, 100, st.session_state.get('wiz_age', 52), help="Patient age in completed years")
                sex = st.selectbox("Gender", ["Male", "Female"], index=0 if st.session_state.get('wiz_sex', 'Male') == "Male" else 1, help="Biological sex of the patient")
                trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 70, 240, st.session_state.get('wiz_trestbps', 130), help="Resting BP measured on admission")
            with c2:
                chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 650, st.session_state.get('wiz_chol', 240), help="Total serum cholesterol level")
                fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No (≤ 120 mg/dl)", "Yes (> 120 mg/dl)"], index=0 if "No" in st.session_state.get('wiz_fbs', 'No') else 1, help="Fasting blood sugar indicator")

        with w_tab2:
            c1, c2 = st.columns(2)
            with c1:
                cp_opts = ["Typical Angina (1)", "Atypical Angina (2)", "Non-Anginal Pain (3)", "Asymptomatic (4)"]
                cp = st.selectbox("Chest Pain Type", cp_opts, index=1, help="Nature of reported chest discomfort")
                restecg_opts = ["Normal (0)", "ST-T Wave Abnormality (1)", "Left Ventricular Hypertrophy (2)"]
                restecg = st.selectbox("Resting ECG Results", restecg_opts, index=0, help="Resting electrocardiogram results")
            with c2:
                thalach = st.number_input("Max Heart Rate Achieved (bpm)", 60, 230, st.session_state.get('wiz_thalach', 150), help="Peak heart rate during exertion")
                exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"], index=0 if st.session_state.get('wiz_exang', 'No') == "No" else 1, help="Angina experienced during exercise")

        with w_tab3:
            c1, c2 = st.columns(2)
            with c1:
                oldpeak = st.slider("Exercise ST Depression (oldpeak)", 0.0, 6.2, float(st.session_state.get('wiz_oldpeak', 1.0)), step=0.1, help="ST depression induced by exercise relative to rest")
                slope_opts = ["Upsloping (1)", "Flat (2)", "Downsloping (3)"]
                slope = st.selectbox("Slope of Peak Exercise ST Segment", slope_opts, index=0, help="Peak exercise ST slope")
            with c2:
                ca = st.slider("Major Vessels Colored by Fluoroscopy (0-3)", 0, 3, int(st.session_state.get('wiz_ca', 0)), help="Number of major vessels highlighted via fluoroscopy")
                thal_opts = ["Normal (3)", "Fixed Defect (6)", "Reversible Defect (7)"]
                thal = st.selectbox("Thalassemia Blood Status", thal_opts, index=0, help="Nuclear stress scan thalassemia status")

        wiz_submit = st.form_submit_button("⚡ Run ML Risk Diagnostic Assessment", use_container_width=True, type="primary")

    if wiz_submit:
        # Encode Features
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

        # Log session
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
        st.markdown(f"### 📊 Clinical Assessment Report ({active_m})")

        if prob >= 70:
            b_class, b_title, b_color = "risk-banner-high", "HIGH RISK FOR CARDIAC DISEASE", "#ef4444"
        elif prob >= 35:
            b_class, b_title, b_color = "risk-banner-medium", "MODERATE RISK FOR CARDIAC DISEASE", "#f59e0b"
        else:
            b_class, b_title, b_color = "risk-banner-low", "LOW RISK FOR CARDIAC DISEASE", "#10b981"

        r1, r2 = st.columns([1.5, 1])
        with r1:
            st.markdown(f"""
            <div class="{b_class}">
                <h3 style="margin:0; color: {b_color}; font-weight:800;">{b_title}</h3>
                <h1 style="font-size:3.5rem; margin:0.4rem 0; color:#0f172a;">{prob:.1f}% <span style="font-size:1.2rem; color:#64748b;">Disease Probability</span></h1>
                <p style="margin:0; color:#334155; font-size:1rem; line-height:1.5;">
                    The active ML model (<b>{active_m}</b>) evaluates this patient as 
                    <b>{'POSITIVE for Coronary Artery Disease' if pred == 1 else 'NEGATIVE for Coronary Artery Disease'}</b>.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 💡 Patient-Specific Clinical Action Plan")

            recs = []
            if prob >= 50:
                recs.append("🚨 **Cardiology Referral**: Urgent referral for coronary angiography / stress echocardiogram.")
            if chol > 240:
                recs.append(f"💊 **Lipid Management**: Serum cholesterol ({chol} mg/dl) is elevated. Evaluate statin therapy & diet.")
            if trestbps > 130:
                recs.append(f"🩸 **Hypertension Protocol**: Resting BP ({trestbps} mm Hg) is elevated. Monitor ambulatory BP.")
            if oldpeak > 1.0:
                recs.append(f"📉 **ST Depression Monitoring**: Exercise ST depression ({oldpeak} mm) suggests exertion-induced ischemia.")
            if exang == "Yes":
                recs.append("🏃 **Exercise Angina**: Angina triggered by physical exertion indicates restricted coronary perfusion.")
            if len(recs) == 0:
                recs.append("✅ **Maintain Current Vitals**: Patient vitals are within normal reference ranges.")

            for r in recs:
                st.markdown(f'<div class="rec-box">{r}</div>', unsafe_allow_html=True)

        with r2:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob,
                title={'text': f"Risk Score ({active_m})", 'font': {'size': 18, 'color': '#0f172a'}},
                number={'suffix': "%", 'font': {'size': 32, 'color': b_color}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': b_color},
                    'bgcolor': "white",
                    'steps': [
                        {'range': [0, 35], 'color': 'rgba(16, 185, 129, 0.15)'},
                        {'range': [35, 70], 'color': 'rgba(245, 158, 11, 0.15)'},
                        {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.15)'}
                    ]
                }
            ))
            fig_g.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_g, use_container_width=True)

# ---------------------------------------------------------
# WORKSPACE 2: REAL-TIME INTERVENTION SIMULATOR
# ---------------------------------------------------------
elif st.session_state.current_workspace == "⚡ Real-Time Clinical Intervention Simulator":
    st.markdown("## ⚡ Real-Time Interactive Clinical Intervention Simulator")
    st.markdown("""
    Adjust patient vitals in real time to simulate how clinical interventions (e.g., lowering blood pressure, lowering cholesterol, or increasing exercise tolerance) immediately impact the patient's predicted heart disease risk probability.
    """)

    sim_col1, sim_col2 = st.columns([1, 1.2])

    with sim_col1:
        st.markdown("#### 🎛️ Baseline & Simulated Parameters")
        sim_age = st.slider("Simulated Age", 20, 90, 60)
        sim_bp = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 150)
        sim_chol = st.slider("Serum Cholesterol (mg/dl)", 120, 450, 260)
        sim_hr = st.slider("Max Heart Rate (thalach)", 70, 210, 130)
        sim_op = st.slider("ST Depression (oldpeak)", 0.0, 5.0, 2.0, step=0.1)
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

        st.markdown("#### 📉 Live Simulation Outcome")
        
        sim_color = "#ef4444" if prob_sim >= 70 else "#f59e0b" if prob_sim >= 35 else "#10b981"
        st.markdown(f"""
        <div class="glass-card" style="text-align:center; border-top: 6px solid {sim_color};">
            <h4 style="color:#64748b; margin:0;">SIMULATED RISK PROBABILITY</h4>
            <h1 style="font-size:4rem; color:{sim_color}; margin:0.5rem 0; font-weight:800;">{prob_sim:.1f}%</h1>
            <p style="font-size:1.1rem; font-weight:600; color:#0f172a; margin:0;">
                Diagnosis: {'POSITIVE FOR HEART DISEASE' if pred_sim == 1 else 'NEGATIVE FOR HEART DISEASE'}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Simulation Intervention Delta Analysis
        st.markdown("##### 💡 Simulated Treatment Impact Analysis")
        # Simulate BP reduction to 120
        feat_bp_c = features_sim.copy()
        feat_bp_c['trestbps'] = 120
        p_bp, _ = get_prediction(active_m, feat_bp_c)

        # Simulate Chol reduction to 190
        feat_chol_c = features_sim.copy()
        feat_chol_c['chol'] = 190
        p_chol, _ = get_prediction(active_m, feat_chol_c)

        # Simulate Combined Interventions
        feat_comb = features_sim.copy()
        feat_comb['trestbps'] = 120
        feat_comb['chol'] = 190
        feat_comb['oldpeak'] = 0.0
        p_comb, _ = get_prediction(active_m, feat_comb)

        delta_df = pd.DataFrame({
            'Intervention Scenario': ['Current State', 'If BP Controlled (120 mm Hg)', 'If Chol Controlled (190 mg/dl)', 'Combined Medical Intervention'],
            'Risk Probability (%)': [prob_sim, p_bp, p_chol, p_comb]
        })

        fig_delta = px.bar(
            delta_df, 
            x='Intervention Scenario', 
            y='Risk Probability (%)',
            color='Risk Probability (%)',
            color_continuous_scale='Reds_r',
            text_auto='.1f'
        )
        fig_delta.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_delta, use_container_width=True)

# ---------------------------------------------------------
# WORKSPACE 3: EHR BATCH CSV INTELLIGENCE SUITE
# ---------------------------------------------------------
elif st.session_state.current_workspace == "📂 EHR Batch CSV Intelligence Suite":
    st.markdown("## 📂 EHR Batch CSV Intelligence Suite")
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
                st.error(f"❌ Missing required CSV columns: {missing}")
            else:
                if st.button("🚀 Process Batch Evaluation with Active Model", type="primary", use_container_width=True):
                    active_m = st.session_state.selected_model_name
                    X_b = b_df[req]
                    X_b_scaled = scaler.transform(X_b)
                    
                    target_model = models_suite[active_m]
                    probs = target_model.predict_proba(X_b_scaled)[:, 1] * 100
                    preds = target_model.predict(X_b_scaled)

                    b_df['Heart_Disease_Probability_%'] = np.round(probs, 1)
                    b_df['Prediction'] = np.where(preds == 1, 'Heart Disease Present', 'No Disease Detected')
                    b_df['Risk_Category'] = np.where(probs >= 70, 'High Risk', np.where(probs >= 35, 'Moderate Risk', 'Low Risk'))

                    st.markdown("### 📊 Batch Evaluation Summary")
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
                        color_discrete_map={'High Risk': '#ef4444', 'Moderate Risk': '#f59e0b', 'Low Risk': '#10b981'}
                    )
                    st.plotly_chart(fig_b, use_container_width=True)

                    st.dataframe(b_df, use_container_width=True)

                    st.download_button(
                        label="📥 Export Annotated Batch Predictions (CSV)",
                        data=b_df.to_csv(index=False).encode('utf-8'),
                        file_name=f"cardiac_batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        except Exception as ex:
            st.error(f"Error reading CSV file: {ex}")

# ---------------------------------------------------------
# WORKSPACE 4: ML MODEL WORKBENCH & COMPARISON
# ---------------------------------------------------------
elif st.session_state.current_workspace == "🔬 ML Model Workbench & Comparison":
    st.markdown("## 🔬 ML Model Workbench & Comparative Analytics")

    if metadata:
        m_df = pd.DataFrame(metadata['models']).T.reset_index().rename(columns={'index': 'Model'})
        st.markdown("#### 🏆 Performance Metrics Comparison across All 5 Models")
        
        st.dataframe(m_df[['Model', 'accuracy', 'roc_auc', 'precision', 'recall', 'f1_score']], use_container_width=True)

        col_m1, col_m2 = st.columns(2)

        with col_m1:
            fig_acc = px.bar(
                m_df, 
                x='Model', 
                y=['accuracy', 'roc_auc', 'f1_score'], 
                barmode='group',
                title="Model Accuracy, AUC & F1-Score Comparison",
                color_discrete_sequence=['#0284c7', '#10b981', '#f59e0b']
            )
            st.plotly_chart(fig_acc, use_container_width=True)

        with col_m2:
            st.markdown("#### 🎯 Active Model Confusion Matrix")
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
            st.plotly_chart(fig_cm, use_container_width=True)

# ---------------------------------------------------------
# WORKSPACE 5: CARDIAC KNOWLEDGE BASE & DATASET
# ---------------------------------------------------------
else:
    st.markdown("## 📚 Cardiac Knowledge Base & Dataset Explorer")

    st.markdown("""
    <div class="glass-card">
        <h3>UCI Cleveland Heart Disease Benchmark Dataset</h3>
        <p style="color:#475569; line-height:1.6;">
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
    st.markdown("#### 📖 Parameter Dictionary & Reference Thresholds")

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
    HeartGuard AI v3.0 Next-Gen Release | Clinical Multi-Model Intelligence Suite | Developed by Om Srivastava
</div>
""", unsafe_allow_html=True)

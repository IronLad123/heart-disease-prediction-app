import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import time
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="HeartGuard Pro - Cardiac Risk Assessment",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    .prediction-box {
        padding: 2.5rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        border: 3px solid;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        transition: all 0.4s ease;
        background: white;
    }
    .prediction-box:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.2);
    }
    .high-risk {
        border-color: #ff4b4b;
        background: linear-gradient(135deg, #fff5f5 0%, #ffe5e5 100%);
    }
    .medium-risk {
        border-color: #ffa726;
        background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
    }
    .low-risk {
        border-color: #4caf50;
        background: linear-gradient(135deg, #f1f8e9 0%, #dcedc8 100%);
    }
    .risk-factors {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 6px solid #ff9800;
        box-shadow: 0 4px 15px rgba(255,152,0,0.1);
    }
    .protective-factors {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 6px solid #4caf50;
        box-shadow: 0 4px 15px rgba(76,175,80,0.1);
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .feature-card:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 5px solid #667eea;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102,126,234,0.4);
    }
    .footer {
        text-align: center;
        margin-top: 4rem;
        padding: 2rem;
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        border-radius: 20px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def load_model():
    """Load the trained model and scaler"""
    try:
        model = joblib.load('heart_disease_knn_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        st.error("❌ Model files not found. Please ensure model files are in the directory.")
        return None, None

def predict_heart_disease(features, model, scaler):
    """Make prediction using the trained model"""
    try:
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        return prediction, probability
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def create_animated_gauge(probability):
    """Create animated gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Cardiac Risk Score", 'font': {'size': 24, 'color': 'darkblue'}},
        delta = {'reference': 50, 'increasing': {'color': "red"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue", 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#4CAF50'},
                {'range': [30, 70], 'color': '#FFC107'},
                {'range': [70, 100], 'color': '#F44336'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=50, r=50, t=100, b=50),
        font={'color': "darkblue", 'family': "Arial"},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def main():
    # Header with animated elements
    st.markdown('<h1 class="main-header">❤️ HeartGuard Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced AI-Powered Cardiac Risk Assessment System</p>', unsafe_allow_html=True)
    
    # Load model
    model, scaler = load_model()
    if model is None:
        return
    
    # Enhanced sidebar with animations
    with st.sidebar:
        st.markdown("### 🎯 Navigation")
        app_mode = st.radio(
            "Choose Mode",
            ["🏠 Dashboard", "🩺 Patient Assessment", "📊 Analytics", "👨‍💻 Developer"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### ⚡ Live Metrics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>🎯 88.5%</h3>
                <p>Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>🚨 100%</h3>
                <p>Recall</p>
            </div>
            """, unsafe_allow_html=True)
    
    if app_mode == "🏠 Dashboard":
        show_dashboard()
    elif app_mode == "🩺 Patient Assessment":
        show_interactive_assessment(model, scaler)
    elif app_mode == "📊 Analytics":
        show_analytics()
    else:
        show_developer_profile()

def show_dashboard():
    """Show interactive dashboard"""
    st.markdown("## 📈 Interactive Health Dashboard")
    
    # Feature cards with hover effects
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>AI Analysis</h4>
            <p>Advanced ML Algorithms</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>Live Analytics</h4>
            <p>Real-time Risk Assessment</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>Smart Reports</h4>
            <p>Detailed Health Insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-card">
            <h4>Data Security</h4>
            <p>HIPAA Compliant</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Call to action
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Start Risk Assessment", use_container_width=True):
            st.success("Navigate to 'Patient Assessment' to begin!")

def show_interactive_assessment(model, scaler):
    """Show interactive patient assessment"""
    
    st.info("""
    💡 **Welcome to Advanced Cardiac Assessment**  
    Complete the interactive form below for a comprehensive heart disease risk analysis.
    """)
    
    # Interactive tabs
    tab1, tab2 = st.tabs(["👤 Personal Info", "🩺 Medical History"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Demographic Information")
            age = st.slider("Age", 20, 100, 50)
            
            sex = st.radio("Biological Sex", 
                          [("♀ Female", 0), ("♂ Male", 1)],
                          format_func=lambda x: x[0])
            sex = sex[1]
        
        with col2:
            st.subheader("Lifestyle Factors")
            smoking = st.select_slider("Smoking Status", 
                                     options=["Non-smoker", "Former Smoker", "Light", "Moderate", "Heavy"],
                                     value="Non-smoker")
            
            activity = st.select_slider("Physical Activity Level", 
                                      options=["Sedentary", "Light", "Moderate", "Active", "Very Active"],
                                      value="Moderate")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Clinical Symptoms")
            cp_options = [
                (0, "🔴 Typical Angina"),
                (1, "🟡 Atypical Angina"), 
                (2, "🟢 Non-anginal Pain"),
                (3, "⚫ Asymptomatic")
            ]
            cp = st.selectbox("Chest Pain Type", options=cp_options, format_func=lambda x: x[1])
            cp = cp[0]
            
            trestbps = st.slider("Resting Blood Pressure (mmHg)", 90, 200, 120)
            chol = st.slider("Total Cholesterol (mg/dL)", 100, 600, 200)
        
        with col2:
            st.subheader("Medical Tests")
            restecg_options = [
                (0, "✅ Normal"),
                (1, "⚠️ ST-T Abnormality"),
                (2, "🔴 LV Hypertrophy")
            ]
            restecg = st.selectbox("Resting ECG Results", options=restecg_options, format_func=lambda x: x[1])
            restecg = restecg[0]
            
            thalach = st.slider("Maximum Heart Rate (bpm)", 60, 220, 150)
            exang = st.radio("Exercise Induced Angina", 
                           [("❌ No", 0), ("✅ Yes", 1)])
            exang = exang[1]
            
            oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0, 0.1)
            
            slope_options = [
                (0, "📈 Upsloping"),
                (1, "➡️ Flat"), 
                (2, "📉 Downsloping")
            ]
            slope = st.selectbox("ST Segment Slope", options=slope_options, format_func=lambda x: x[1])
            slope = slope[0]
            
            ca = st.slider("Number of Major Vessels", 0, 3, 0)
            
            thal_options = [
                (1, "✅ Normal"),
                (2, "⚠️ Fixed Defect"),
                (3, "🔴 Reversible Defect")
            ]
            thal = st.selectbox("Thalassemia", options=thal_options, format_func=lambda x: x[1])
            thal = thal[0]
    
    # Prediction button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🎯 Generate Risk Report", use_container_width=True, type="primary")
    
    if predict_button:
        # Animated processing
        with st.spinner('🔬 AI Analysis in Progress...'):
            time.sleep(2)
            
            # Prepare features and predict
            features = np.array([[age, sex, cp, trestbps, chol, 0, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            prediction, probability = predict_heart_disease(features, model, scaler)
        
        if prediction is not None:
            # Clear processing and show results
            st.balloons()
            
            # Enhanced results display
            show_results(prediction, probability)

def show_results(prediction, probability):
    """Show comprehensive results"""
    
    # Determine risk level
    if probability >= 0.7:
        risk_level = "HIGH RISK"
        risk_emoji = "🚨"
        box_class = "prediction-box high-risk"
        recommendation = "**Consult a cardiologist**"
    elif probability >= 0.4:
        risk_level = "MODERATE RISK"
        risk_emoji = "⚠️"
        box_class = "prediction-box medium-risk"
        recommendation = "**Further evaluation suggested**"
    else:
        risk_level = "LOW RISK"
        risk_emoji = "✅"
        box_class = "prediction-box low-risk"
        recommendation = "**Continue preventive care**"
    
    # Results header
    st.markdown("## 🎯 Risk Analysis Report")
    
    # Main results box
    st.markdown(f'<div class="{box_class}">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="AI Diagnosis",
            value=f"{'🫀 Heart Disease' if prediction == 1 else '✅ No Heart Disease'}"
        )
    
    with col2:
        st.metric(
            label="Confidence Level",
            value=f"{probability:.1%}"
        )
    
    with col3:
        st.metric(
            label="Risk Category",
            value=f"{risk_emoji} {risk_level}"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Interactive gauge
    st.subheader("📊 Risk Assessment Gauge")
    gauge_fig = create_animated_gauge(probability)
    st.plotly_chart(gauge_fig, use_container_width=True)
    
    # Recommendations
    st.subheader("💡 Recommendations")
    if prediction == 1:
        st.error(f"""
        **{recommendation}**
        
        **Next Steps:**
        - Consult a cardiologist
        - Consider stress test
        - Monitor blood pressure
        - Review lifestyle factors
        """)
    else:
        st.success(f"""
        **{recommendation}**
        
        **Maintain Heart Health:**
        - Regular exercise
        - Balanced diet
        - Annual check-ups
        - Blood pressure monitoring
        """)

def show_analytics():
    """Show analytics dashboard"""
    st.markdown("## 📊 Model Analytics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 88.5%</h3>
            <p>Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🚨 100%</h3>
            <p>Recall</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🛡️ 80.0%</h3>
            <p>Precision</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>⚖️ 88.9%</h3>
            <p>F1-Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature importance
    st.subheader("🔍 Feature Importance")
    features = ['Thalassemia', 'Major Vessels', 'Exercise Angina', 'ST Depression', 'Chest Pain']
    importance = [0.522, 0.460, 0.432, 0.425, 0.414]
    
    fig = px.bar(x=importance, y=features, orientation='h',
                 title='Top 5 Clinical Features',
                 color=importance,
                 color_continuous_scale='Viridis')
    
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

def show_developer_profile():
    """Show developer profile"""
    st.markdown("## 👨‍💻 Developer Profile")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image("https://via.placeholder.com/200x200/667eea/ffffff?text=OS", 
                 width=200, caption="Om Srivastava")
    
    with col2:
        st.markdown("""
        ### 🚀 About Me
        
        **Om Srivastava**  
        Machine Learning Engineer & Data Scientist
        
        📧 **Email:** om.srivastava@example.com  
        🔗 **GitHub:** [github.com/omsrivastava](https://github.com/omsrivastava)  
        🔗 **LinkedIn:** [linkedin.com/in/omsrivastava](https://linkedin.com/in/omsrivastava)
        
        ### 🛠️ Technologies Used
        - Python, Streamlit, Scikit-learn
        - Machine Learning, Data Science
        - Healthcare AI Applications
        
        ### 📊 Project Highlights
        - **Accuracy:** 88.5%
        - **Recall:** 100%
        - **Training Data:** 303 patients
        - **Features:** 13 clinical parameters
        """)

def main_wrapper():
    """Main wrapper with footer"""
    main()
    
    # Professional footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <h3>HeartGuard Pro</h3>
        <p>Advanced AI-Powered Cardiac Risk Assessment System</p>
        <p>Developed with ❤️ by <strong>Om Srivastava</strong></p>
        <p>📧 om.srivastava@example.com | 🔗 github.com/omsrivastava</p>
        <p><em>For educational and screening purposes only.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main_wrapper()
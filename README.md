# 🫀 HeartGuard AI — Heart Disease Risk Prediction Suite

<p align="center">
  <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Streamlit App" />
  <img src="https://img.shields.io/badge/Python-3.9%2B-0284c7?style=for-the-badge&logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/Accuracy-88.5%25-10b981?style=for-the-badge&logo=scikitlearn" alt="Accuracy" />
  <img src="https://img.shields.io/badge/Interpretability-TreeSHAP-8b5cf6?style=for-the-badge" alt="SHAP" />
</p>

<p align="center">
  <b>An AI-powered clinical risk assessment web application designed to evaluate patient cardiovascular risk using Machine Learning ensembles, feature interpretability via SHAP values, and instant medical parameter analysis.</b>
</p>

<p align="center">
  <a href="https://omheart-disease-prediction-app-c9c95zynkbx7ott7vvtmdum.streamlit.app"><strong>🌐 Try Live Streamlit App</strong></a> &nbsp;&middot;&nbsp;
  <a href="#-key-features">Key Features</a> &nbsp;&middot;&nbsp;
  <a href="#-clinical-parameters-evaluated">Clinical Parameters</a> &nbsp;&middot;&nbsp;
  <a href="#-tech-stack">Tech Stack</a> &nbsp;&middot;&nbsp;
  <a href="#-quick-start">Quick Start</a>
</p>

---

## 🌟 Key Features

- ⚡ **Instant Cardiovascular Assessment**: Real-time predictive health scoring based on 13 clinical parameters.
- 🎯 **88.5% Classification Accuracy**: Trained and evaluated on standard UCI Heart Disease datasets.
- 📊 **TreeSHAP Feature Interpretability**: Explains individual risk factors showing exactly which parameters increased or decreased patient risk.
- 🖥️ **Interactive Streamlit Dashboard**: Clean, responsive, user-friendly clinical UI with visual risk gauge.
- 🔒 **Client-Side Privacy**: Non-persistent data processing ensuring patient health data remains private.

---

## 🧠 Clinical Parameters Evaluated

The predictive backend evaluates 13 clinical features:

1. **Age**: Patient's age in years
2. **Sex**: Gender (`1` = Male, `0` = Female)
3. **Chest Pain Type (cp)**: Value 0–3 (`0`: Typical Angina, `1`: Atypical Angina, `2`: Non-anginal, `3`: Asymptomatic)
4. **Resting Blood Pressure (trestbps)**: Resting blood pressure in mm Hg on admission
5. **Serum Cholesterol (chol)**: Serum cholesterol level in mg/dl
6. **Fasting Blood Sugar (fbs)**: Fasting blood sugar > 120 mg/dl (`1` = True, `0` = False)
7. **Resting ECG Results (restecg)**: Values 0, 1, or 2
8. **Maximum Heart Rate Achieved (thalach)**: Maximum heart rate recorded during exercise
9. **Exercise Induced Angina (exang)**: `1` = Yes, `0` = No
10. **ST Depression (oldpeak)**: ST depression induced by exercise relative to rest
11. **Slope of Peak Exercise ST Segment (slope)**: Values 0, 1, or 2
12. **Number of Major Vessels (ca)**: Number of major vessels (0–3) colored by fluoroscopy
13. **Thalassemia (thal)**: `1` = Normal, `2` = Fixed defect, `3` = Reversible defect

---

## 🛠️ Tech Stack & Dependencies

- **Language**: Python 3.9+
- **Web Dashboard**: Streamlit
- **Machine Learning**: Scikit-learn, Random Forest, XGBoost
- **Interpretability**: SHAP (SHapley Additive exPlanations)
- **Data Processing**: Pandas, NumPy
- **Visualizations**: Matplotlib, Seaborn
- **Deployment**: Streamlit Cloud

---

## 🚀 Quick Start & Local Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/IronLad123/heart-disease-prediction-app.git
cd heart-disease-prediction-app
```

### 2️⃣ Create Virtual Environment

```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

Open your browser and navigate to **`http://localhost:8501`**.

---

## 📊 Model Performance

- **Target Metric**: Binary Classification (`0` = Low Risk / Normal, `1` = High Cardiovascular Risk)
- **Model Accuracy**: **88.5%**
- **Evaluation Metrics**: Precision, Recall, F1-Score, and ROC-AUC Curve analysis.

---

## ⚠️ Medical Disclaimer

This application is built for educational and research demonstration purposes only. It is not a substitute for professional medical advice, clinical diagnosis, or treatment. Always consult a qualified physician for healthcare decisions.

---

## 👨‍💻 Developer

**Om Srivastava ([@IronLad123](https://github.com/IronLad123))**  
Email: [srivastavaom078@gmail.com](mailto:srivastavaom078@gmail.com)  
*B.Tech in Computer Science & Engineering (Specialization in Data Science)*

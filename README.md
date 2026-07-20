# 🫀 Heart Disease Prediction App
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://omheart-disease-prediction-app-c9c95zynkbx7ott7vvtmdum.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-88.5%25-brightgreen)
An **AI-powered healthcare web application** designed to assess the risk of heart disease in patients using Machine Learning algorithms. The system analyzes clinical parameters to deliver instant, high-accuracy risk evaluations via an intuitive, interactive dashboard.
🌐 **Live Demo:** [Heart Disease Risk Assessment App](https://omheart-disease-prediction-app-c9c95zynkbx7ott7vvtmdum.streamlit.app)
---
## 📌 Features
- ⚡ **Instant Risk Assessment**: Get real-time predictive health insights based on patient inputs.
- 🎯 **High Accuracy**: Achieves **88.5% classification accuracy** trained on standard clinical datasets.
- 📊 **Interactive Data Visualization**: Explores medical metrics visually using charts and health indicators.
- 🖥️ **User-Friendly Dashboard**: Built with Streamlit for a clean, seamless user experience.
- 🔒 **Privacy-Focused**: Client-side parameter processing ensuring sensitive data isn't permanently stored.
---
## 🧠 Medical Parameters Evaluated
The model accepts 13 key clinical features for prediction:
1. **Age**: Patient's age in years
2. **Sex**: Gender (`1` = Male, `0` = Female)
3. **Chest Pain Type (cp)**: Value 0 to 3 (Typical angina, Atypical angina, Non-anginal pain, Asymptomatic)
4. **Resting Blood Pressure (trestbps)**: Resting blood pressure in mm Hg on admission
5. **Serum Cholesterol (chol)**: Serum cholesterol level in mg/dl
6. **Fasting Blood Sugar (fbs)**: Fasting blood sugar > 120 mg/dl (`1` = True, `0` = False)
7. **Resting ECG Results (restecg)**: Values 0, 1, or 2
8. **Maximum Heart Rate Achieved (thalach)**: Maximum heart rate recorded
9. **Exercise Induced Angina (exang)**: `1` = Yes, `0` = No
10. **ST Depression (oldpeak)**: ST depression induced by exercise relative to rest
11. **Slope of Peak Exercise ST Segment (slope)**: Values 0, 1, or 2
12. **Number of Major Vessels (ca)**: Number of major vessels (0–3) colored by fluoroscopy
13. **Thalassemia (thal)**: `1` = Normal, `2` = Fixed defect, `3` = Reversable defect
---
## 🛠️ Tech Stack & Dependencies
- **Language:** Python 3.9+
- **Frontend Framework:** Streamlit
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Deployment:** Streamlit Cloud
---
## 🚀 Local Installation & Setup
To run this application on your local machine, follow these steps:
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/IronLad123/heart-disease-prediction-app.git
cd heart-disease-prediction-app
2️⃣ Create a Virtual Environment (Optional but recommended)
bash


# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
# On Windows
python -m venv venv
venv\Scripts\activate
3️⃣ Install Dependencies
bash


pip install -r requirements.txt
4️⃣ Launch the Streamlit App
bash


streamlit run app.py
Open your browser and navigate to http://localhost:8501.

📊 Model & Performance Metrics
The predictive backend utilizes Machine Learning algorithms (such as Logistic Regression / Random Forest) trained on the UCI Heart Disease Dataset.

Target Metric: Binary Classification (0 = Low Risk / Healthy, 1 = High Risk / Heart Disease Detected)
Model Accuracy: 88.5%
Evaluation Criteria: Precision, Recall, F1-Score, and ROC-AUC Curve analysis.

⚠️ Disclaimer
This application is built for educational and demonstration purposes only. It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for any health concerns.

## Developer
Om Srivastava
Email: srivastavaom078@gmail.com
## 🌐 Live Demo
**Live App:** 
https://omheart-disease-prediction-app-c9c95zynkbx7ott7vvtmdum.streamlit.app

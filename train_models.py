"""
HeartGuard Pro - Multi-Model Training Engine
Trains, evaluates, and exports multiple ML models (Random Forest, Gradient Boosting, KNN, Logistic Regression, Soft Voting Ensemble)
on the clean UCI Cleveland Heart Disease Dataset.
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

def train_and_export():
    print("🚀 Starting Multi-Model Training Engine...")

    # Load UCI Cleveland dataset
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
    column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    
    try:
        df = pd.read_csv(url, names=column_names, na_values='?')
        print("✅ Downloaded UCI Cleveland dataset.")
    except Exception as e:
        print(f"⚠️ Could not load from URL, trying local fallback: {e}")
        df = pd.read_csv('Heart Disease Data/processed.cleveland.data', names=column_names, na_values='?')

    # Clean dataset (drop NaNs and convert target to binary 0/1)
    df = df.dropna().reset_index(drop=True)
    df['target'] = (df['target'] > 0).astype(int)

    X = df.drop('target', axis=1)
    y = df['target']

    # Train/Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Standard Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Instantiate Models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7),
        'Logistic Regression': LogisticRegression(random_state=42)
    }

    # Add Soft Voting Ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('rf', models['Random Forest']),
            ('gb', models['Gradient Boosting']),
            ('knn', models['K-Nearest Neighbors']),
            ('lr', models['Logistic Regression'])
        ],
        voting='soft'
    )
    models['Voting Ensemble'] = ensemble

    # Train and Evaluate
    results = {}
    model_files = {}

    for name, model in models.items():
        print(f"⚙️ Training {name}...")
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Save model pickle file
        filename = f"model_{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, filename)
        model_files[name] = filename

        # Extract feature importances if available
        if hasattr(model, 'feature_importances_'):
            feat_imp = model.feature_importances_.tolist()
        elif name == 'Logistic Regression':
            feat_imp = np.abs(model.coef_[0]).tolist()
        else:
            # Correlation-based proxy for distance models
            feat_imp = np.abs(df.corr()['target'].drop('target').values).tolist()

        # Normalize feature importances
        sum_imp = sum(feat_imp) if sum(feat_imp) > 0 else 1.0
        feat_imp = [round(x / sum_imp, 4) for x in feat_imp]

        results[name] = {
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1_score': round(f1, 4),
            'roc_auc': round(auc, 4),
            'confusion_matrix': cm,
            'filename': filename,
            'feature_importance': dict(zip(X.columns, feat_imp))
        }

        print(f"   {name:20} -> Accuracy: {acc*100:.1f}%, AUC: {auc:.3f}, F1: {f1:.3f}")

    # Export scaler and backward-compatible model
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(models['Voting Ensemble'], 'heart_disease_knn_model.pkl')  # Backward compatibility

    # Export metadata
    metadata = {
        'features': X.columns.tolist(),
        'dataset_size': len(df),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'positive_cases': int(y.sum()),
        'negative_cases': int(len(y) - y.sum()),
        'models': results,
        'feature_names': {
            'age': 'Age (years)',
            'sex': 'Gender',
            'cp': 'Chest Pain Type',
            'trestbps': 'Resting Blood Pressure',
            'chol': 'Serum Cholesterol',
            'fbs': 'Fasting Blood Sugar > 120',
            'restecg': 'Resting ECG Results',
            'thalach': 'Max Heart Rate Achieved',
            'exang': 'Exercise Induced Angina',
            'oldpeak': 'ST Depression (oldpeak)',
            'slope': 'Slope of ST Segment',
            'ca': 'Major Vessels (ca)',
            'thal': 'Thalassemia'
        }
    }

    with open('models_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    with open('model_metadata.json', 'w') as f:
        json.dump({
            'model_name': 'Voting Ensemble',
            'accuracy': results['Voting Ensemble']['accuracy'],
            'precision': results['Voting Ensemble']['precision'],
            'recall': results['Voting Ensemble']['recall'],
            'f1_score': results['Voting Ensemble']['f1_score'],
            'features': X.columns.tolist(),
            'feature_importance': {'importance': {str(i): v for i, v in enumerate(results['Voting Ensemble']['feature_importance'].values())}}
        }, f, indent=2)

    print("🎉 All 5 Models, Scaler, and Metadata Successfully Saved!")

if __name__ == '__main__':
    train_and_export()

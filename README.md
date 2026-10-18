# Explainable AI (XAI) in Healthcare - Heart Disease Risk Prediction

An end-to-end Explainable AI (XAI) system for predicting cardiovascular disease risk. This project combines machine learning predictive models with state-of-the-art explainability methods (**SHAP** and **LIME**) through an interactive web interface.

---

## 📌 Features

- **Accurate Clinical Prediction**: Machine learning model trained on cardiovascular health indicators.
- **SHAP (SHapley Additive exPlanations)**: Provides global and local feature importance values based on cooperative game theory.
- **LIME (Local Interpretable Model-agnostic Explanations)**: Generates faithful local surrogate models around individual patient predictions.
- **Interactive UI**: A modern dashboard built with React and Vite to input clinical parameters and visualize real-time risk predictions with XAI explanations.
- **FastAPI Backend**: High-performance RESTful API serving predictions and explanation data.

---

## 🏗️ Project Architecture

```
xai-healthcare/
├── backend/
│   ├── app.py                # FastAPI server endpoints
│   ├── test_api.py           # API testing script
│   ├── model.pkl             # Trained ML model
│   ├── scaler.pkl            # Feature scaler
│   └── feature_names.pkl     # Feature definitions
├── frontend/
│   ├── src/                  # React components & UI logic
│   ├── package.json          # Node dependencies
│   └── vite.config.js        # Vite configuration
├── data/
│   ├── heart.csv             # Heart disease dataset
│   └── data_downloader.py    # Dataset acquisition script
├── models/
│   └── train_model.py        # Model training and artifact generation
├── xai/
│   ├── shap_explainer.py     # SHAP explanation module
│   └── lime_explainer.py     # LIME explanation module
├── Internship_report.pdf     # Project documentation & report
└── README.md
```

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python**: 3.9+
- **Node.js**: 18+

### 2. Backend Setup
```bash
cd backend
pip install fastapi uvicorn scikit-learn pandas numpy shap lime joblib
python app.py
```
The FastAPI documentation will be available at `http://localhost:8000/docs`.

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
The frontend will run at `http://localhost:5173`.

---

## 🔬 Explainability Methods

- **SHAP**: Computes the contribution of each clinical metric (cholesterol, resting blood pressure, max heart rate, etc.) toward the predicted risk score.
- **LIME**: Perturbs individual patient inputs to generate local linear decision boundaries, showing which clinical indicators increase or decrease disease probability.

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def train_model():
    # Load data
    data_path = "xai-healthcare/data/heart.csv"
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    
    # Features and Target
    X = df.drop("target", axis=1)
    y = df["target"]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to keep column names (important for SHAP/LIME)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save artifacts
    joblib.dump(model, "xai-healthcare/backend/model.pkl")
    joblib.dump(scaler, "xai-healthcare/backend/scaler.pkl")
    # Save feature names for reference
    joblib.dump(X.columns.tolist(), "xai-healthcare/backend/feature_names.pkl")
    
    print("Model and scaler saved to xai-healthcare/backend/")

if __name__ == "__main__":
    train_model()

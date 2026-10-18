import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

class ShapExplainer:
    def __init__(self, model_path, scaler_path, feature_names_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = joblib.load(feature_names_path)
        self.explainer = shap.TreeExplainer(self.model)

    def explain(self, input_data):
        # input_data is a dict or a single row DataFrame
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data

        # Scale
        input_scaled = self.scaler.transform(input_df)
        input_scaled_df = pd.DataFrame(input_scaled, columns=self.feature_names)

        # Get SHAP values
        shap_values = self.explainer(input_scaled_df)
        
        # Local explanation for the first row
        explanation = {
            "base_value": float(shap_values.base_values[0]),
            "shap_values": shap_values.values[0].tolist(),
            "feature_names": self.feature_names,
            "feature_values": input_df.iloc[0].tolist()
        }
        
        return explanation

    def get_summary_plot_base64(self, X_test):
        # Global explanation plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(self.explainer.shap_values(X_test), X_test, show=False)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return img_str

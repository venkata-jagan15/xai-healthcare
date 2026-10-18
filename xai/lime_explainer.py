from lime import lime_tabular
import joblib
import pandas as pd
import numpy as np

class LimeExplainer:
    def __init__(self, model_path, scaler_path, feature_names_path, train_data_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = joblib.load(feature_names_path)
        
        # Load training data for LIME background
        train_df = pd.read_csv(train_data_path)
        X_train = train_df.drop("target", axis=1)
        X_train_scaled = self.scaler.transform(X_train)
        
        self.explainer = lime_tabular.LimeTabularExplainer(
            training_data=np.array(X_train_scaled),
            feature_names=self.feature_names,
            class_names=['No Disease', 'Disease'],
            mode='classification'
        )

    def explain(self, input_data):
        # input_data is a dict or a single row DataFrame
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data

        # Scale
        input_scaled = self.scaler.transform(input_df)
        
        # Explain instance
        exp = self.explainer.explain_instance(
            input_scaled[0],
            self.model.predict_proba,
            num_features=len(self.feature_names)
        )
        
        return {
            "exp_list": exp.as_list(),
            "local_pred": exp.local_pred.tolist() if hasattr(exp.local_pred, 'tolist') else exp.local_pred,
            "intercept": exp.intercept
        }

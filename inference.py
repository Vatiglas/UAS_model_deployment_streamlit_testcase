"""
inference.py
============
Core inference engine for Credit Score Classification.
Loads the trained Random Forest model (TUNED via GridSearchCV: n_estimators=100,
max_depth=20, min_samples_split=2, accuracy 73.40%) + preprocessor + label
encoder, and exposes a clean predict() interface for the Streamlit app layer.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent

# ── Feature groups (must match training exactly) ───────────────────────────
NUMERICAL_FEATURES = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
    "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date",
    "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries",
    "Outstanding_Debt", "Credit_Utilization_Ratio", "Total_EMI_per_month",
    "Amount_invested_monthly", "Monthly_Balance", "Credit_History_Months",
]
ORDINAL_FEATURES = ["Credit_Mix"]
NOMINAL_FEATURES = ["Occupation", "Payment_Behaviour", "Payment_of_Min_Amount"]
ALL_FEATURES     = NUMERICAL_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES

OCCUPATION_OPTIONS = [
    "Accountant", "Architect", "Developer", "Doctor", "Engineer",
    "Entrepreneur", "Journalist", "Lawyer", "Manager", "Mechanic",
    "Media_Manager", "Musician", "Scientist", "Teacher", "Writer",
]
PAYMENT_BEHAVIOUR_OPTIONS = [
    "High_spent_Large_value_payments", "High_spent_Medium_value_payments",
    "High_spent_Small_value_payments", "Low_spent_Large_value_payments",
    "Low_spent_Medium_value_payments", "Low_spent_Small_value_payments",
]

LABEL_COLORS = {
    "Good":     "#2ecc71",
    "Standard": "#3498db",
    "Poor":     "#e74c3c",
}
LABEL_DESCRIPTIONS = {
    "Good":     "Nasabah memiliki riwayat kredit yang sangat baik. Risiko rendah.",
    "Standard": "Nasabah memiliki riwayat kredit yang cukup baik. Risiko moderat.",
    "Poor":     "Nasabah memiliki riwayat kredit yang buruk. Risiko tinggi.",
}


class CreditScoreInferenceEngine:
    """
    Loads saved artefacts (Random Forest — TUNED via GridSearchCV) and
    performs single inference.

    Parameters
    ----------
    model_path        : path to model_random_forest.pkl
    preprocessor_path : path to CreditDataPreprocessor .pkl
    label_encoder_path: path to LabelEncoder .pkl
    """

    def __init__(
        self,
        model_path:         str = None,
        preprocessor_path:  str = None,
        label_encoder_path: str = None,
    ):
        model_path          = model_path         or str(BASE_DIR / "model" / "model_random_forest.pkl")
        preprocessor_path   = preprocessor_path  or str(BASE_DIR / "model" / "preprocessor.pkl")
        label_encoder_path  = label_encoder_path or str(BASE_DIR / "model" / "label_encoder.pkl")

        self._model        = joblib.load(model_path)
        self._preprocessor = joblib.load(preprocessor_path)
        self._le           = joblib.load(label_encoder_path)

    def predict(self, input_data: dict) -> dict:
        """
        Predict credit score for a single input dictionary.

        Returns
        -------
        dict with keys: label, label_index, probabilities, color, description
        """
        df  = self._dict_to_dataframe(input_data)
        X_t = self._preprocessor.transform(df)

        pred_idx   = self._model.predict(X_t)[0]
        pred_proba = self._model.predict_proba(X_t)[0]

        label = self._le.inverse_transform([pred_idx])[0]
        proba_dict = {
            self._le.inverse_transform([i])[0]: float(round(p * 100, 2))
            for i, p in enumerate(pred_proba)
        }

        return {
            "label":         label,
            "label_index":   int(pred_idx),
            "probabilities": proba_dict,
            "color":         LABEL_COLORS[label],
            "description":   LABEL_DESCRIPTIONS[label],
        }

    @staticmethod
    def _dict_to_dataframe(data: dict) -> pd.DataFrame:
        """Convert raw input dict → properly typed DataFrame."""
        row = {
            "Age":                       float(data.get("Age", 30)),
            "Annual_Income":             float(data.get("Annual_Income", 50000)),
            "Monthly_Inhand_Salary":     float(data.get("Monthly_Inhand_Salary", 4000)),
            "Num_Bank_Accounts":         float(data.get("Num_Bank_Accounts", 3)),
            "Num_Credit_Card":           float(data.get("Num_Credit_Card", 3)),
            "Interest_Rate":             float(data.get("Interest_Rate", 15)),
            "Num_of_Loan":               float(data.get("Num_of_Loan", 2)),
            "Delay_from_due_date":       float(data.get("Delay_from_due_date", 10)),
            "Num_of_Delayed_Payment":    float(data.get("Num_of_Delayed_Payment", 5)),
            "Changed_Credit_Limit":      float(data.get("Changed_Credit_Limit", 5)),
            "Num_Credit_Inquiries":      float(data.get("Num_Credit_Inquiries", 3)),
            "Outstanding_Debt":          float(data.get("Outstanding_Debt", 1000)),
            "Credit_Utilization_Ratio":  float(data.get("Credit_Utilization_Ratio", 30)),
            "Total_EMI_per_month":       float(data.get("Total_EMI_per_month", 200)),
            "Amount_invested_monthly":   float(data.get("Amount_invested_monthly", 100)),
            "Monthly_Balance":           float(data.get("Monthly_Balance", 500)),
            "Credit_History_Months":     float(data.get("Credit_History_Months", 60)),
            "Credit_Mix":                str(data.get("Credit_Mix", "Standard")),
            "Occupation":                str(data.get("Occupation", "Engineer")),
            "Payment_Behaviour":         str(data.get("Payment_Behaviour", "Low_spent_Small_value_payments")),
            "Payment_of_Min_Amount":     str(data.get("Payment_of_Min_Amount", "No")),
        }
        return pd.DataFrame([row])[ALL_FEATURES]

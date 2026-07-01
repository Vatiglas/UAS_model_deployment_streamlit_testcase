"""
preprocessing.py
================
OOP-based preprocessing class for Credit Score Classification pipeline.
Handles data cleaning, feature engineering, and sklearn ColumnTransformer pipeline.

IMPORTANT — Data Leakage Prevention:
    clean() is a stateless, pure transformation (no fitting), so it is safe
    to call separately on train_df and test_df.
    fit_transform() fits the ColumnTransformer ONLY on training data;
    transform() must be used for test/validation/inference data.
"""

import re
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer


# ── Feature Groups ─────────────────────────────────────────────────────────────
NUMERICAL_FEATURES = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
    "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date",
    "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries",
    "Outstanding_Debt", "Credit_Utilization_Ratio", "Total_EMI_per_month",
    "Amount_invested_monthly", "Monthly_Balance", "Credit_History_Months",
]

ORDINAL_FEATURES  = ["Credit_Mix"]
NOMINAL_FEATURES  = ["Occupation", "Payment_Behaviour", "Payment_of_Min_Amount"]

CREDIT_MIX_ORDER  = ["Bad", "Standard", "Good"]

# Columns dropped before modelling (identifiers / free text)
DROP_COLS = ["Unnamed: 0", "ID", "Customer_ID", "SSN", "Name",
             "Month", "Type_of_Loan", "Credit_History_Age"]

TARGET_COL = "Credit_Score"


class CreditDataPreprocessor:
    """
    Cleans raw Credit Score CSV data and builds an sklearn preprocessing pipeline.

    Usage
    -----
    preprocessor = CreditDataPreprocessor()
    X_train_t, X_test_t = preprocessor.fit_transform(X_train, X_test)
    """

    def __init__(self):
        self._pipeline: ColumnTransformer | None = None
        self._is_fitted: bool = False

    # ── Static Cleaning Helpers ────────────────────────────────────────────────

    @staticmethod
    def _clean_numeric(series: pd.Series) -> pd.Series:
        """Extract first numeric token from a string series; non-parsable → NaN."""
        extracted = series.astype(str).str.extract(r"([\-]?\d+\.?\d*)")[0]
        return pd.to_numeric(extracted, errors="coerce")

    @staticmethod
    def _parse_credit_history(val) -> float:
        """Convert '9 Years and 8 Months' → 116 (total months)."""
        if pd.isna(val):
            return np.nan
        match = re.search(r"(\d+)\s*Years?\s*and\s*(\d+)\s*Months?", str(val))
        return int(match.group(1)) * 12 + int(match.group(2)) if match else np.nan

    # ── Public Interface ───────────────────────────────────────────────────────

    def clean(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Apply all cleaning rules to a raw DataFrame.

        Returns
        -------
        X : pd.DataFrame  — feature matrix (cleaned, pre-encoding)
        y : pd.Series     — target labels
        """
        df = df.copy()

        # --- Numeric columns stored as strings ---
        df["Age"] = self._clean_numeric(df["Age"])
        df.loc[(df["Age"] < 10) | (df["Age"] > 100), "Age"] = np.nan

        df["Annual_Income"] = pd.to_numeric(
            df["Annual_Income"].astype(str).str.replace("_", ""), errors="coerce"
        )
        df["Num_of_Loan"] = self._clean_numeric(df["Num_of_Loan"])
        df.loc[df["Num_of_Loan"] < 0, "Num_of_Loan"] = np.nan

        df["Num_of_Delayed_Payment"] = self._clean_numeric(df["Num_of_Delayed_Payment"])
        df.loc[df["Num_of_Delayed_Payment"] < 0, "Num_of_Delayed_Payment"] = np.nan

        df["Changed_Credit_Limit"] = pd.to_numeric(
            df["Changed_Credit_Limit"].astype(str).str.replace("_", ""), errors="coerce"
        )
        df["Outstanding_Debt"] = pd.to_numeric(
            df["Outstanding_Debt"].astype(str).str.replace("_", ""), errors="coerce"
        )
        df["Amount_invested_monthly"] = pd.to_numeric(
            df["Amount_invested_monthly"].astype(str).str.replace("_", ""), errors="coerce"
        )

        # --- Feature engineering ---
        df["Credit_History_Months"] = df["Credit_History_Age"].apply(
            self._parse_credit_history
        )

        # --- Categorical noise ---
        df["Credit_Mix"]        = df["Credit_Mix"].replace("_", np.nan)
        df["Occupation"]        = df["Occupation"].replace("_______", np.nan)
        df["Payment_Behaviour"] = df["Payment_Behaviour"].replace("!@9#%8", np.nan)

        # --- Extract target & drop irrelevant columns ---
        y = df[TARGET_COL].copy()
        X = df.drop(columns=DROP_COLS + [TARGET_COL], errors="ignore")
        X = X[NUMERICAL_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES]

        return X, y

    def build_pipeline(self) -> ColumnTransformer:
        """Construct and return the sklearn ColumnTransformer pipeline."""
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])
        ord_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(
                categories=[CREDIT_MIX_ORDER],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )),
        ])
        nom_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                drop="first", handle_unknown="ignore", sparse_output=False
            )),
        ])

        self._pipeline = ColumnTransformer([
            ("num", num_pipe, NUMERICAL_FEATURES),
            ("ord", ord_pipe, ORDINAL_FEATURES),
            ("nom", nom_pipe, NOMINAL_FEATURES),
        ], remainder="drop")

        return self._pipeline

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit on training data and transform both train and test."""
        if self._pipeline is None:
            self.build_pipeline()

        X_train_t = self._pipeline.fit_transform(X_train)
        X_test_t  = self._pipeline.transform(X_test)
        self._is_fitted = True
        return X_train_t, X_test_t

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform new data using a fitted pipeline."""
        if not self._is_fitted:
            raise RuntimeError("Preprocessor not fitted yet. Call fit_transform() first.")
        return self._pipeline.transform(X)

    def get_feature_names(self) -> list[str]:
        """Return all feature names after encoding."""
        if self._pipeline is None or not self._is_fitted:
            raise RuntimeError("Pipeline not fitted yet.")
        ohe_names = list(
            self._pipeline.named_transformers_["nom"]["encoder"]
            .get_feature_names_out(NOMINAL_FEATURES)
        )
        return NUMERICAL_FEATURES + ORDINAL_FEATURES + ohe_names

    @property
    def pipeline(self) -> ColumnTransformer:
        return self._pipeline

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

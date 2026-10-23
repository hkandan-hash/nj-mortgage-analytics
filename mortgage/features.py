"""Model scopes (which applications each model uses) and feature definitions.

Two rules keep the models honest:
  * Only information known when the application is decided is used. Interest rate, rate spread,
    loan costs and denial reasons exist only after the decision, so they never enter the
    denial model.
  * Race, ethnicity, sex and age are never model inputs. They are kept for the fairness audit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from . import config

# loan_to_income is deliberately left out: with log income and log loan amount both present it is
# collinear with them, and "income doubles, everything else equal" stops being interpretable.
NUMERIC = ["log_loan_amount", "log_income", "ltv", "dti", "units", "tract_minority_pct", "tract_income_pct"]
FLAGS = ["business_purpose", "manufactured"]
CATEGORICAL = ["loan_type", "occupancy", "aus", "credit_score_model", "conforming", "county"]
LENDER = ["lender_id"]
PROTECTED = ["race_ethnicity", "sex", "age_band"]  # audit only
# Reference levels for linear models, so each coefficient reads as "compared with the typical case".
REFERENCE = {"loan_type": "Conventional", "occupancy": "Principal residence", "aus": "Desktop Underwriter",
             "credit_score_model": "Equifax Beacon 5.0", "conforming": "Conforming", "county": "Bergen"}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_loan_amount"] = np.log(out.loan_amount)
    out["log_income"] = np.log(out.income_k * 1000)
    return out


def _funnel(df: pd.DataFrame, steps: list[tuple[str, pd.Series]]) -> tuple[pd.DataFrame, list[dict]]:
    """Apply filters in order, recording how many rows each one removes."""
    log = [{"step": "All applications with a credit decision", "rows": len(df), "removed": 0}]
    keep = pd.Series(True, index=df.index)
    for name, condition in steps:
        before = int(keep.sum())
        keep &= condition.fillna(False)
        log.append({"step": name, "rows": int(keep.sum()), "removed": before - int(keep.sum())})
    return df[keep].copy(), log


def classification_scope(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Applications with complete financials, so the model learns from the applicant's profile
    rather than from which fields happen to be blank on incomplete applications."""
    return _funnel(add_features(df), [
        ("Not exempt from detailed reporting", df.is_exempt.eq(0)),
        ("Closed-end, not a reverse mortgage", df.open_end.eq(0) & df.reverse_mortgage.eq(0)),
        ("Known county", df.county.ne("Unknown")),
        ("Income, property value, LTV and DTI all reported",
         df[["income_k", "property_value", "ltv", "dti", "loan_amount"]].notna().all(axis=1)),
    ])


def regression_scope(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Originated 30-year fixed-rate first-lien purchase loans with a reported rate spread."""
    return _funnel(add_features(df), [
        ("Originated", df.action.eq("Originated")),
        ("Not exempt from detailed reporting", df.is_exempt.eq(0)),
        ("First lien, closed-end, not reverse", df.lien.eq("First lien") & df.open_end.eq(0) & df.reverse_mortgage.eq(0)),
        ("30-year fixed rate", df.loan_term_months.eq(360) & df.is_fixed_rate.eq(1)),
        ("Known county", df.county.ne("Unknown")),
        ("Income, property value, LTV and DTI all reported",
         df[["income_k", "property_value", "ltv", "dti", "loan_amount"]].notna().all(axis=1)),
        ("Rate spread reported and within -2 to +5 points", df.rate_spread.between(-2, 5)),
    ])


def preprocessor(target_type: str, scale: bool = True, lender: bool = True) -> ColumnTransformer:
    """One-hot categories (rare levels pooled), optional scaling, cross-fitted lender target encoding.

    sklearn's TargetEncoder uses internal cross-fitting in fit_transform, so a training row's
    lender encoding never includes that row's own outcome.
    """
    numeric = Pipeline([("scale", StandardScaler())]) if scale else "passthrough"
    transformers = [
        ("num", numeric, NUMERIC),
        ("flag", "passthrough", FLAGS),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50, sparse_output=False,
                              drop=[REFERENCE[col] for col in CATEGORICAL] if scale else None), CATEGORICAL),
    ]
    if lender:
        encoder = TargetEncoder(target_type=target_type, random_state=config.RANDOM_STATE)
        # Scale the encoded lender column too, so linear coefficients are per standard deviation like the other numerics.
        lender_step = Pipeline([("encode", encoder), ("scale", StandardScaler())]) if scale else encoder
        transformers.append(("lender", lender_step, LENDER))
    return ColumnTransformer(transformers, verbose_feature_names_out=True)


LABELS = {
    "dti": "Debt-to-income", "lender_id": "Lender", "loan_type": "Loan type", "ltv": "Loan-to-value",
    "aus": "Underwriting system", "log_loan_amount": "Loan amount", "log_income": "Income", "county": "County",
    "occupancy": "Occupancy", "credit_score_model": "Credit score model", "conforming": "Jumbo vs conforming",
    "tract_minority_pct": "Tract minority share", "tract_income_pct": "Tract income level", "units": "Number of units",
    "business_purpose": "Business purpose", "manufactured": "Manufactured home",
}


def clean_feature_name(name: str) -> str:
    """'cat__loan_type_FHA' -> 'loan_type = FHA', 'num__ltv' -> 'ltv'."""
    kind, _, rest = name.partition("__")
    if kind == "cat":
        for col in CATEGORICAL:
            if rest.startswith(col + "_"):
                return f"{col} = {rest[len(col) + 1:]}"
    if kind == "lender":
        return "lender (target-encoded)"
    return rest

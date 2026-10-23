"""Cleaning: raw HMDA text -> typed, labelled application table (data/processed/*.parquet).

All rows are kept here; columns get proper types and readable labels, and implausible values
become missing. Which rows each model uses is decided in features.py, so the EDA can still
describe the full population.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config
from .ingest import raw_path

log = logging.getLogger(__name__)
MISSING_CODES = {"NA", "Exempt", "", "1111", "8888", "9999"}


def to_number(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.where(~s.isin(MISSING_CODES)), errors="coerce")


def parse_dti(s: pd.Series) -> pd.Series:
    """'<20%' -> 15, '30%-<36%' -> 33, '44' -> 44, 'NA' -> NaN."""
    mapped = s.map(config.DTI_BANDS)
    return mapped.fillna(to_number(s))


def label(s: pd.Series, mapping: dict[str, str], other: str = "Other", missing: str = "Not reported") -> pd.Series:
    out = s.map(mapping)
    out = out.where(out.notna(), np.where(s.isin(MISSING_CODES), missing, other))
    return out.astype("string")


def race_ethnicity(df: pd.DataFrame) -> pd.Series:
    """Combined group used only for the fairness audit, never as a model feature."""
    race = df["derived_race"].replace({
        "Black or African American": "Black",
        "Race Not Available": "Not available",
        "American Indian or Alaska Native": "Other or multiple",
        "Native Hawaiian or Other Pacific Islander": "Other or multiple",
        "2 or more minority races": "Other or multiple",
        "Free Form Text Only": "Other or multiple",
    })
    hispanic = df["derived_ethnicity"].eq("Hispanic or Latino")
    return race.where(~hispanic, "Hispanic or Latino").astype("string")


def clean_year(year: int) -> pd.DataFrame:
    raw = pd.read_csv(raw_path(year), dtype=str, keep_default_na=False)
    df = pd.DataFrame({
        "year": year,
        "lender_id": raw["lei"],
        "county_fips": raw["county_code"],
        "county": raw["county_code"].map(config.COUNTIES).fillna("Unknown"),
        "action": raw["action_taken"].map({"1": "Originated", "2": "Approved, not accepted", "3": "Denied"}),
        "denied": raw["action_taken"].eq("3").astype(int),
        "loan_type": label(raw["loan_type"], config.LOAN_TYPES),
        "lien": label(raw["lien_status"], config.LIEN),
        "occupancy": label(raw["occupancy_type"], config.OCCUPANCY),
        "aus": label(raw["aus-1"], config.AUS, missing="Exempt"),
        "credit_score_model": label(raw["applicant_credit_score_type"], config.CREDIT_SCORE_MODEL,
                                    other="Other or newer model", missing="Exempt"),
        "conforming": raw["conforming_loan_limit"].map({"C": "Conforming", "NC": "Jumbo"}).fillna("Not reported"),
        "business_purpose": raw["business_or_commercial_purpose"].eq("1").astype(int),
        "open_end": raw["open-end_line_of_credit"].eq("1").astype(int),
        "reverse_mortgage": raw["reverse_mortgage"].eq("1").astype(int),
        "manufactured": raw["construction_method"].eq("2").astype(int),
        "units": to_number(raw["total_units"]).fillna(5).clip(upper=5),  # "5-24", ">149"... -> 5+
        "is_exempt": raw["aus-1"].eq("1111").astype(int),
        "loan_amount": to_number(raw["loan_amount"]),
        "property_value": to_number(raw["property_value"]),
        "income_k": to_number(raw["income"]),
        "dti": parse_dti(raw["debt_to_income_ratio"]),
        "ltv": to_number(raw["loan_to_value_ratio"]),
        "loan_term_months": to_number(raw["loan_term"]),
        "is_fixed_rate": raw["intro_rate_period"].isin(["NA"]).astype(int),
        "interest_rate": to_number(raw["interest_rate"]),
        "rate_spread": to_number(raw["rate_spread"]),
        "denial_reason": raw["denial_reason-1"].map(config.DENIAL_REASONS),
        "race_ethnicity": race_ethnicity(raw),
        "sex": raw["derived_sex"].replace({"Sex Not Available": "Not available"}).astype("string"),
        "age_band": raw["applicant_age"].replace({"8888": "Not available"}).astype("string"),
        "tract_minority_pct": to_number(raw["tract_minority_population_percent"]),
        "tract_income_pct": to_number(raw["tract_to_msa_income_percentage"]),
    })

    # Implausible values become missing (documented rules, counted below).
    rules = {
        "income_k": (df.income_k <= 0) | (df.income_k > 10_000),          # <= $0 or > $10M a year
        "ltv": (df.ltv <= 0) | (df.ltv > 150),
        "dti": (df.dti < 0) | (df.dti > 100),
        "property_value": df.property_value < 10_000,
        "interest_rate": (df.interest_rate < 1) | (df.interest_rate > 20),
    }
    for col, bad in rules.items():
        if bad.any():
            log.info("%d: %s implausible values in %s set to missing", year, f"{int(bad.sum()):,}", col)
        df.loc[bad, col] = np.nan
    return df


STRING_COLUMNS = ["lender_id", "county_fips", "county", "action", "loan_type", "lien", "occupancy", "aus",
                  "credit_score_model", "conforming", "denial_reason", "race_ethnicity", "sex", "age_band"]


def processed_path(year: int):
    # Compressed CSV rather than Parquet: portable, opens in Tableau and Excel, no extra binary deps.
    return config.PROCESSED_DIR / f"applications_{year}.csv.gz"


def clean(years: list[int] = config.YEARS) -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for year in years:
        df = clean_year(year)
        df.to_csv(processed_path(year), index=False)
        log.info("Wrote %s (%s rows, denial rate %.1f%%)", processed_path(year).name, f"{len(df):,}", 100 * df.denied.mean())


def load(year: int) -> pd.DataFrame:
    return pd.read_csv(processed_path(year), dtype={c: "string" for c in STRING_COLUMNS})

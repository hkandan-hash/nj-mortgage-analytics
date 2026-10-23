"""Exports: Tableau-ready CSVs (exports/tableau/) and the live dashboard's data (docs/data.js)."""
from __future__ import annotations

import json
import logging
import urllib.request

import numpy as np
import pandas as pd

from . import config
from .clean import load
from .eda import DTI_ORDER, LTV_ORDER, dti_band, ltv_band
from .features import regression_scope

log = logging.getLogger(__name__)
COUNTY_GEOJSON_URL = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"


def county_shapes() -> dict:
    """New Jersey county boundaries (US Census-derived GeoJSON), simplified to 3 decimals."""
    cache = config.RAW_DIR / "reference" / "us_counties.geojson"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(COUNTY_GEOJSON_URL, cache)
    geo = json.loads(cache.read_text(encoding="latin-1"))

    def simplify(coords):
        if isinstance(coords[0], (int, float)):
            return [round(coords[0], 3), round(coords[1], 3)]
        return [simplify(c) for c in coords]

    features = [
        {"type": "Feature", "id": f["id"], "properties": {"name": config.COUNTIES[f["id"]]},
         "geometry": {"type": f["geometry"]["type"], "coordinates": simplify(f["geometry"]["coordinates"])}}
        for f in geo["features"] if f["id"] in config.COUNTIES
    ]
    if len(features) != len(config.COUNTIES):
        raise ValueError(f"Expected {len(config.COUNTIES)} NJ counties, found {len(features)}")
    return {"type": "FeatureCollection", "features": features}


def records(df: pd.DataFrame, digits: int = 4) -> list[dict]:
    return json.loads(df.round(digits).replace({np.nan: None}).to_json(orient="records"))


def run_export() -> None:
    config.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    frames = {y: load(y) for y in config.YEARS}
    both = pd.concat(frames.values(), ignore_index=True)
    reg_metrics = json.loads((config.REPORT_DIR / "metrics" / "regression.json").read_text())
    cls_metrics = json.loads((config.REPORT_DIR / "metrics" / "classification.json").read_text())
    scored = pd.read_csv(config.PROCESSED_DIR / f"classification_scored_{config.TEST_YEAR}.csv.gz")
    reg_pred = pd.read_csv(config.PROCESSED_DIR / f"regression_predictions_{config.TEST_YEAR}.csv.gz")

    # ---------------------------------------------------------------- aggregates
    spreads = pd.concat([regression_scope(frames[y])[0] for y in config.YEARS])
    county = both[both.county != "Unknown"].groupby(["year", "county_fips", "county"]).agg(
        applications=("denied", "size"), denial_rate=("denied", "mean"), median_loan=("loan_amount", "median"),
        median_income_k=("income_k", "median"), median_rate=("interest_rate", "median")).reset_index()
    county = county.merge(spreads.groupby(["year", "county"]).rate_spread.mean().rename("avg_rate_spread").reset_index(),
                          on=["year", "county"], how="left")
    model_view = scored.groupby("county").agg(avg_predicted_denial=("denial_probability", "mean"),
                                              flagged_rate=("flagged", "mean")).reset_index().assign(year=config.TEST_YEAR)
    county = county.merge(model_view, on=["year", "county"], how="left")

    grid = both.assign(dti_band=dti_band(both.dti).astype(str), ltv_band=ltv_band(both.ltv).astype(str)) \
        .query("dti_band != 'nan' and ltv_band != 'nan'") \
        .groupby(["year", "dti_band", "ltv_band"]).denied.agg(applications="size", denial_rate="mean").reset_index()
    reasons = both[both.denied == 1].assign(denial_reason=lambda d: d.denial_reason.fillna("Not reported")) \
        .groupby(["year", "denial_reason"]).size().rename("denials").reset_index()
    reasons["share"] = reasons.denials / reasons.groupby("year").denials.transform("sum")
    by_type = both.groupby(["year", "loan_type"]).denied.agg(applications="size", denial_rate="mean").reset_index()
    kpis = both.groupby("year").agg(applications=("denied", "size"), denial_rate=("denied", "mean"),
                                    median_loan=("loan_amount", "median"), median_income_k=("income_k", "median"),
                                    median_rate=("interest_rate", "median")).reset_index()

    model_rows = [{"task": "Regression (rate spread)", **m} for m in reg_metrics["models"]] + \
                 [{"task": "Classification (denial)", **m} for m in cls_metrics["models"]]
    importance = [{"task": "Regression (rate spread)", **r} for r in reg_metrics["importance"]] + \
                 [{"task": "Classification (denial)", **r} for r in cls_metrics["importance"]]
    fairness = [{"attribute": attr, **r} for attr, rows in cls_metrics["fairness"].items() for r in rows]
    roc = [{"model": m, "fpr": f, "tpr": t} for m, v in cls_metrics["roc"].items() for f, t in zip(v["fpr"], v["tpr"])]

    # ---------------------------------------------------------------- Tableau extracts
    keep = ["year", "county_fips", "county", "action", "denied", "denial_reason", "loan_type", "occupancy", "aus",
            "conforming", "loan_amount", "property_value", "income_k", "ltv", "dti", "race_ethnicity", "sex", "age_band",
            "denial_probability", "flagged"]
    tableau = {
        f"applications_{config.TEST_YEAR}_scored.csv": scored[keep],
        "county_summary.csv": county,
        "denial_by_dti_ltv.csv": grid,
        "denial_reasons.csv": reasons,
        "denial_by_loan_type.csv": by_type,
        "model_metrics.csv": pd.DataFrame(model_rows),
        "feature_importance.csv": pd.DataFrame(importance),
        "fairness_audit.csv": pd.DataFrame(fairness),
        "pricing_effects.csv": pd.DataFrame(reg_metrics["effects"]),
        "pricing_gaps.csv": pd.DataFrame(reg_metrics["group_gaps"]),
        "roc_curves.csv": pd.DataFrame(roc),
    }
    for name, df in tableau.items():
        df.to_csv(config.EXPORT_DIR / name, index=False)
        log.info("Exported %s (%s rows)", name, f"{len(df):,}")

    # ---------------------------------------------------------------- dashboard data
    data = {
        "years": config.YEARS, "trainYear": config.TRAIN_YEAR, "testYear": config.TEST_YEAR,
        "kpis": records(kpis), "counties": records(county), "grid": records(grid), "dtiOrder": DTI_ORDER, "ltvOrder": LTV_ORDER,
        "reasons": records(reasons), "byLoanType": records(by_type),
        "regression": {k: reg_metrics[k] for k in ["best_model", "models", "effects", "importance", "group_gaps",
                                                    "ols_r2_with_lender", "ols_r2_without_lender", "train_rows", "test_rows"]},
        "classification": {k: cls_metrics[k] for k in ["best_model", "models", "importance", "confusion", "roc",
                                                         "calibration", "fairness", "train_rows", "test_rows"]},
        "geo": county_shapes(),
    }
    config.DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, default=float, separators=(",", ":")).replace("NaN", "null")
    (config.DASHBOARD_DIR / "data.js").write_text("window.DASHBOARD_DATA = " + payload + ";\n", encoding="utf-8")
    log.info("Wrote docs/data.js (%.0f KB)", len(payload) / 1024)

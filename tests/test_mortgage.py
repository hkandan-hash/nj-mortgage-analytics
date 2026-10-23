import numpy as np
import pandas as pd
import pytest

from mortgage import classification, features, regression
from mortgage.clean import label, parse_dti, race_ethnicity, to_number


def test_parse_dti_maps_bands_to_midpoints_and_keeps_exact_values():
    s = pd.Series(["<20%", "20%-<30%", "44", "50%-60%", ">60%", "NA", "Exempt"])
    out = parse_dti(s)
    assert out.tolist()[:5] == [15.0, 25.0, 44.0, 55.0, 65.0]
    assert out.iloc[5:].isna().all()


def test_to_number_treats_hmda_missing_codes_as_missing():
    out = to_number(pd.Series(["1111", "8888", "NA", "Exempt", "250"]))
    assert out.isna().tolist() == [True, True, True, True, False]


def test_label_distinguishes_missing_from_unknown_codes():
    out = label(pd.Series(["1", "1111", "42"]), {"1": "Conventional"})
    assert out.tolist() == ["Conventional", "Not reported", "Other"]


def test_hispanic_ethnicity_overrides_race_for_the_audit_group():
    df = pd.DataFrame({"derived_race": ["White", "White", "Black or African American"],
                       "derived_ethnicity": ["Hispanic or Latino", "Not Hispanic or Latino", "Not Hispanic or Latino"]})
    assert race_ethnicity(df).tolist() == ["Hispanic or Latino", "White", "Black"]


@pytest.mark.parametrize("module", [regression, classification])
def test_models_never_see_protected_attributes(module):
    assert not set(module.FEATURES) & set(features.PROTECTED)


def test_denial_model_excludes_post_decision_fields():
    leaky = {"interest_rate", "rate_spread", "denial_reason", "action", "denied"}
    assert not set(classification.FEATURES) & leaky


def make_apps(n=6):
    base = dict(year=2024, lender_id="L1", county_fips="34003", county="Bergen", denied=0, loan_type="Conventional",
                lien="First lien", occupancy="Principal residence", aus="Desktop Underwriter",
                credit_score_model="Equifax Beacon 5.0", conforming="Conforming", business_purpose=0, open_end=0,
                reverse_mortgage=0, manufactured=0, units=1.0, is_exempt=0, loan_amount=400_000.0, property_value=500_000.0,
                income_k=150.0, dti=40.0, ltv=80.0, loan_term_months=360.0, is_fixed_rate=1, interest_rate=6.5,
                rate_spread=0.3, action="Originated", tract_minority_pct=30.0, tract_income_pct=110.0)
    return pd.DataFrame([base] * n)


def test_classification_scope_drops_incomplete_and_exempt_rows_and_counts_them():
    df = make_apps(6)
    df.loc[0, "dti"] = np.nan
    df.loc[1, "is_exempt"] = 1
    df.loc[2, "county"] = "Unknown"
    scoped, funnel = features.classification_scope(df)
    assert len(scoped) == 3
    assert funnel[0]["rows"] == 6 and funnel[-1]["rows"] == 3
    assert sum(step["removed"] for step in funnel) == 3


def test_regression_scope_keeps_only_originated_30_year_fixed_with_plausible_spread():
    df = make_apps(5)
    df.loc[0, "action"] = "Denied"
    df.loc[1, "loan_term_months"] = 180.0
    df.loc[2, "is_fixed_rate"] = 0
    df.loc[3, "rate_spread"] = 7.5
    scoped, _ = features.regression_scope(df)
    assert len(scoped) == 1


def test_add_features_logs_amounts():
    out = features.add_features(make_apps(1))
    assert out.log_income.iloc[0] == pytest.approx(np.log(150_000))
    assert out.log_loan_amount.iloc[0] == pytest.approx(np.log(400_000))


def test_best_f1_threshold_separates_perfect_scores():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    t = classification.best_f1_threshold(y, p)
    assert 0.3 < t <= 0.8
    assert classification.scores(y, p, t)["F1"] == 1.0


def test_regression_metrics_on_perfect_predictions():
    y = np.array([0.1, 0.5, 0.9])
    m = regression.metrics(y, y)
    assert m["MAE"] == 0 and m["RMSE"] == 0 and m["R2"] == 1


def test_preprocessor_uses_reference_levels_for_linear_models():
    # Categories need >= 50 rows (min_frequency) to stay distinct, so use a realistic count.
    df = features.add_features(make_apps(120))
    df.loc[:59, "loan_type"] = "FHA"
    prep = features.preprocessor("continuous", scale=True, lender=False)
    prep.fit(df[features.NUMERIC + features.FLAGS + features.CATEGORICAL])
    names = [features.clean_feature_name(n) for n in prep.get_feature_names_out()]
    assert "loan_type = Conventional" not in names  # reference level is dropped
    assert "loan_type = FHA" in names

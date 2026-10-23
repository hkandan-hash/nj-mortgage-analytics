"""Regression: what drives the rate spread a borrower pays above the market benchmark?

Target: rate_spread (APR minus the average prime offer rate for a comparable loan on the lock
date). Train on 2023 originations, test on 2024 (out-of-time). Models: mean baseline, OLS with
robust standard errors, Ridge, Lasso, median (quantile) regression, and gradient boosting.
Writes reports/02_regression.md, figures, and reports/metrics/regression.json.
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from . import charts as c
from . import config
from .clean import load
from .eda import md_table
from .features import CATEGORICAL, FLAGS, LABELS, LENDER, NUMERIC, clean_feature_name, preprocessor, regression_scope

log = logging.getLogger(__name__)
TARGET = "rate_spread"
FEATURES = NUMERIC + FLAGS + CATEGORICAL + LENDER


def metrics(y, pred) -> dict:
    return {"MAE": float(mean_absolute_error(y, pred)), "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
            "R2": float(r2_score(y, pred))}


class StatsmodelsOLS:
    """OLS with heteroskedasticity-robust (HC3) standard errors, on the shared design matrix."""

    def __init__(self, lender: bool = True):
        self.prep = preprocessor("continuous", scale=True, lender=lender)

    def fit(self, X, y):
        Z = self.prep.fit_transform(X, y)
        self.names = ["const"] + [clean_feature_name(n) for n in self.prep.get_feature_names_out()]
        self.result = sm.OLS(np.asarray(y), sm.add_constant(Z, has_constant="add")).fit(cov_type="HC3")
        return self

    def predict(self, X):
        return self.result.predict(sm.add_constant(self.prep.transform(X), has_constant="add"))


class StatsmodelsQuantile:
    """Median regression: less sensitive to the long tail of high spreads than OLS."""

    def __init__(self):
        self.prep = preprocessor("continuous", scale=True)

    def fit(self, X, y):
        Z = self.prep.fit_transform(X, y)
        self.result = sm.QuantReg(np.asarray(y), sm.add_constant(Z, has_constant="add")).fit(q=0.5, max_iter=5000)
        return self

    def predict(self, X):
        return self.result.predict(sm.add_constant(self.prep.transform(X), has_constant="add"))


def run_regression() -> None:
    train, train_funnel = regression_scope(load(config.TRAIN_YEAR))
    test, test_funnel = regression_scope(load(config.TEST_YEAR))
    X_tr, y_tr = train[FEATURES], train[TARGET]
    X_te, y_te = test[FEATURES], test[TARGET]
    log.info("Regression: train %s rows (%d), test %s rows (%d)", f"{len(train):,}", config.TRAIN_YEAR, f"{len(test):,}", config.TEST_YEAR)

    models = {
        "OLS": StatsmodelsOLS(),
        "OLS without lender": StatsmodelsOLS(lender=False),
        "Ridge": Pipeline([("prep", preprocessor("continuous")), ("model", RidgeCV(alphas=np.logspace(-2, 3, 20)))]),
        "Lasso": Pipeline([("prep", preprocessor("continuous")),
                           ("model", LassoCV(cv=5, alphas=30, random_state=config.RANDOM_STATE, max_iter=5000))]),
        "Median (quantile) regression": StatsmodelsQuantile(),
        "Gradient boosting": Pipeline([("prep", preprocessor("continuous", scale=False)),
                                       ("model", HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                                                               early_stopping=True, random_state=config.RANDOM_STATE))]),
    }
    results, preds = [], {}
    baseline = np.full(len(y_te), y_tr.mean())
    results.append({"model": "Baseline (training mean)", **metrics(y_te, baseline), "fit_seconds": 0.0})
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X_tr, y_tr)
        preds[name] = np.asarray(model.predict(X_te))
        results.append({"model": name, **metrics(y_te, preds[name]), "fit_seconds": round(time.perf_counter() - start, 1)})
        log.info("%s: %s", name, {k: round(v, 3) for k, v in results[-1].items() if k != "model"})
    res = pd.DataFrame(results)
    best_name = res.loc[res.R2.idxmax(), "model"]
    ols = models["OLS"]
    lasso = models["Lasso"].named_steps["model"]

    # ---------------------------------------------------------------- interpretable effects (OLS)
    prep = ols.prep
    scaler = prep.named_transformers_["num"].named_steps["scale"]
    coef = pd.Series(ols.result.params, index=ols.names)
    se = pd.Series(ols.result.bse, index=ols.names)
    sd = pd.Series(scaler.scale_, index=NUMERIC)
    effects = []  # effect in percentage points of rate spread, for a natural change in each input
    for feat, step, label in [("ltv", 10, "+10 points loan-to-value"), ("dti", 10, "+10 points debt-to-income"),
                              ("log_income", np.log(2), "Income doubles"), ("log_loan_amount", np.log(2), "Loan amount doubles"),
                              ("tract_minority_pct", 10, "+10 points tract minority share"),
                              ("tract_income_pct", 25, "+25 points tract income (% of area median)")]:
        k = step / sd[feat]
        effects.append({"change": label, "effect": coef[feat] * k, "ci": 1.96 * se[feat] * k})
    for name, label in [("loan_type = FHA", "FHA (vs conventional)"), ("loan_type = VA", "VA (vs conventional)"),
                        ("occupancy = Investment property", "Investment property (vs own home)"),
                        ("occupancy = Second residence", "Second home (vs own home)"),
                        ("conforming = Jumbo", "Jumbo (vs conforming)"),
                        ("aus = No AUS used", "No automated underwriting (vs DU)"),
                        ("aus = Loan Product Advisor", "Freddie Mac LPA (vs DU)")]:
        if name in coef:
            effects.append({"change": label, "effect": coef[name], "ci": 1.96 * se[name]})
    eff = pd.DataFrame(effects)
    top_up = eff.loc[eff.effect.idxmax()]
    top_down = eff.loc[eff.effect.idxmin()]

    fig, ax = c.figure(9, 5.2)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=c.GRID, linewidth=0.8)
    e = eff.iloc[::-1]
    colors = [c.RED if v > 0 else c.BLUE for v in e.effect]
    ax.barh(e.change, e.effect, xerr=e.ci, color=colors, height=0.6, error_kw={"ecolor": c.INK_2, "elinewidth": 1, "capsize": 2})
    ax.axvline(0, color=c.AXIS, linewidth=1)
    ax.tick_params(axis="y", labelsize=8.5, colors=c.INK_2)
    ax.set_xlabel("Change in rate spread (percentage points), 95% interval", color=c.MUTED, fontsize=9)
    c.headline(fig, f"{top_up.change.split(' (')[0]} adds {top_up.effect:.2f} points; {top_down.change.split(' (')[0]} saves {abs(top_down.effect):.2f}",
               f"OLS estimates holding everything else equal (incl. lender and county), trained on {config.TRAIN_YEAR}. Red raises the spread, blue lowers it.")
    fig_effects = c.save(fig, "reg_effects.png", top=0.84)

    # ---------------------------------------------------------------- model comparison
    fig, ax = c.figure(9, 4.2)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=c.GRID, linewidth=0.8)
    r = res.sort_values("R2")
    bar_colors = [c.ORANGE if m == best_name else (c.MUTED if m.startswith("Baseline") else c.BLUE) for m in r.model]
    ax.barh(r.model, r.R2.clip(lower=0), color=bar_colors, height=0.6)
    for i, (r2, rmse) in enumerate(zip(r.R2, r.RMSE)):
        ax.text(max(r2, 0) + 0.005, i, f"RÂ² {r2:.3f} Â· RMSE {rmse:.3f}", va="center", fontsize=8.5, color=c.INK_2)
    ax.set_xlim(0, max(r.R2.max() * 1.45, 0.1))
    ax.tick_params(axis="y", labelsize=9, colors=c.INK_2)
    ax.set_xticks([])
    c.headline(fig, f"{best_name} explains {res.R2.max():.0%} of pricing variation in the following year",
               f"Out-of-time test: trained on {config.TRAIN_YEAR}, scored on {len(test):,} loans from {config.TEST_YEAR}. Higher RÂ² is better.")
    fig_models = c.save(fig, "reg_model_comparison.png")

    # ---------------------------------------------------------------- permutation importance (best model)
    best = models[best_name] if best_name in models else models["Gradient boosting"]
    sample = test.sample(min(10_000, len(test)), random_state=config.RANDOM_STATE)
    perm = permutation_importance(best, sample[FEATURES], sample[TARGET], n_repeats=5,
                                  random_state=config.RANDOM_STATE, scoring="r2", n_jobs=-1)
    imp = pd.DataFrame({"feature": [LABELS[f] for f in FEATURES], "importance": perm.importances_mean, "std": perm.importances_std}) \
        .sort_values("importance", ascending=False)
    fig, ax = c.figure(9, 4.6)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=c.GRID, linewidth=0.8)
    top = imp.head(10).iloc[::-1]
    ax.barh(top.feature, top.importance, xerr=top["std"], color=c.BLUE, height=0.6,
            error_kw={"ecolor": c.INK_2, "elinewidth": 1})
    ax.tick_params(axis="y", labelsize=9, colors=c.INK_2)
    ax.set_xlabel("Drop in test RÂ² when the feature is shuffled", color=c.MUTED, fontsize=9)
    c.headline(fig, f"Which lender you use matters most for the price you pay",
               f"Permutation importance, {best_name}, {config.TEST_YEAR} test sample of 10,000 loans")
    fig_importance = c.save(fig, "reg_importance.png")

    # ---------------------------------------------------------------- pricing gaps by group
    test = test.assign(predicted=preds[best_name], residual=test[TARGET] - preds[best_name])
    groups = test.groupby("race_ethnicity").agg(loans=(TARGET, "size"), actual=(TARGET, "mean"),
                                                predicted=("predicted", "mean"), residual=("residual", "mean"))
    groups = groups[groups.loans >= 200].sort_values("loans", ascending=False)
    groups["residual_se"] = test.groupby("race_ethnicity").residual.sem().reindex(groups.index)
    ref = groups.loc["White"]
    groups["raw_gap_vs_white_bps"] = (groups.actual - ref.actual) * 100
    groups["unexplained_gap_vs_white_bps"] = (groups.residual - ref.residual) * 100
    significant = groups[(groups.index != "White") & (groups.residual.abs() > 1.96 * groups.residual_se) & (groups.loans >= 1000)]
    largest = groups.drop("White").loc[lambda d: d.loans >= 1000].raw_gap_vs_white_bps.idxmax()
    fig, ax = c.figure(9, 4.2)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=c.GRID, linewidth=0.8)
    g = groups.iloc[::-1]
    ax.barh(g.index, g.residual * 100, xerr=1.96 * g.residual_se * 100, color=c.BLUE, height=0.6,
            error_kw={"ecolor": c.INK_2, "elinewidth": 1, "capsize": 2})
    ax.axvline(0, color=c.AXIS, linewidth=1)
    ax.tick_params(axis="y", labelsize=9, colors=c.INK_2)
    ax.set_xlabel("Actual minus predicted rate spread (basis points), 95% interval", color=c.MUTED, fontsize=9)
    g_l = groups.loc[largest]
    c.headline(fig, f"Most of the {g_l.raw_gap_vs_white_bps:.0f} bp raw gap for {largest} borrowers is explained; "
                    f"{g_l.unexplained_gap_vs_white_bps:.0f} bp is not",
               "Average unexplained spread (actual minus predicted) by applicant group. The model never sees race or ethnicity.")
    fig_gaps = c.save(fig, "reg_group_gaps.png")

    # ---------------------------------------------------------------- report
    n_zero = int((lasso.coef_ == 0).sum())
    no_lender_r2 = res.set_index("model").loc["OLS without lender", "R2"]
    ols_r2 = res.set_index("model").loc["OLS", "R2"]
    table = res.assign(MAE=res.MAE.map("{:.3f}".format), RMSE=res.RMSE.map("{:.3f}".format), R2=res.R2.map("{:.3f}".format))
    eff_table = eff.assign(effect=eff.effect.map("{:+.3f}".format), ci=eff.ci.map("Â±{:.3f}".format)) \
                   .rename(columns={"effect": "effect (pct. points)", "ci": "95% interval"})
    gap_table = pd.DataFrame({
        "group": groups.index, "loans": groups.loans.map("{:,}".format),
        "actual spread": groups.actual.map("{:.3f}".format), "predicted": groups.predicted.map("{:.3f}".format),
        "raw gap vs White (bps)": groups.raw_gap_vs_white_bps.map("{:+.1f}".format),
        "unexplained (bps)": (groups.residual * 100).map("{:+.1f}".format),
        "95% interval (bps)": (1.96 * groups.residual_se * 100).map("Â±{:.1f}".format),
    })
    gap_summary = (
        "Groups with an unexplained spread significantly different from zero (95% level, at least 1,000 loans): "
        + (", ".join(f"{g} ({r.residual * 100:+.1f} bps)" for g, r in significant.iterrows()) or "none") + "."
    )
    funnel = pd.DataFrame(test_funnel).assign(rows=lambda d: d.rows.map("{:,}".format), removed=lambda d: d.removed.map("{:,}".format))
    lines = [
        "# 2. Regression: what drives mortgage pricing?", "",
        f"**Question.** Among borrowers who got a loan, what explains how much they paid above the market rate?", "",
        f"**Target.** `rate_spread`: the loan's APR minus the average prime offer rate for a comparable loan on the rate-lock date, in percentage points. "
        f"Mean {test[TARGET].mean():.3f}, standard deviation {test[TARGET].std():.3f} in {config.TEST_YEAR}.", "",
        f"**Design.** Fit on {len(train):,} loans from {config.TRAIN_YEAR}; evaluated on {len(test):,} loans from {config.TEST_YEAR} that the models never saw. "
        "Features are loan, borrower, property, county and lender characteristics. Race, ethnicity, sex and age are excluded.", "",
        "## Sample", "", md_table(funnel), "",
        "## Model comparison", "", f"![Model comparison]({fig_models})", "", md_table(table), "",
        f"- **{best_name}** performs best (RÂ² {res.R2.max():.3f} on next year's loans); the linear models trail it, so pricing has non-linear patterns and interactions.",
        f"- Lasso kept {len(lasso.coef_) - n_zero} of {len(lasso.coef_)} features (alpha {lasso.alpha_:.4f}), and Ridge and Lasso score almost the same as OLS. "
        "With this many observations, regularization adds little, which suggests OLS is not overfitting.",
        f"- **Lender identity explains a large share of pricing.** OLS reaches RÂ² {ols_r2:.3f} with the lender feature and {no_lender_r2:.3f} without it. "
        "Two similar borrowers can pay noticeably different prices depending on where they apply.", "",
        "## What moves the price", "", f"![Effects]({fig_effects})", "", md_table(eff_table), "",
        "Effects are from OLS with heteroskedasticity-robust (HC3) standard errors, holding all other features (including lender and county) fixed. "
        "Associations, not causal effects: HMDA has no credit scores, which drive pricing and correlate with several of these features.", "",
        "Two rows need care when reading:",
        "- *Loan amount doubles* lowers the spread largely for a mechanical reason: rate spread is based on APR, which includes fixed closing "
        "costs, and those costs weigh more on a small loan.",
        "- *Income doubles* is estimated with debt-to-income held fixed, which means debts double too. It is not evidence that higher earners pay more.", "",
        "## Feature importance", "", f"![Permutation importance]({fig_importance})", "",
        "## Pricing gaps by group", "", f"![Group gaps]({fig_gaps})", "", md_table(gap_table), "",
        f"- {largest} borrowers paid {g_l.raw_gap_vs_white_bps:.0f} bps more than White borrowers on average; after accounting for loan, borrower "
        f"and lender characteristics, {g_l.unexplained_gap_vs_white_bps:.1f} bps of that difference remains unexplained.",
        f"- {gap_summary} On a $400,000 loan, 5 bps is roughly $200 a year in interest.", "",
        "Unexplained gap = average actual spread minus the model's prediction for that group. 1 basis point = 0.01 percentage points. "
        "Without credit scores this cannot separate unequal treatment from unmeasured credit differences; a gap near zero means "
        "the observed loan and lender characteristics account for the average pricing difference.", "",
    ]
    (config.REPORT_DIR / "02_regression.md").write_text("\n".join(lines), encoding="utf-8")

    out = config.REPORT_DIR / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    (out / "regression.json").write_text(json.dumps({
        "train_rows": len(train), "test_rows": len(test), "best_model": best_name,
        "models": results, "effects": eff.to_dict("records"), "importance": imp.to_dict("records"),
        "group_gaps": groups.reset_index().to_dict("records"),
        "ols_r2_with_lender": ols_r2, "ols_r2_without_lender": no_lender_r2,
    }, indent=2, default=float))
    test[["lender_id", "county", "loan_type", TARGET, "predicted", "residual"]].to_csv(
        config.PROCESSED_DIR / f"regression_predictions_{config.TEST_YEAR}.csv.gz", index=False)
    log.info("Wrote reports/02_regression.md (best: %s)", best_name)

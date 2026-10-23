"""Exploratory data analysis: writes reports/01_eda.md and its figures.

Covers both years (all applications with a credit decision), so the EDA describes the whole
population, not just the rows that later qualify for modeling.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import charts as c
from . import config
from .clean import load
from .features import classification_scope, regression_scope

log = logging.getLogger(__name__)

DTI_ORDER = ["<20%", "20-29%", "30-35%", "36-42%", "43-49%", "50-60%", ">60%"]
LTV_ORDER = ["≤60%", "60-80%", "80-90%", "90-95%", "95-97%", ">97%"]


def dti_band(dti: pd.Series) -> pd.Series:
    return pd.cut(dti, [-1, 19.9, 29.9, 35.9, 42.9, 49.9, 60, 101], labels=DTI_ORDER)


def ltv_band(ltv: pd.Series) -> pd.Series:
    return pd.cut(ltv, [0, 60, 80, 90, 95, 97, 151], labels=LTV_ORDER)


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(cols))) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def run_eda() -> None:
    frames = {y: load(y) for y in config.YEARS}
    both = pd.concat(frames.values(), ignore_index=True)
    test = frames[config.TEST_YEAR]
    lines = ["# 1. Exploratory data analysis", "",
             f"Home-purchase mortgage applications in New Jersey that reached a credit decision, {config.YEARS[0]}–{config.YEARS[-1]} "
             "(HMDA, via the FFIEC Data Browser API). Figures are for all such applications unless stated.", ""]

    # ---------------------------------------------------------------- overview
    overview = both.groupby("year").agg(
        applications=("denied", "size"), denial_rate=("denied", "mean"),
        median_loan=("loan_amount", "median"), median_income_k=("income_k", "median"),
        median_rate=("interest_rate", "median"),
    ).reset_index()
    overview["applications"] = overview.applications.map("{:,}".format)
    overview["denial_rate"] = overview.denial_rate.map(c.pct)
    overview["median_loan"] = overview.median_loan.map(lambda v: f"${v:,.0f}")
    overview["median_income_k"] = overview.median_income_k.map(lambda v: f"${v:,.0f}k")
    overview["median_rate"] = overview.median_rate.map(lambda v: f"{v:.3f}%")
    lines += ["## Overview", "", md_table(overview), ""]

    # ---------------------------------------------------------------- missingness
    fields = ["income_k", "property_value", "ltv", "dti", "loan_term_months", "interest_rate"]
    miss = pd.DataFrame({
        "field": fields,
        "missing (approved)": [c.pct(both.loc[both.denied == 0, f].isna().mean()) for f in fields],
        "missing (denied)": [c.pct(both.loc[both.denied == 1, f].isna().mean()) for f in fields],
    })
    lines += ["## Missing data", "",
              "Key fields are missing far more often on denied applications, largely because incomplete applications are denied. "
              "`interest_rate` exists only for approved loans, so it can never be used to predict denial (target leakage).", "",
              md_table(miss), ""]

    # ---------------------------------------------------------------- distributions
    fig, axes = c.figure(11, 3.6, ncols=4)
    panels = [("loan_amount", "Loan amount ($k)", 1e-3, (0, 1500)), ("income_k", "Income ($k)", 1, (0, 600)),
              ("ltv", "Loan-to-value (%)", 1, (0, 110)), ("dti", "Debt-to-income (%)", 1, (0, 70))]
    for ax, (col, label, scale, rng) in zip(axes, panels):
        vals = (test[col] * scale).dropna()
        ax.hist(vals.clip(*rng), bins=40, color=c.BLUE, edgecolor=c.SURFACE, linewidth=0.5)
        ax.set_title(label, fontsize=10, color=c.INK_2, loc="left")
        ax.set_yticks([])
    c.headline(fig, f"The most common loan-to-value is exactly 80%; the median loan is ${test.loan_amount.median() / 1000:,.0f}k",
               f"Distributions, {config.TEST_YEAR} applications. Values past the axis are grouped at the edge; DTI outside 36–49% "
               "is reported only in bands, shown at band midpoints.")
    lines += ["## Distributions", "", f"![Distributions]({c.save(fig, 'eda_distributions.png', top=0.78)})", "",
              f"- Loan-to-value clusters at exactly 80% ({c.pct((test.ltv.round() == 80).mean())} of {config.TEST_YEAR} applications), "
              "the point where private mortgage insurance is no longer required.",
              f"- Income is heavily right-skewed (mean ${test.income_k.mean():,.0f}k vs median ${test.income_k.median():,.0f}k), "
              "so models use log income.", ""]

    # ---------------------------------------------------------------- denial vs DTI / LTV
    t = test.assign(dti_band=dti_band(test.dti), ltv_band=ltv_band(test.ltv))
    by_dti = t.groupby("dti_band", observed=True).denied.agg(["mean", "size"])
    by_ltv = t.groupby("ltv_band", observed=True).denied.agg(["mean", "size"])
    fig, axes = c.figure(10, 4.2, ncols=2)
    for ax, data, label in ((axes[0], by_dti, "Debt-to-income"), (axes[1], by_ltv, "Loan-to-value")):
        ax.bar(range(len(data)), data["mean"], color=c.BLUE, width=0.62)
        ax.set_xticks(range(len(data)), data.index, fontsize=8.5)
        for i, v in enumerate(data["mean"]):
            ax.text(i, v + 0.004, c.pct(v, 0), ha="center", fontsize=8.5, color=c.INK_2)
        ax.set_yticks([])
        ax.set_title(label, fontsize=10, color=c.INK_2, loc="left")
    c.headline(fig, f"Denial rates climb steeply once debt-to-income passes 50%",
               f"Share of applications denied, {config.TEST_YEAR}")
    mix = t.assign(investor=t.occupancy.eq("Investment property"), no_aus=t.aus.eq("No AUS used")) \
           .groupby("dti_band", observed=True)[["investor", "no_aus"]].mean()
    mid = by_dti.loc[["20-29%", "30-35%", "36-42%", "43-49%"], "mean"]
    low_ltv = by_ltv.loc[["≤60%", "60-80%"], "mean"]
    mid_ltv = by_ltv.loc[["80-90%", "90-95%", "95-97%"], "mean"]
    lines += ["## What goes with denial", "", f"![Denial by DTI and LTV]({c.save(fig, 'eda_denial_dti_ltv.png', top=0.8)})", "",
              f"- Denial is lowest ({c.pct(mid.min(), 0)}–{c.pct(mid.max(), 0)}) for DTI between 20% and 49%, then jumps to "
              f"{c.pct(by_dti.loc['50-60%', 'mean'], 0)} at 50–60% and {c.pct(by_dti.loc['>60%', 'mean'], 0)} above 60%.",
              f"- Applicants with DTI under 20% are denied more often ({c.pct(by_dti['mean'].iloc[0], 0)}) than the middle bands. "
              f"They are a different mix: {c.pct(mix.loc['<20%', 'investor'], 0)} are investors and {c.pct(mix.loc['<20%', 'no_aus'], 0)} "
              f"were not run through automated underwriting, vs {c.pct(mix.loc['20-29%', 'investor'], 0)} and {c.pct(mix.loc['20-29%', 'no_aus'], 0)} "
              "in the 20–29% band. One-variable views mislead here; the models control for these together.",
              f"- Denial rises with loan-to-value: about {c.pct(low_ltv.mean(), 0)} at 80% or below, {c.pct(mid_ltv.min(), 0)}–{c.pct(mid_ltv.max(), 0)} "
              f"between 80% and 97%, and {c.pct(by_ltv['mean'].iloc[-1], 0)} above 97%.", ""]

    # ---------------------------------------------------------------- denial reasons
    reasons = test.loc[test.denied == 1, "denial_reason"].fillna("Not reported").value_counts(normalize=True)
    fig, ax = c.figure(9, 4.2)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=c.GRID, linewidth=0.8)
    r = reasons.sort_values()
    ax.barh(r.index, r.values, color=c.BLUE, height=0.62)
    for i, v in enumerate(r.values):
        ax.text(v + 0.004, i, c.pct(v, 0), va="center", fontsize=8.5, color=c.INK_2)
    ax.tick_params(axis="y", labelsize=9, colors=c.INK_2)
    ax.set_xticks([])
    c.headline(fig, f"Debt-to-income is the most common reason given for a denial",
               f"Primary denial reason, share of denied applications, {config.TEST_YEAR}")
    lines += ["## Why applications are denied", "", f"![Denial reasons]({c.save(fig, 'eda_denial_reasons.png')})", "",
              f"- The top three reasons ({', '.join(reasons.index[:3])}) cover {c.pct(reasons.iloc[:3].sum())} of denials.", ""]

    # ---------------------------------------------------------------- counties
    county = test[test.county != "Unknown"].groupby("county").denied.agg(["mean", "size"]).sort_values("mean")
    fig, ax = c.figure(9, 5.6)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=c.GRID, linewidth=0.8)
    ax.barh(county.index, county["mean"], color=c.BLUE, height=0.62)
    ax.axvline(test.denied.mean(), color=c.INK_2, linewidth=1, linestyle="--")
    ax.text(test.denied.mean(), len(county) - 0.4, f" state average {c.pct(test.denied.mean())}", fontsize=8.5, color=c.INK_2)
    for i, v in enumerate(county["mean"]):
        ax.text(v + 0.002, i, c.pct(v), va="center", fontsize=8, color=c.INK_2)
    ax.tick_params(axis="y", labelsize=8.5, colors=c.INK_2)
    ax.set_xticks([])
    c.headline(fig, f"Denial rates range from {c.pct(county['mean'].min(), 0)} in {county.index[0]} to {c.pct(county['mean'].max(), 0)} in {county.index[-1]}",
               f"Share of applications denied by county, {config.TEST_YEAR} (unadjusted for applicant differences)")
    lines += ["## Geography", "", f"![Denial by county]({c.save(fig, 'eda_county_denial.png', top=0.86)})", "",
              "These are raw rates. Counties differ in income, loan size and lender mix, so the models below test whether location still matters "
              "once those are accounted for.", ""]

    # ---------------------------------------------------------------- rate spread vs LTV
    reg, _ = regression_scope(test)
    reg = reg.assign(ltv_band=ltv_band(reg.ltv))
    spread = reg.groupby(["loan_type", "ltv_band"], observed=True).rate_spread.agg(["mean", "size"]).reset_index()
    spread = spread[spread["size"] >= 50]
    fig, ax = c.figure(9, 4.4)
    for color, lt in zip(c.SERIES, ["Conventional", "FHA", "VA"]):
        s = spread[spread.loan_type == lt]
        ax.plot([LTV_ORDER.index(b) for b in s.ltv_band], s["mean"], color=color, linewidth=2, marker="o", markersize=5, label=lt)
    ax.set_xticks(range(len(LTV_ORDER)), LTV_ORDER)
    ax.axhline(0, color=c.AXIS, linewidth=1)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_ylabel("Avg rate spread (points)", color=c.MUTED, fontsize=9)
    c.headline(fig, "Less equity means a higher price over market; FHA pays the most, VA prices below market",
               f"Average rate spread (APR minus the market benchmark on the lock date), originated 30-year fixed loans, {config.TEST_YEAR}")
    lines += ["## Pricing", "", f"![Rate spread by LTV]({c.save(fig, 'eda_rate_spread_ltv.png')})", "",
              "Rate spread removes the effect of market rates moving through the year, because HMDA does not publish application dates. "
              "That makes it the right target for comparing pricing across borrowers and years.", ""]

    # ---------------------------------------------------------------- correlations
    num_cols = ["log_loan_amount", "log_income", "ltv", "dti", "loan_to_income", "tract_minority_pct", "tract_income_pct"]
    cls, _ = classification_scope(test)
    corr = cls[num_cols + ["denied"]].corr()
    spread_corr = reg[num_cols + ["rate_spread"]].corr()["rate_spread"]
    fig, ax = c.figure(7.5, 6.2)
    ax.grid(False)
    full = corr.copy()
    full.loc["rate_spread"] = spread_corr.reindex(full.columns).fillna(np.nan)
    full["rate_spread"] = list(spread_corr.reindex(full.index[:-1]).values) + [1.0]
    im = ax.imshow(full.values, cmap=matplotlib_diverging(), vmin=-1, vmax=1)
    ax.set_xticks(range(len(full.columns)), full.columns, rotation=40, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(full.index)), full.index, fontsize=8.5)
    for i in range(len(full.index)):
        for j in range(len(full.columns)):
            v = full.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=c.INK if abs(v) < 0.6 else "white")
    c.headline(fig, "Loan size and income move together; no single number predicts denial",
               "Pearson correlations (denied: classification sample; rate_spread: regression sample)")
    lines += ["## Correlations", "", f"![Correlation matrix]({c.save(fig, 'eda_correlation.png', top=0.86)})", "",
              f"- The strongest single correlate of denial is DTI (r = {corr.loc['dti', 'denied']:.2f}); everything else is weaker. "
              "Denial depends on combinations of factors, which is why the classification models matter.",
              f"- `log_loan_amount` and `log_income` correlate at r = {corr.loc['log_loan_amount', 'log_income']:.2f}, "
              "so regularized models (Ridge, Lasso) are a sensible check on the plain OLS estimates.", ""]

    # ---------------------------------------------------------------- unadjusted group rates
    groups = test.groupby("race_ethnicity").denied.agg(["mean", "size"]).sort_values("size", ascending=False)
    groups = groups[groups["size"] >= 200]
    table = pd.DataFrame({"group": groups.index, "applications": groups["size"].map("{:,}".format),
                          "denial rate": groups["mean"].map(c.pct)})
    lines += ["## Unadjusted denial rates by group", "",
              "Reported for transparency and examined properly in the fairness audit (`03_classification.md`). These raw gaps do not "
              "account for income, debt, loan size or credit history, and HMDA does not include credit scores.", "",
              md_table(table), ""]

    (config.REPORT_DIR / "01_eda.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote reports/01_eda.md")


def matplotlib_diverging():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])

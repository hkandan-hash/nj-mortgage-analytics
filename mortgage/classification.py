"""Classification: can we predict which applications will be denied?

Train on 2023 applications (80% fit / 20% to choose the decision threshold), evaluate on 2024
(out-of-time). Models: prior baseline, logistic regression, decision tree, random forest, and
gradient boosting. Includes calibration, permutation importance, and a fairness audit.

Intended use: flag applications that look likely to be denied so a loan officer or housing
counselor can follow up early. Not to automate denials.
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score, log_loss,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text

from . import charts as c
from . import config
from .clean import load
from .eda import md_table
from .features import CATEGORICAL, FLAGS, LABELS, LENDER, NUMERIC, clean_feature_name, classification_scope, preprocessor

log = logging.getLogger(__name__)
TARGET = "denied"
FEATURES = NUMERIC + FLAGS + CATEGORICAL + LENDER


def best_f1_threshold(y, p) -> float:
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    return float(thresholds[np.argmax(f1[:-1])])


def scores(y, p, threshold) -> dict:
    flag = (p >= threshold).astype(int)
    return {
        "ROC AUC": float(roc_auc_score(y, p)), "PR AUC": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, p)), "Log loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        "Threshold": float(threshold), "Precision": float(precision_score(y, flag, zero_division=0)),
        "Recall": float(recall_score(y, flag)), "F1": float(f1_score(y, flag)), "Flagged": float(flag.mean()),
    }


def run_classification() -> None:
    train_all, _ = classification_scope(load(config.TRAIN_YEAR))
    test, test_funnel = classification_scope(load(config.TEST_YEAR))
    fit, valid = train_test_split(train_all, test_size=0.2, stratify=train_all[TARGET], random_state=config.RANDOM_STATE)
    log.info("Classification: fit %s, validation %s (%d); test %s (%d)", f"{len(fit):,}", f"{len(valid):,}",
             config.TRAIN_YEAR, f"{len(test):,}", config.TEST_YEAR)

    models = {
        "Logistic regression": Pipeline([("prep", preprocessor("binary")),
                                         ("model", LogisticRegression(C=1.0, max_iter=3000))]),
        "Decision tree": Pipeline([("prep", preprocessor("binary", scale=False)),
                                   ("model", DecisionTreeClassifier(max_depth=5, min_samples_leaf=200, random_state=config.RANDOM_STATE))]),
        "Random forest": Pipeline([("prep", preprocessor("binary", scale=False)),
                                   ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=20, max_features="sqrt",
                                                                    n_jobs=-1, random_state=config.RANDOM_STATE))]),
        "Gradient boosting": Pipeline([("prep", preprocessor("binary", scale=False)),
                                       ("model", HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, early_stopping=True,
                                                                                random_state=config.RANDOM_STATE))]),
    }
    prior = fit[TARGET].mean()
    rows = [{"model": "Baseline (training denial rate)", **scores(test[TARGET], np.full(len(test), prior), prior), "fit_seconds": 0.0}]
    test_probs, curves = {}, {}
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(fit[FEATURES], fit[TARGET])
        threshold = best_f1_threshold(valid[TARGET], model.predict_proba(valid[FEATURES])[:, 1])
        p = model.predict_proba(test[FEATURES])[:, 1]
        test_probs[name] = p
        rows.append({"model": name, **scores(test[TARGET], p, threshold), "fit_seconds": round(time.perf_counter() - start, 1)})
        fpr, tpr, _ = roc_curve(test[TARGET], p)
        prec, rec, _ = precision_recall_curve(test[TARGET], p)
        curves[name] = {"fpr": fpr, "tpr": tpr, "precision": prec, "recall": rec}
        log.info("%s: AUC %.3f, PR AUC %.3f, F1 %.3f", name, rows[-1]["ROC AUC"], rows[-1]["PR AUC"], rows[-1]["F1"])
    res = pd.DataFrame(rows)
    best_name = res.loc[res["PR AUC"].idxmax(), "model"]
    best, best_row = models[best_name], res.set_index("model").loc[best_name]
    p_best = test_probs[best_name]
    flag = (p_best >= best_row.Threshold).astype(int)

    # ---------------------------------------------------------------- ROC + PR curves
    fig, axes = c.figure(11, 4.6, ncols=2)
    for color, name in zip(c.SERIES, models):
        axes[0].plot(curves[name]["fpr"], curves[name]["tpr"], color=color, linewidth=2,
                     label=f"{name} ({res.set_index('model').loc[name, 'ROC AUC']:.3f})")
        axes[1].plot(curves[name]["recall"], curves[name]["precision"], color=color, linewidth=2,
                     label=f"{name} ({res.set_index('model').loc[name, 'PR AUC']:.3f})")
    axes[0].plot([0, 1], [0, 1], color=c.AXIS, linewidth=1, linestyle="--")
    axes[1].axhline(test[TARGET].mean(), color=c.AXIS, linewidth=1, linestyle="--")
    axes[0].set_xlabel("False positive rate", color=c.MUTED, fontsize=9)
    axes[0].set_ylabel("True positive rate", color=c.MUTED, fontsize=9)
    axes[1].set_xlabel("Recall (share of denials caught)", color=c.MUTED, fontsize=9)
    axes[1].set_ylabel("Precision", color=c.MUTED, fontsize=9)
    axes[0].set_title("ROC curve (AUC)", fontsize=10, color=c.INK_2, loc="left")
    axes[1].set_title("Precision-recall (average precision)", fontsize=10, color=c.INK_2, loc="left")
    for ax in axes:
        ax.legend(frameon=False, fontsize=8, loc="lower right" if ax is axes[0] else "upper right")
    c.headline(fig, f"{best_name} catches {best_row.Recall:.0%} of denials while flagging {best_row.Flagged:.1%} of applications",
               f"Out-of-time test on {len(test):,} applications from {config.TEST_YEAR}; dashed lines show a random guess")
    fig_curves = c.save(fig, "cls_roc_pr.png", top=0.82)

    # ---------------------------------------------------------------- calibration
    frac_pos, mean_pred = calibration_curve(test[TARGET], p_best, n_bins=10, strategy="quantile")
    fig, ax = c.figure(6.5, 5)
    ax.plot([0, 1], [0, 1], color=c.AXIS, linewidth=1, linestyle="--")
    ax.plot(mean_pred, frac_pos, color=c.BLUE, linewidth=2, marker="o", markersize=5)
    lim = max(mean_pred.max(), frac_pos.max()) * 1.08
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Predicted denial probability", color=c.MUTED, fontsize=9)
    ax.set_ylabel("Actual denial rate", color=c.MUTED, fontsize=9)
    c.headline(fig, "Predicted risk tracks actual denial rates", f"{best_name}, deciles of predicted risk, {config.TEST_YEAR}")
    fig_calibration = c.save(fig, "cls_calibration.png")

    # ---------------------------------------------------------------- permutation importance
    sample = test.sample(min(15_000, len(test)), random_state=config.RANDOM_STATE)
    perm = permutation_importance(best, sample[FEATURES], sample[TARGET], n_repeats=5, random_state=config.RANDOM_STATE,
                                  scoring="average_precision", n_jobs=-1)
    imp = pd.DataFrame({"feature": [LABELS[f] for f in FEATURES], "importance": perm.importances_mean, "std": perm.importances_std}) \
        .sort_values("importance", ascending=False)
    fig, ax = c.figure(9, 4.6)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=c.GRID, linewidth=0.8)
    top = imp.head(10).iloc[::-1]
    ax.barh(top.feature, top.importance, xerr=top["std"], color=c.BLUE, height=0.6, error_kw={"ecolor": c.INK_2, "elinewidth": 1})
    ax.tick_params(axis="y", labelsize=9, colors=c.INK_2)
    ax.set_xlabel("Drop in average precision when the feature is shuffled", color=c.MUTED, fontsize=9)
    c.headline(fig, f"{imp.iloc[0].feature} and {imp.iloc[1].feature.lower()} drive the denial predictions",
               f"Permutation importance, {best_name}, {config.TEST_YEAR} sample of 15,000 applications")
    fig_importance = c.save(fig, "cls_importance.png")

    # ---------------------------------------------------------------- interpretable views
    logit = models["Logistic regression"]
    names = [clean_feature_name(n) for n in logit.named_steps["prep"].get_feature_names_out()]
    odds = pd.DataFrame({"feature": names, "odds_ratio": np.exp(logit.named_steps["model"].coef_[0])})
    odds = odds[~odds.feature.str.startswith("county")]
    odds = pd.concat([odds.nlargest(6, "odds_ratio"), odds.nsmallest(4, "odds_ratio")])
    tree = models["Decision tree"]
    tree_names = [clean_feature_name(n) for n in tree.named_steps["prep"].get_feature_names_out()]
    tree_rules = export_text(tree.named_steps["model"], feature_names=tree_names, max_depth=3, decimals=1, show_weights=False)

    # ---------------------------------------------------------------- fairness audit
    audit = test.assign(p=p_best, flag=flag)
    def group_metrics(g: pd.DataFrame) -> pd.Series:
        approved, denied = g[g[TARGET] == 0], g[g[TARGET] == 1]
        return pd.Series({
            "applications": len(g), "actual_denial_rate": g[TARGET].mean(), "mean_predicted": g.p.mean(),
            "flag_rate": g.flag.mean(),
            "false_positive_rate": approved.flag.mean() if len(approved) else np.nan,
            "false_negative_rate": 1 - denied.flag.mean() if len(denied) else np.nan,
        })
    fairness = {}
    for attr in ["race_ethnicity", "sex"]:
        t = audit.groupby(attr).apply(group_metrics, include_groups=False)
        fairness[attr] = t[t.applications >= 500].sort_values("applications", ascending=False)
    fr = fairness["race_ethnicity"]
    fig, ax = c.figure(9, 4.4)
    x = np.arange(len(fr))
    ax.bar(x - 0.2, fr.actual_denial_rate, width=0.38, color=c.BLUE, label="Actual denial rate")
    ax.bar(x + 0.2, fr.mean_predicted, width=0.38, color=c.ORANGE, label="Average predicted risk", edgecolor=c.SURFACE)
    ax.set_xticks(x, fr.index, fontsize=8.5)
    ax.yaxis.set_major_formatter(__import__("matplotlib").ticker.PercentFormatter(1.0, decimals=0))
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    c.headline(fig, "Without seeing race, the model's risk scores still differ by group, as the outcomes do",
               f"{best_name}, {config.TEST_YEAR}. Predicted risk below the actual rate means the model under-predicts that group's denials.")
    fig_fairness = c.save(fig, "cls_fairness.png")

    # ---------------------------------------------------------------- report
    cm = confusion_matrix(test[TARGET], flag)
    tn, fp, fn, tp = cm.ravel()
    fmt = {"ROC AUC": "{:.3f}", "PR AUC": "{:.3f}", "Brier": "{:.4f}", "Log loss": "{:.3f}", "Threshold": "{:.2f}",
           "Precision": "{:.1%}", "Recall": "{:.1%}", "F1": "{:.3f}", "Flagged": "{:.1%}", "fit_seconds": "{:.1f}"}
    table = res.copy()
    for col, f in fmt.items():
        table[col] = table[col].map(f.format)
    def fair_table(t: pd.DataFrame) -> str:
        return md_table(pd.DataFrame({
            "group": t.index, "applications": t.applications.map("{:,.0f}".format),
            "actual denial rate": t.actual_denial_rate.map("{:.1%}".format), "avg predicted": t.mean_predicted.map("{:.1%}".format),
            "flagged": t.flag_rate.map("{:.1%}".format), "false positive rate": t.false_positive_rate.map("{:.1%}".format),
            "false negative rate": t.false_negative_rate.map("{:.1%}".format),
        }))
    fpr_range = (fr.false_positive_rate.min(), fr.false_positive_rate.max())
    hi_fpr, lo_fpr = fr.false_positive_rate.idxmax(), fr.false_positive_rate.idxmin()
    under = fr[(fr.mean_predicted < fr.actual_denial_rate)]
    funnel = pd.DataFrame(test_funnel).assign(rows=lambda d: d.rows.map("{:,}".format), removed=lambda d: d.removed.map("{:,}".format))
    lines = [
        "# 3. Classification: predicting mortgage denials", "",
        "**Question.** Using only what is known when an application is decided, how well can we predict a denial?", "",
        "**Intended use.** An early-warning flag so a loan officer or housing counselor can contact likely-to-be-denied applicants and help "
        "them fix the issue (paying down debt, a larger down payment, a different loan program). Not for making or automating credit decisions.", "",
        f"**Design.** Fit on 80% of {config.TRAIN_YEAR} applications; the other 20% chooses each model's decision threshold (maximum F1). "
        f"Everything is then scored on {len(test):,} applications from {config.TEST_YEAR}. Race, ethnicity, sex and age are never inputs. "
        "Interest rate, pricing and denial-reason fields are excluded because they only exist after the decision.", "",
        f"Denial rate in scope: {test[TARGET].mean():.1%} ({config.TEST_YEAR}). Accuracy is not reported: predicting \"approve\" for everyone would be "
        f"{1 - test[TARGET].mean():.1%} accurate and useless.", "",
        "## Sample", "", md_table(funnel), "",
        "## Model comparison", "", f"![ROC and PR curves]({fig_curves})", "", md_table(table), "",
        f"- **{best_name}** is best on the metric that matters for a rare outcome, average precision ({best_row['PR AUC']:.3f}, "
        f"vs {test[TARGET].mean():.3f} for a random guess). ROC AUC is {best_row['ROC AUC']:.3f}.",
        f"- At its threshold ({best_row.Threshold:.2f}), it flags {best_row.Flagged:.1%} of applications and catches {best_row.Recall:.0%} of denials; "
        f"{best_row.Precision:.0%} of flagged applications are actually denied.",
        f"- Confusion matrix ({config.TEST_YEAR}): {tp:,} denials caught, {fn:,} missed, {fp:,} approved applications flagged, {tn:,} correctly left alone.",
        f"- Logistic regression trails (average precision {res.set_index('model').loc['Logistic regression', 'PR AUC']:.3f}). Denial depends on "
        "thresholds and interactions, such as debt-to-income above about 50% and lender-specific practices (see the tree rules below), that a "
        "linear model cannot represent. Its odds ratios are still useful for explaining direction.", "",
        "## Calibration", "", f"![Calibration]({fig_calibration})", "",
        f"Brier score {best_row.Brier:.4f}. If the points follow the diagonal, a predicted 30% risk means about 30% of such applications are denied, "
        "so the score can be read as a probability.", "",
        "## What drives the predictions", "", f"![Permutation importance]({fig_importance})", "",
        "Logistic regression odds ratios (above 1 raises the odds of denial; numeric features are per standard deviation):", "",
        md_table(odds.assign(odds_ratio=odds.odds_ratio.map("{:.2f}".format))), "",
        "Top of the depth-5 decision tree, as readable rules:", "", "```", tree_rules.strip(), "```", "",
        "## Fairness audit", "", f"![Fairness]({fig_fairness})", "",
        fair_table(fairness["race_ethnicity"]), "", fair_table(fairness["sex"]), "",
        f"- **False positive rates differ {fpr_range[1] / fpr_range[0]:.1f}×.** Approved {hi_fpr} applicants are wrongly flagged {fpr_range[1]:.1%} "
        f"of the time vs {fpr_range[0]:.1%} for {lo_fpr} applicants. Their actual denial rates differ by less "
        f"({fr.loc[hi_fpr, 'actual_denial_rate']:.1%} vs {fr.loc[lo_fpr, 'actual_denial_rate']:.1%}, "
        f"{fr.loc[hi_fpr, 'actual_denial_rate'] / fr.loc[lo_fpr, 'actual_denial_rate']:.1f}×), so the gap is not just a base-rate effect: "
        f"approved {hi_fpr} applicants more often share the profile the model associates with denial. A flag used for outreach costs a phone "
        "call; used for anything stricter, this disparity would be a serious problem and would need mitigation (e.g. group-aware threshold review).",
        f"- False negative rates are high for every group ({fr.false_negative_rate.min():.0%}–{fr.false_negative_rate.max():.0%}): about half of "
        "denials are not predictable from the published application data.",
        "- The model under-predicts denials for: " + (", ".join(f"{g} ({r.mean_predicted:.1%} predicted vs {r.actual_denial_rate:.1%} actual)"
                                                       for g, r in under.iterrows()) or "no group") + ". Part of their denial risk is not captured "
        "by the published data, plausibly credit history, which HMDA does not include.",
        "- Excluding race from the inputs does not make a model fair by itself: lender, county and tract features correlate with it. "
        "That is why this audit reports outcomes by group instead of assuming neutrality.", "",
    ]
    (config.REPORT_DIR / "03_classification.md").write_text("\n".join(lines), encoding="utf-8")

    out = config.REPORT_DIR / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    (out / "classification.json").write_text(json.dumps({
        "train_rows": len(fit), "validation_rows": len(valid), "test_rows": len(test), "best_model": best_name,
        "models": rows, "importance": imp.to_dict("records"),
        "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        "roc": {n: {"fpr": v["fpr"][:: max(1, len(v["fpr"]) // 150)].round(4).tolist(),
                    "tpr": v["tpr"][:: max(1, len(v["tpr"]) // 150)].round(4).tolist()} for n, v in curves.items()},
        "calibration": {"predicted": mean_pred.round(4).tolist(), "actual": frac_pos.round(4).tolist()},
        "fairness": {k: v.reset_index().to_dict("records") for k, v in fairness.items()},
        "odds_ratios": odds.to_dict("records"),
    }, indent=2, default=float))
    scored = test.assign(denial_probability=p_best.round(4), flagged=flag)
    scored.to_csv(config.PROCESSED_DIR / f"classification_scored_{config.TEST_YEAR}.csv.gz", index=False)
    log.info("Wrote reports/03_classification.md (best: %s)", best_name)

# Tableau report: build and publish

This rebuilds the project's findings as a **Tableau Story** with four story points and publishes it to Tableau Public, which gives it a live public URL. It takes about 45 to 60 minutes. Tableau Public is free.

The repository already has a live web dashboard with the same content at https://hkandan-hash.github.io/nj-mortgage-analytics/. Use it as the reference design and to check your numbers.

## 1. Install and sign in

1. Install **Tableau Public** from https://public.tableau.com/app/discover. It's free and runs on Windows and macOS.
2. Create a Tableau Public account. Publishing requires one, and the published URL will be under your profile.

## 2. Connect the data

Use **Connect → To a File → Text file**, and add each CSV from [`exports/tableau/`](../exports/tableau/) as its own data source:

| File | Grain | Used in |
|---|---|---|
| `applications_2024_scored.csv` | one row per 2024 application in the model sample, with the model's `denial_probability` and `flagged` | Overview, drivers, fairness |
| `county_summary.csv` | year × county | Map |
| `denial_by_dti_ltv.csv` | year × DTI band × LTV band | Heatmap |
| `denial_reasons.csv` | year × reason | Reasons bar |
| `model_metrics.csv` | task × model | Model comparison |
| `feature_importance.csv` | task × feature | Drivers |
| `pricing_effects.csv` | one row per effect | Pricing |
| `pricing_gaps.csv` | one row per group | Fairness |
| `fairness_audit.csv` | attribute × group | Fairness |
| `roc_curves.csv` | model × point | Model comparison |

Set these field types:
- In `county_summary`, give `county_fips` the **County** geographic role, or use `county` with **State/Province = New Jersey**.
- Set `year` to a **Discrete** dimension everywhere.

## 3. Calculated fields

In `applications_2024_scored`:

```
// Denial rate
AVG([denied])

// Avg predicted risk
AVG([denial_probability])

// Flag outcome
IF [flagged] = 1 AND [denied] = 1 THEN "Caught denial"
ELSEIF [flagged] = 1 AND [denied] = 0 THEN "False alarm"
ELSEIF [flagged] = 0 AND [denied] = 1 THEN "Missed denial"
ELSE "Correctly not flagged" END

// Risk band
IF [denial_probability] >= 0.5 THEN "High (50%+)"
ELSEIF [denial_probability] >= 0.2 THEN "Elevated (20-50%)"
ELSEIF [denial_probability] >= 0.08 THEN "Moderate (8-20%)"
ELSE "Low (<8%)" END
```

In `pricing_gaps`:

```
// Unexplained gap (bps)
[residual] * 100

// Significant
IF ABS([residual]) > 1.96 * [residual_se] THEN "Significant" ELSE "Not significant" END
```

In `fairness_audit`, add a filter `attribute = "race_ethnicity"`. Make a second copy of the sheet filtered to `sex`.

## 4. Sheets

| # | Sheet | How |
|---|---|---|
| 1 | KPI tiles | From `county_summary`: SUM(applications), weighted denial rate, filtered to year 2024 |
| 2 | County map | `county_summary`: county on Detail, `denial_rate` on Color (sequential blue), year as a filter shown as a single-value toggle |
| 3 | DTI × LTV heatmap | `denial_by_dti_ltv`: `dti_band` on Rows and `ltv_band` on Columns (sort both manually in band order), `denial_rate` on Color and Label; filter `applications >= 50` |
| 4 | Denial reasons | `denial_reasons`: horizontal bar of `share`, sorted descending, year filter |
| 5 | Regression models | `model_metrics` filtered to the regression task: bar of `R2` by model, best model highlighted |
| 6 | Pricing effects | `pricing_effects`: horizontal bar of `effect`, with error bars from a dual axis using `effect ± ci`, colored by sign |
| 7 | Classification models | `model_metrics` filtered to classification: table of ROC AUC, PR AUC, Precision, Recall, F1 |
| 8 | ROC curves | `roc_curves`: `fpr` on Columns, `tpr` on Rows, Mark type Line, `model` on Color and Path |
| 9 | Denial drivers | `feature_importance` filtered to classification: top 8 features by `importance` |
| 10 | Risk vs outcome | `applications_2024_scored`: `Risk band` on Columns, `Denial rate` as bars, count of applications as labels |
| 11 | Fairness, denials | `fairness_audit` (race_ethnicity): `actual_denial_rate` and `mean_predicted` side by side by group |
| 12 | Fairness, pricing | `pricing_gaps`: `Unexplained gap (bps)` by group, colored by `Significant` |

## 5. Story (the report)

Create **Story → New Story**, size *Automatic*, with four story points. Each caption should state the finding, not just the topic:

1. **"1 in 9 New Jersey purchase applications is denied, and it varies by county"**: dashboard with KPI tiles, the county map and denial reasons.
2. **"Debt-to-income above 50% is where denials take off"**: heatmap and reasons.
3. **"Lender choice and equity drive pricing; gradient boosting predicts next year's denials best"**: regression models, pricing effects, classification models, ROC and drivers.
4. **"Fairness: most pricing gaps are explained, but error rates differ by group"**: both fairness sheets plus the error-rate table.

Use the same colors as the web dashboard so the two match:
- **Categorical:** `#2a78d6`, `#eb6834`, `#1baf7a`, `#eda100`.
- **Sequential:** blue, light to dark.
- **Diverging:** red–blue.

## 6. Publish (this makes it live)

1. **File → Save to Tableau Public As…**, then name it *NJ Mortgage Lending 2023–2024*.
2. In the browser that opens, check that the Story renders and turn on **Show Viz on Profile**.
3. Copy the share URL and add it to the top of the project README, under "Live dashboards".

Tableau Public stores the data inside the published workbook. All the data here is public HMDA data aggregated by the project, with no personal identifiers, so it's fine to publish.

## 7. Check your numbers

Your Tableau numbers should match these values from the reports:

| Check | Value |
|---|---|
| 2024 applications / denial rate | 94,174 / 11.4% |
| Highest and lowest county denial rate, 2024 | Essex 14.8%, Gloucester 8.3% |
| DTI 50–60% denial rate, 2024 | 26% (vs 6–7% for DTI 20–49%) |
| Best regression model (out-of-time R²) | Gradient boosting, 0.465 |
| Best classifier (ROC AUC / average precision) | Gradient boosting, 0.848 / 0.586 |
| Scored applications in `applications_2024_scored.csv` | 79,309 |

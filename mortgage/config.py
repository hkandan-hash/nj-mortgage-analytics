"""Paths, years, and the HMDA code tables used across the project."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
EXPORT_DIR = ROOT / "exports" / "tableau"
DASHBOARD_DIR = ROOT / "docs"

STATE = "NJ"
TRAIN_YEAR = 2023  # models are fit on this year...
TEST_YEAR = 2024   # ...and evaluated on the next one (out-of-time)
YEARS = [TRAIN_YEAR, TEST_YEAR]

# FFIEC HMDA Data Browser API: home-purchase applications that reached a credit decision
# (1 = originated, 2 = approved but not accepted, 3 = denied).
API_URL = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"
API_PARAMS = {"states": STATE, "actions_taken": "1,2,3", "loan_purposes": "1"}

RANDOM_STATE = 42

COUNTIES = {
    "34001": "Atlantic", "34003": "Bergen", "34005": "Burlington", "34007": "Camden", "34009": "Cape May",
    "34011": "Cumberland", "34013": "Essex", "34015": "Gloucester", "34017": "Hudson", "34019": "Hunterdon",
    "34021": "Mercer", "34023": "Middlesex", "34025": "Monmouth", "34027": "Morris", "34029": "Ocean",
    "34031": "Passaic", "34033": "Salem", "34035": "Somerset", "34037": "Sussex", "34039": "Union",
    "34041": "Warren",
}

LOAN_TYPES = {"1": "Conventional", "2": "FHA", "3": "VA", "4": "USDA"}
OCCUPANCY = {"1": "Principal residence", "2": "Second residence", "3": "Investment property"}
LIEN = {"1": "First lien", "2": "Subordinate lien"}
AUS = {
    "1": "Desktop Underwriter", "2": "Loan Product Advisor", "3": "FHA TOTAL Scorecard", "4": "USDA GUS",
    "5": "Other AUS", "6": "No AUS used", "7": "Internal proprietary system",
}
CREDIT_SCORE_MODEL = {
    "1": "Equifax Beacon 5.0", "2": "Experian Fair Isaac", "3": "FICO Classic 04", "7": "More than one model",
    "8": "Other model", "9": "Not applicable",
}  # rare and newly added codes are grouped as "Other or newer model"
DENIAL_REASONS = {
    "1": "Debt-to-income ratio", "2": "Employment history", "3": "Credit history", "4": "Collateral",
    "5": "Insufficient cash", "6": "Unverifiable information", "7": "Application incomplete",
    "8": "Mortgage insurance denied", "9": "Other",
}
# Debt-to-income is published as bands outside 36-49%; map each band to its midpoint.
DTI_BANDS = {"<20%": 15.0, "20%-<30%": 25.0, "30%-<36%": 33.0, "50%-60%": 55.0, ">60%": 65.0}

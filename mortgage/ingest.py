"""Ingestion: download raw HMDA files from the FFIEC API into data/raw, untouched.

Each file is written atomically and recorded in data/raw/manifest.json with its row count,
SHA-256 and download time, so every downstream number traces back to exact source bytes.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from . import config

log = logging.getLogger(__name__)
REQUIRED_COLUMNS = {
    "activity_year", "lei", "county_code", "action_taken", "loan_type", "lien_status", "loan_amount",
    "property_value", "income", "debt_to_income_ratio", "loan_to_value_ratio", "interest_rate", "rate_spread",
    "loan_term", "intro_rate_period", "occupancy_type", "aus-1", "applicant_credit_score_type",
    "denial_reason-1", "derived_race", "derived_ethnicity", "derived_sex", "applicant_age",
}


def raw_path(year: int) -> Path:
    return config.RAW_DIR / f"hmda_{config.STATE.lower()}_{year}.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate(path: Path) -> int:
    """Check the header has every column we rely on; return the data row count."""
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = set(next(reader))
        missing = REQUIRED_COLUMNS - header
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        return sum(1 for _ in reader)


def ingest(years: list[int] = config.YEARS, force: bool = False) -> dict:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = config.RAW_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for year in years:
        target = raw_path(year)
        if target.exists() and not force:
            log.info("Already have %s", target.name)
        else:
            query = urllib.parse.urlencode({**config.API_PARAMS, "years": year})
            log.info("Downloading %s", target.name)
            tmp = target.with_suffix(".part")
            with urllib.request.urlopen(f"{config.API_URL}?{query}", timeout=600) as resp, tmp.open("wb") as out:
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
            tmp.replace(target)
        rows = validate(target)
        manifest[target.name] = {
            "year": year,
            "rows": rows,
            "sha256": sha256(target),
            "source": f"{config.API_URL}?{urllib.parse.urlencode({**config.API_PARAMS, 'years': year})}",
            "recorded_at": manifest.get(target.name, {}).get("recorded_at") or datetime.now(timezone.utc).isoformat(),
        }
        log.info("%s: %s rows", target.name, f"{rows:,}")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest

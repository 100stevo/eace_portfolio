"""
Module 1: Data Loader & Raw Validator
--------------------------------------
Loads military, VA, and civilian(-comparator) source extracts, validates
required columns, merges on Patient_ID, and computes Days_Since_Injury.

Designed for the DoW/VA secure environment (Polars for speed on large
registry extracts). Run standalone via:
    python data_loader.py --military ../data/military_source.csv \
                           --va ../data/va_source.csv \
                           --civilian ../data/civilian_source.csv \
                           --out ../outputs/merged_raw.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

REQUIRED_COLUMNS = {
    "military": {"Patient_ID", "Age", "ISS_Score", "Cohort_Type", "Injury_Date"},
    "va": {"Patient_ID", "Days_Post_Injury", "SMFA_Mobility"},
    "civilian": {"Patient_ID"},
}


class SchemaValidationError(Exception):
    """Raised when a required column is missing from a source file."""


def _read_any(path: str | Path) -> pl.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)
    return pl.read_csv(path, try_parse_dates=True)


def _validate_columns(df: pl.DataFrame, required: set[str], source_name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise SchemaValidationError(
            f"Source '{source_name}' is missing required columns: {sorted(missing)}"
        )


def load_raw_data(
    military_source_path: str | Path,
    va_source_path: str | Path,
    civilian_source_path: str | Path,
) -> pl.DataFrame:
    """
    Load, validate, and merge the three registry extracts into a single
    standardized Polars DataFrame.

    Parameters
    ----------
    military_source_path : path to DoW registry extract (patient-level, one row/patient)
    va_source_path : path to VA/FITBIR/METALS longitudinal follow-up extract
                      (patient-timepoint level, one row/patient/follow-up)
    civilian_source_path : path to comparator / outcome extract (e.g. RTD status)

    Returns
    -------
    pl.DataFrame
        Merged, standardized dataset with a `Days_Since_Injury` column
        (aliased from Days_Post_Injury for schema consistency), one row
        per patient-timepoint.

    Raises
    ------
    SchemaValidationError
        If any source is missing a required column.
    FileNotFoundError
        If a source path does not exist.
    """
    for label, p in (
        ("military", military_source_path),
        ("va", va_source_path),
        ("civilian", civilian_source_path),
    ):
        if not Path(p).exists():
            raise FileNotFoundError(f"{label} source not found at: {p}")

    military = _read_any(military_source_path)
    va = _read_any(va_source_path)
    civilian = _read_any(civilian_source_path)

    _validate_columns(military, REQUIRED_COLUMNS["military"], "military")
    _validate_columns(va, REQUIRED_COLUMNS["va"], "va")
    _validate_columns(civilian, REQUIRED_COLUMNS["civilian"], "civilian")

    # Merge: military (1 row/patient) <- va (N rows/patient) <- civilian (1 row/patient, outcome)
    merged = va.join(military, on="Patient_ID", how="left")
    merged = merged.join(civilian, on="Patient_ID", how="left")

    if "Days_Post_Injury" in merged.columns:
        merged = merged.rename({"Days_Post_Injury": "Days_Since_Injury"})
    elif "Injury_Date" in merged.columns:
        # fallback: derive from calendar dates if a per-record event date exists
        merged = merged.with_columns(
            (pl.col("Injury_Date").cast(pl.Date)).alias("Injury_Date")
        )

    unresolved_ids = merged.filter(pl.col("Age").is_null())["Patient_ID"].n_unique()
    if unresolved_ids:
        print(
            f"[data_loader] WARNING: {unresolved_ids} Patient_IDs in follow-up data "
            "did not match a military source record.",
            file=sys.stderr,
        )

    return merged


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Load and merge trauma registry sources.")
    parser.add_argument("--military", required=True)
    parser.add_argument("--va", required=True)
    parser.add_argument("--civilian", required=True)
    parser.add_argument("--out", required=True, help="Output path (.parquet or .csv)")
    args = parser.parse_args()

    df = load_raw_data(args.military, args.va, args.civilian)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".parquet":
        df.write_parquet(out_path)
    else:
        df.write_csv(out_path)

    print(f"[data_loader] Merged {df.height} rows x {df.width} cols -> {out_path}")


if __name__ == "__main__":
    _cli()

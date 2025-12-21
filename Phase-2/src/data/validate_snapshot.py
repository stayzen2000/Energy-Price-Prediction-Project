from __future__ import annotations  # future import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd

EXPECTED_COLUMNS = [
    "timestamp",
    "zone_id",
    "demand_mw",
    "demand_forecast_mw",
    "price_per_mwh",
    "temp_c",
    "humidity",
    "wind_speed",
    "precipitation",
]

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(msg: str) -> None:
    print(f"\n❌ VALIDATION FAILED: {msg}\n", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate frozen Phase-2 CSV snapshot.")
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to Phase 2 frozen CSV snapshot",
    )
    parser.add_argument(
        "--write_reports",
        action="store_true",
        help="Write reports/data_snapshot.md and reports/validation_summary.md",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        fail(f"CSV not found at: {csv_path}")

    # --- Load ---
    df = pd.read_csv(csv_path)

    # --- Column contract ---
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if missing_cols:
        fail(f"Missing expected columns: {missing_cols}")
    if extra_cols:
        print(f"⚠️ Extra columns present (not fatal): {extra_cols}")

    # --- Timestamp parsing ---
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        bad = df[df["timestamp"].isna()].head(5)
        fail(f"Unparseable timestamps found. Examples:\n{bad}")

    # Enforce sort
    df = df.sort_values("timestamp").reset_index(drop=True)

    # --- Time continuity checks ---
    # duplicates
    dup_count = df["timestamp"].duplicated().sum()
    if dup_count > 0:
        examples = df.loc[df["timestamp"].duplicated(keep=False), "timestamp"].head(10)
        fail(f"Duplicate timestamps found: {dup_count}. Examples:\n{examples}")

    # hourly frequency check
    deltas = df["timestamp"].diff().dropna()
    # delta should be exactly 1 hour everywhere
    expected_delta = pd.Timedelta(hours=1)
    bad_delta_mask = deltas != expected_delta
    if bad_delta_mask.any():
        bad_idx = bad_delta_mask[bad_delta_mask].index[:10]
        # show surrounding rows for context
        context = df.loc[np.r_[bad_idx - 1, bad_idx, bad_idx + 1].clip(0, len(df) - 1), ["timestamp"]]
        fail(
            "Timestamps are not strictly hourly continuous (1h steps). "
            f"Found {bad_delta_mask.sum()} non-1h gaps. Context:\n{context}"
        )

    # --- Zone check ---
    zone_unique = df["zone_id"].nunique(dropna=True)
    if zone_unique != 1:
        zones = df["zone_id"].dropna().unique()[:20]
        fail(f"Expected exactly 1 zone_id, found {zone_unique}. Examples: {zones}")

    # --- Row count sanity ---
    row_count = len(df)
    if row_count < 15000 or row_count > 20000:
        print(f"⚠️ Row count is outside expected range (~17,200). Found: {row_count}")

    # --- Missingness ---
    missing_pct = (df[EXPECTED_COLUMNS].isna().mean() * 100).round(3).sort_values(ascending=False)

    # --- Simple value sanity checks (not too strict) ---
    # demand should be positive and not crazy-large
    zero_target_count = (df["demand_mw"] == 0).sum()
    neg_target_count = (df["demand_mw"] < 0).sum()

    if neg_target_count > 0:
        bad = df[df["demand_mw"] < 0][["timestamp", "demand_mw"]].head(5)
        fail(f"Negative demand_mw found (invalid). Examples:\n{bad}")

    if zero_target_count > 0:
        examples = df.loc[df["demand_mw"] == 0, ["timestamp", "demand_mw"]].head(5)
        print(f"⚠️ Found demand_mw == 0 for {zero_target_count} rows. Treat as missing later.")
        print(examples.to_string(index=False))


    # weather sanity (wide bounds to avoid false failures)
    if (df["temp_c"] < -60).any() or (df["temp_c"] > 60).any():
        bad = df[(df["temp_c"] < -60) | (df["temp_c"] > 60)][["timestamp", "temp_c"]].head(5)
        fail(f"Unrealistic temp_c values found. Examples:\n{bad}")

    # humidity sanity (0-100-ish)
    if (df["humidity"] < 0).any() or (df["humidity"] > 100).any():
        bad = df[(df["humidity"] < 0) | (df["humidity"] > 100)][["timestamp", "humidity"]].head(5)
        fail(f"Humidity out of [0,100] found. Examples:\n{bad}")

    # --- Summary ---
    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()
    checksum = sha256_file(csv_path)

    summary_lines = []
    summary_lines.append("✅ VALIDATION PASSED")
    summary_lines.append(f"CSV: {csv_path}")
    summary_lines.append(f"SHA256: {checksum}")
    summary_lines.append(f"Rows: {row_count:,}")
    summary_lines.append(f"Time range: {ts_min} → {ts_max}")
    summary_lines.append(f"Columns: {list(df.columns)}")
    summary_lines.append("\nMissingness (%):")
    summary_lines.append(missing_pct.to_string())

    print("\n" + "\n".join(summary_lines) + "\n")

    # --- Optional reports ---
    if args.write_reports:
        reports_dir = csv_path.parents[2] / "reports"  # Phase-2/reports
        reports_dir.mkdir(parents=True, exist_ok=True)

        snapshot_md = reports_dir / "data_snapshot.md"
        validation_md = reports_dir / "validation_summary.md"

        snapshot_md.write_text(
            "\n".join(
                [
                    "# Phase 2 Data Snapshot",
                    "",
                    f"- CSV: `{csv_path}`",
                    f"- SHA256: `{checksum}`",
                    f"- Rows: `{row_count:,}`",
                    f"- Time range: `{ts_min}` → `{ts_max}`",
                    f"- Unique zone_id: `{zone_unique}`",
                    "",
                    "## Columns",
                    "",
                    "\n".join([f"- `{c}`" for c in EXPECTED_COLUMNS]),
                    "",
                ]
            ),
            encoding="utf-8",
        )

        validation_md.write_text(
            "\n".join(
                [
                    "# Phase 2 Validation Summary",
                    "",
                    "## Result",
                    "",
                    "✅ PASSED",
                    "",
                    "## Missingness (%)",
                    "",
                    "```",
                    missing_pct.to_string(),
                    "```",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        print(f"📝 Wrote: {snapshot_md}")
        print(f"📝 Wrote: {validation_md}")


if __name__ == "__main__":
    main()
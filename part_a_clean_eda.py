"""Part A: clean the raw export, build the modelling table, and write delay evidence.

Run:  python part_a_clean_eda.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
from shipment_prep import clean_shipments, build_features, FEATURES  # noqa: E402

DATA = os.path.join(HERE, "data")
OUTPUTS = os.path.join(HERE, "outputs")
os.makedirs(OUTPUTS, exist_ok=True)

DISTANCE_BINS = [0, 200, 400, 600, 2000]
DISTANCE_LABELS = ["0-200", "201-400", "401-600", "601+"]


def main():
    raw = pd.read_csv(os.path.join(DATA, "raw_shipments.csv"))
    hubs = pd.read_csv(os.path.join(DATA, "hubs.csv"))

    # "Before" measurements: after removing duplicates, before any filling.
    dedup = raw.drop_duplicates().reset_index(drop=True)

    clean, medians = clean_shipments(raw)
    modelling = build_features(clean, hubs)
    modelling["is_delayed"] = (modelling["delivery_hours"] > modelling["promised_hours"]).astype(int)

    modelling.to_csv(os.path.join(DATA, "modelling_table.csv"), index=False)

    write_checks(len(raw), dedup, modelling, medians)
    write_delay_by_group(modelling)
    write_correlations(modelling)

    print("Part A complete:",
          f"{len(modelling)} rows, delayed={int(modelling['is_delayed'].sum())}")


def write_checks(raw_rows, dedup, modelling, medians):
    q1 = modelling["delivery_hours"].quantile(0.25)
    q3 = modelling["delivery_hours"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = int(((modelling["delivery_hours"] < lower) | (modelling["delivery_hours"] > upper)).sum())

    rows = [
        ("raw_rows", raw_rows),
        ("duplicate_rows_removed", raw_rows - len(dedup)),
        ("rows_after_dedup", len(dedup)),
        ("missing_distance_km_before", int(dedup["distance_km"].isna().sum())),
        ("missing_parcel_weight_before", int(dedup["parcel_weight_kg"].isna().sum())),
        ("sentinel_weight_rows", int((dedup["parcel_weight_kg"] > 100).sum())),
        ("missing_declared_value_before", int(dedup["declared_value"].isna().sum())),
        ("missing_payment_mode_before", int(dedup["payment_mode"].isna().sum())),
        ("service_type_spellings_before", int(dedup["service_type"].nunique())),
        ("service_type_spellings_after", int(modelling["service_type"].nunique())),
        ("median_distance_km", round(medians["distance_km"], 4)),
        ("median_parcel_weight_kg", round(medians["parcel_weight_kg"], 4)),
        ("median_declared_value_inr", round(medians["declared_value_inr"], 4)),
        ("declared_value_mean_inr", round(float(modelling["declared_value_inr"].mean()), 4)),
        ("declared_value_median_inr", round(float(modelling["declared_value_inr"].median()), 4)),
        ("log_declared_value_mean", round(float(modelling["log_declared_value"].mean()), 4)),
        ("log_declared_value_median", round(float(modelling["log_declared_value"].median()), 4)),
        ("rows_after_hub_merge", len(modelling)),
        ("nulls_in_features_and_target",
         int(modelling[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum())),
        ("delayed_count", int(modelling["is_delayed"].sum())),
        ("delay_rate", round(float(modelling["is_delayed"].mean()), 4)),
        ("hours_q1", round(float(q1), 4)),
        ("hours_q3", round(float(q3), 4)),
        ("hours_lower_bound", round(float(lower), 4)),
        ("hours_upper_bound", round(float(upper), 4)),
        ("hours_outlier_count", outliers),
    ]
    checks = pd.DataFrame(
        {"metric": [m for m, _ in rows],
         "value": pd.Series([v for _, v in rows], dtype=object)}
    )
    checks.to_csv(os.path.join(OUTPUTS, "part_a_checks.csv"), index=False)


def write_delay_by_group(modelling):
    df = modelling.copy()
    df["distance_band"] = pd.cut(df["distance_km"], bins=DISTANCE_BINS,
                                 labels=DISTANCE_LABELS, right=True)
    rows = []
    for dim in ["service_type", "hub_tier", "distance_band"]:
        for group, sub in df.groupby(dim, observed=True):
            rows.append({
                "dimension": dim,
                "group": group,
                "shipments": int(len(sub)),
                "delayed": int(sub["is_delayed"].sum()),
                "delay_rate": round(float(sub["is_delayed"].mean()), 4),
                "mean_hours": round(float(sub["delivery_hours"].mean()), 4),
            })
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUTS, "part_a_delay_by_group.csv"), index=False)


def write_correlations(modelling):
    numeric = modelling.select_dtypes(include=[np.number])
    corr = numeric.corr().round(4)
    corr.to_csv(os.path.join(OUTPUTS, "part_a_correlations.csv"))


if __name__ == "__main__":
    main()

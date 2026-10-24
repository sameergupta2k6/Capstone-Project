"""Shared constants and preprocessing for the Delay Intelligence capstone.

This module is imported by every Part so that cleaning and feature building are
defined exactly once (consistency between training and prediction).
"""
import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

# Columns whose gaps are filled with a stored median.
_FILL_COLUMNS = ["distance_km", "parcel_weight_kg", "declared_value_inr"]
# Parcels above this weight are a scanner fault: the network accepts up to 40 kg.
_WEIGHT_SENTINEL_LIMIT = 100


def clean_shipments(raw, medians=None):
    """Turn a raw shipments export into a clean table.

    Returns (clean_df, medians). When ``medians`` is None the three fill-in
    medians are computed from this table's non-missing values; otherwise the
    supplied dictionary is used so new parcels are filled exactly as training was.
    """
    df = raw.drop_duplicates().reset_index(drop=True)

    df["service_type"] = df["service_type"].str.strip().str.title()

    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    declared = (
        df["declared_value"]
        .str.replace("\u20b9", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    df["declared_value_inr"] = declared.astype(float)
    df = df.drop(columns=["declared_value"])

    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])

    df.loc[df["parcel_weight_kg"] > _WEIGHT_SENTINEL_LIMIT, "parcel_weight_kg"] = np.nan

    if medians is None:
        medians = {col: float(df[col].median()) for col in _FILL_COLUMNS}
    for col in _FILL_COLUMNS:
        df[col] = df[col].fillna(medians[col])

    return df, medians


def build_features(clean, hubs):
    """Attach hub attributes and derive the engineered columns.

    Never reads ``delivery_hours`` so it also works on undelivered parcels.
    """
    hub_cols = ["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]
    df = clean.merge(hubs[hub_cols], on="hub_id", how="left")

    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)
    df["log_declared_value"] = np.log(df["declared_value_inr"])

    return df

"""Part D (scoring): turn new parcels into a flagged action list.

Run:  python predict_delay.py
Uses the shared cleaning/feature functions so prediction matches training exactly.
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
from shipment_prep import clean_shipments, build_features  # noqa: E402

DATA = os.path.join(HERE, "data")
OUTPUTS = os.path.join(HERE, "outputs")
MODELS = os.path.join(HERE, "models")


def main():
    new = pd.read_csv(os.path.join(DATA, "new_shipments.csv"))
    hubs = pd.read_csv(os.path.join(DATA, "hubs.csv"))
    model = joblib.load(os.path.join(MODELS, "delay_model.pkl"))
    threshold = float(pd.read_csv(os.path.join(MODELS, "delay_threshold.csv"))["threshold"].iloc[0])
    feature_order = pd.read_csv(os.path.join(MODELS, "feature_order.csv"))["feature"].tolist()
    impute = pd.read_csv(os.path.join(MODELS, "impute_values.csv"))
    medians = dict(zip(impute["column"], impute["median"].astype(float)))
    hub_segments = pd.read_csv(os.path.join(OUTPUTS, "hub_segments.csv"))

    clean, _ = clean_shipments(new, medians=medians)
    feat = build_features(clean, hubs)

    X = feat[feature_order]
    pos = list(model.classes_).index(1)
    feat["delay_probability"] = np.round(model.predict_proba(X)[:, pos], 4)
    feat["flagged"] = (feat["delay_probability"] >= threshold).astype(int)

    segment_by_hub = dict(zip(hub_segments["hub_id"], hub_segments["segment"]))
    feat["segment"] = feat["hub_id"].map(segment_by_hub)

    conditions = [
        (feat["flagged"] == 1) & (feat["segment"] == "Escalate"),
        (feat["flagged"] == 1),
    ]
    feat["action"] = np.select(conditions,
                               ["Escalate to hub manager", "Proactive customer call"],
                               default="No action")

    out = feat[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
    out.to_csv(os.path.join(OUTPUTS, "new_shipment_actions.csv"), index=False)

    print("predict_delay complete:",
          f"{len(out)} parcels, flagged={int(out['flagged'].sum())}, "
          f"escalations={int((out['action'] == 'Escalate to hub manager').sum())}")


if __name__ == "__main__":
    main()

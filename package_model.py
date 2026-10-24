"""Part D (packaging): fit the final classifier and persist reusable artefacts.

Run:  python package_model.py
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
from shipment_prep import FEATURES, clean_shipments  # noqa: E402

DATA = os.path.join(HERE, "data")
OUTPUTS = os.path.join(HERE, "outputs")
MODELS = os.path.join(HERE, "models")
os.makedirs(MODELS, exist_ok=True)
os.makedirs(OUTPUTS, exist_ok=True)


def main():
    config = pd.read_csv(os.path.join(OUTPUTS, "part_c_final_config.csv")).iloc[0]
    k = int(config["selector_k"])
    n_estimators = int(config["n_estimators"])
    max_depth = None if str(config["max_depth"]) == "unlimited" else int(config["max_depth"])
    min_samples_leaf = int(config["min_samples_leaf"])
    threshold = float(config["threshold"])

    modelling = pd.read_csv(os.path.join(DATA, "modelling_table.csv"))
    X = modelling[FEATURES]
    y = modelling["is_delayed"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    final = Pipeline([
        ("selector", SelectKBest(f_classif, k=k)),
        ("model", RandomForestClassifier(random_state=42, n_estimators=n_estimators,
                                         max_depth=max_depth, min_samples_leaf=min_samples_leaf)),
    ]).fit(X_train, y_train)

    joblib.dump(final, os.path.join(MODELS, "delay_model.pkl"))
    pd.DataFrame({"threshold": [threshold]}).to_csv(os.path.join(MODELS, "delay_threshold.csv"), index=False)
    pd.DataFrame({"feature": FEATURES}).to_csv(os.path.join(MODELS, "feature_order.csv"), index=False)

    raw = pd.read_csv(os.path.join(DATA, "raw_shipments.csv"))
    _, medians = clean_shipments(raw)
    pd.DataFrame({"column": list(medians.keys()),
                  "median": [round(v, 4) for v in medians.values()]}
                 ).to_csv(os.path.join(MODELS, "impute_values.csv"), index=False)

    reloaded = joblib.load(os.path.join(MODELS, "delay_model.pkl"))
    pos = list(final.classes_).index(1)
    p_mem = final.predict_proba(X_test)[:, pos]
    p_reload = reloaded.predict_proba(X_test)[:, pos]
    diff = float(np.max(np.abs(p_mem - p_reload)))

    rows = [
        ("test_rows_scored", len(X_test)),
        ("reload_max_abs_probability_diff", round(diff, 12)),
        ("threshold", threshold),
    ]
    pd.DataFrame({"metric": [m for m, _ in rows],
                  "value": pd.Series([v for _, v in rows], dtype=object)}
                 ).to_csv(os.path.join(OUTPUTS, "part_d_reload_check.csv"), index=False)

    print("package_model complete:",
          f"test_rows={len(X_test)}, reload_diff={diff:.2e}, threshold={threshold}")


if __name__ == "__main__":
    main()

"""Part B: regression on delivery_hours.

Run:  python part_b_regression.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
from shipment_prep import FEATURES  # noqa: E402

DATA = os.path.join(HERE, "data")
OUTPUTS = os.path.join(HERE, "outputs")
os.makedirs(OUTPUTS, exist_ok=True)

TREE_DEPTHS = [1, 2, 3, 4, 6, 8, 12, None]
LEARNING_RATES = [0.01, 0.1, 1.0]


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def score_all(y_true, y_pred):
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "MSE": round(mse, 4),
        "RMSE": round(float(np.sqrt(mse)), 4),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
    }


def main():
    modelling = pd.read_csv(os.path.join(DATA, "modelling_table.csv"))
    X = modelling[FEATURES]
    y = modelling["delivery_hours"]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    assert (len(X_train), len(X_val), len(X_test)) == (840, 180, 180)

    write_scaling_checks(X_train, X_test)

    metric_rows = []

    # Task 3: mean baseline on validation.
    baseline_pred = np.full(len(y_val), y_train.mean())
    metric_rows.append({"model": "mean_baseline", "split": "val", **score_all(y_val, baseline_pred)})

    # Task 4: tree-depth diagnostic -> best depth.
    best_depth = write_tree_depth(X_train, y_train, X_val, y_val)

    # Task 5: four candidate models scored on validation.
    candidates = {
        "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    }
    fitted = {}
    val_rmse = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        pred = model.predict(X_val)
        val_rmse[name] = rmse(y_val, pred)
        metric_rows.append({"model": name, "split": "val", **score_all(y_val, pred)})

    # Task 6: learning-rate experiment.
    write_learning_rate(X_train, y_train, X_val, y_val)

    # Task 7: cross-validated comparison of the two best, then final test score.
    two_best = sorted(val_rmse, key=val_rmse.get)[:2]
    write_cv(two_best, candidates, X_train, X_val, y_train, y_val)
    print("validation RMSE:", {k: round(v, 4) for k, v in val_rmse.items()})

    final_name = min(val_rmse, key=val_rmse.get)
    final_pred = fitted[final_name].predict(X_test)
    metric_rows.append({"model": final_name, "split": "test", **score_all(y_test, final_pred)})

    pd.DataFrame(metric_rows).to_csv(os.path.join(OUTPUTS, "part_b_metrics.csv"), index=False)

    print("Part B complete:", f"best_depth={best_depth}, final={final_name}, "
          f"test_R2={metric_rows[-1]['R2']}")


def write_scaling_checks(X_train, X_test):
    scaler = StandardScaler().fit(X_train)
    train_scaled = scaler.transform(X_train)
    test_scaled = scaler.transform(X_test)
    rows = [
        ("train_rows", len(X_train)),
        ("val_rows", 180),
        ("test_rows", len(X_test)),
        ("train_scaled_max_abs_mean", round(float(np.max(np.abs(train_scaled.mean(axis=0)))), 12)),
        ("test_scaled_max_abs_mean", round(float(np.max(np.abs(test_scaled.mean(axis=0)))), 4)),
    ]
    pd.DataFrame({"metric": [m for m, _ in rows],
                  "value": pd.Series([v for _, v in rows], dtype=object)}
                 ).to_csv(os.path.join(OUTPUTS, "part_b_checks.csv"), index=False)


def write_tree_depth(X_train, y_train, X_val, y_val):
    rows = []
    for d in TREE_DEPTHS:
        tree = DecisionTreeRegressor(max_depth=d, random_state=42).fit(X_train, y_train)
        tr = rmse(y_train, tree.predict(X_train))
        vr = rmse(y_val, tree.predict(X_val))
        rows.append({
            "max_depth": "unlimited" if d is None else d,
            "_sort": np.inf if d is None else d,
            "train_rmse": tr,
            "val_rmse": vr,
        })
    best = min(rows, key=lambda r: (r["val_rmse"], r["_sort"]))
    best_depth = None if best["max_depth"] == "unlimited" else int(best["max_depth"])

    out = pd.DataFrame([{k: (round(v, 4) if isinstance(v, float) and k != "_sort" else v)
                         for k, v in r.items() if k != "_sort"} for r in rows])
    out.to_csv(os.path.join(OUTPUTS, "part_b_tree_depth.csv"), index=False)
    return best_depth


def write_learning_rate(X_train, y_train, X_val, y_val):
    rows = []
    for lr in LEARNING_RATES:
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        rows.append({
            "learning_rate": lr,
            "val_rmse": round(rmse(y_val, pred), 4),
            "val_r2": round(float(r2_score(y_val, pred)), 4),
        })
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUTS, "part_b_gb_learning_rate.csv"), index=False)


def write_cv(two_best, candidates, X_train, X_val, y_train, y_val):
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for name in two_best:
        # cross_val_score clones the estimator per fold, so passing the fitted
        # candidate simply reuses its hyperparameters.
        scores = cross_val_score(candidates[name], X_trainval, y_trainval, cv=kf)
        rows.append({
            "model": name,
            "mean_r2": round(float(scores.mean()), 4),
            "std_r2": round(float(scores.std()), 4),
        })
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUTS, "part_b_cv.csv"), index=False)


if __name__ == "__main__":
    main()

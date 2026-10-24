"""Part C: delay classifier, feature selection, tuning, and the alert threshold.

Run:  python part_c_classification.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
from shipment_prep import FEATURES, NOISE_COLUMNS  # noqa: E402

DATA = os.path.join(HERE, "data")
OUTPUTS = os.path.join(HERE, "outputs")
os.makedirs(OUTPUTS, exist_ok=True)

TREE_DEPTHS = [2, 3, 4, 5, 6, 8, 10]
SELECT_KS = [5, 7, 10, 12, 17]
FINAL_K = 7
PARAM_GRID = {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10]}
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TIE_BREAK = ["default_forest_same_7_features", "grid_search", "randomized_search"]


def split_data(modelling):
    X = modelling[FEATURES]
    y = modelling["is_delayed"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test


def cm_row(name, y_true, y_pred, precision_override=None):
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel())
    prec = precision_override if precision_override is not None else precision_score(y_true, y_pred, zero_division=0)
    return {
        "model": name, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def main():
    modelling = pd.read_csv(os.path.join(DATA, "modelling_table.csv"))
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(modelling)
    assert (len(X_train), len(X_val), len(X_test)) == (840, 180, 180)
    assert (int(y_train.sum()), int(y_val.sum()), int(y_test.sum())) == (195, 42, 41)

    best_depth = write_models(X_train, X_val, y_train, y_val)
    write_feature_scores(X_train, y_train)
    write_selectk(X_train, X_val, y_train, y_val)
    tuning = write_tuning(X_train, X_val, y_train, y_val)

    winner = pick_winner(tuning)
    final_model = build_final_model(tuning[winner]["params"]).fit(X_train, y_train)

    threshold = write_thresholds(final_model, X_val, y_val)
    write_final_test(final_model, X_test, y_test, threshold)
    write_final_config(tuning[winner]["params"], winner, threshold)

    print("Part C complete:", f"tree_depth={best_depth}, winner={winner}, threshold={threshold}")


def write_models(X_train, X_val, y_train, y_val):
    rows = []

    # Blind baseline: everything predicted on-time.
    blind_pred = np.zeros(len(y_val), dtype=int)
    rows.append(cm_row("blind_all_on_time", y_val, blind_pred, precision_override=0))

    logreg = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())]).fit(X_train, y_train)
    rows.append(cm_row("logistic_regression", y_val, logreg.predict(X_val)))

    best_depth, best_f1 = None, -1.0
    for d in TREE_DEPTHS:
        clf = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42).fit(X_train, y_train)
        f1 = f1_score(y_val, clf.predict(X_val), zero_division=0)
        if f1 > best_f1:
            best_f1, best_depth = f1, d
    best_tree = DecisionTreeClassifier(criterion="gini", max_depth=best_depth, random_state=42).fit(X_train, y_train)
    rows.append(cm_row(f"decision_tree_depth_{best_depth}", y_val, best_tree.predict(X_val)))

    forest = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    rows.append(cm_row("random_forest_default", y_val, forest.predict(X_val)))

    pd.DataFrame(rows).to_csv(os.path.join(OUTPUTS, "part_c_models.csv"), index=False)
    return best_depth


def write_feature_scores(X_train, y_train):
    forest = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    selector = SelectKBest(f_classif, k="all").fit(X_train, y_train)
    df = pd.DataFrame({
        "feature": FEATURES,
        "f_score": selector.scores_,
        "rf_importance": forest.feature_importances_,
    })
    df["f_rank"] = df["f_score"].rank(ascending=False, method="first").astype(int)
    df["is_noise"] = df["feature"].isin(NOISE_COLUMNS)
    df = df.sort_values("f_rank").reset_index(drop=True)
    df["f_score"] = df["f_score"].round(4)
    df["rf_importance"] = df["rf_importance"].round(4)
    df = df[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
    df.to_csv(os.path.join(OUTPUTS, "part_c_feature_scores.csv"), index=False)


def write_selectk(X_train, X_val, y_train, y_val):
    rows = []
    for k in SELECT_KS:
        pipe = Pipeline([
            ("selector", SelectKBest(f_classif, k=k)),
            ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]).fit(X_train, y_train)
        rows.append({"k": k, "val_f1": round(float(f1_score(y_val, pipe.predict(X_val), zero_division=0)), 4)})
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUTS, "part_c_selectk.csv"), index=False)


def write_tuning(X_train, X_val, y_train, y_val):
    selector = SelectKBest(f_classif, k=FINAL_K).fit(X_train, y_train)
    X_train_sel = selector.transform(X_train)
    X_val_sel = selector.transform(X_val)

    grid = GridSearchCV(RandomForestClassifier(random_state=42), PARAM_GRID, cv=5).fit(X_train_sel, y_train)
    rand = RandomizedSearchCV(RandomForestClassifier(random_state=42), PARAM_GRID,
                              n_iter=9, cv=5, random_state=42).fit(X_train_sel, y_train)
    default_rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_sel, y_train)

    def val_f1(model):
        return round(float(f1_score(y_val, model.predict(X_val_sel), zero_division=0)), 4)

    tuning = {
        "grid_search": {
            "candidates": 27, "cv_folds": 5, "total_fits": 135,
            "best_params": grid.best_params_, "best_cv_accuracy": round(float(grid.best_score_), 4),
            "val_f1": val_f1(grid.best_estimator_), "params": grid.best_params_,
        },
        "randomized_search": {
            "candidates": 9, "cv_folds": 5, "total_fits": 45,
            "best_params": rand.best_params_, "best_cv_accuracy": round(float(rand.best_score_), 4),
            "val_f1": val_f1(rand.best_estimator_), "params": rand.best_params_,
        },
        "default_forest_same_7_features": {
            "candidates": 1, "cv_folds": 0, "total_fits": 1,
            "best_params": {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1},
            "best_cv_accuracy": "",
            "val_f1": val_f1(default_rf),
            "params": {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1},
        },
    }
    order = ["grid_search", "randomized_search", "default_forest_same_7_features"]
    out = pd.DataFrame([{
        "search": name,
        "candidates": tuning[name]["candidates"],
        "cv_folds": tuning[name]["cv_folds"],
        "total_fits": tuning[name]["total_fits"],
        "best_params": str(tuning[name]["best_params"]),
        "best_cv_accuracy": tuning[name]["best_cv_accuracy"],
        "val_f1": tuning[name]["val_f1"],
    } for name in order])
    out.to_csv(os.path.join(OUTPUTS, "part_c_tuning.csv"), index=False)
    return tuning


def pick_winner(tuning):
    best_f1 = max(info["val_f1"] for info in tuning.values())
    for name in TIE_BREAK:
        if abs(tuning[name]["val_f1"] - best_f1) < 1e-12:
            return name
    return max(tuning, key=lambda n: tuning[n]["val_f1"])


def build_final_model(params):
    return Pipeline([
        ("selector", SelectKBest(f_classif, k=FINAL_K)),
        ("model", RandomForestClassifier(random_state=42, **params)),
    ])


def write_thresholds(final_model, X_val, y_val):
    pos = list(final_model.classes_).index(1)
    proba = final_model.predict_proba(X_val)[:, pos]
    rows = []
    for t in THRESHOLDS:
        pred = (proba >= t).astype(int)
        rows.append(cm_row(str(t), y_val, pred))
    out = pd.DataFrame(rows).rename(columns={"model": "threshold"})
    out.to_csv(os.path.join(OUTPUTS, "part_c_thresholds.csv"), index=False)

    eligible = [t for t, r in zip(THRESHOLDS, rows) if r["recall"] >= 0.80]
    return max(eligible) if eligible else 0.1


def write_final_test(final_model, X_test, y_test, threshold):
    pos = list(final_model.classes_).index(1)
    proba = final_model.predict_proba(X_test)[:, pos]
    pred = (proba >= threshold).astype(int)
    row = cm_row(str(threshold), y_test, pred)
    out = pd.DataFrame([row]).rename(columns={"model": "threshold"})
    out.to_csv(os.path.join(OUTPUTS, "part_c_final_test.csv"), index=False)


def write_final_config(params, winner, threshold):
    max_depth = params.get("max_depth", None)
    row = {
        "selector_k": FINAL_K,
        "n_estimators": params.get("n_estimators", 100),
        "max_depth": "unlimited" if max_depth is None else max_depth,
        "min_samples_leaf": params.get("min_samples_leaf", 1),
        "threshold": threshold,
        "chosen_from": winner,
    }
    pd.DataFrame([row]).to_csv(os.path.join(OUTPUTS, "part_c_final_config.csv"), index=False)


if __name__ == "__main__":
    main()

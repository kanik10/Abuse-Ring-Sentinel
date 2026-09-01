"""
Day 3 — classifier.py

With only 70 clusters (20 positive / 50 negative), a single 80/20
train/test split would be unstable — the test set would have ~4 positive
examples, and the reported precision/recall would swing wildly depending
on which 4 happened to land there. Instead: stratified 5-fold CV, with
every cluster scored exactly once by a model that never saw it during
training (out-of-fold predictions), then metrics computed on those.

This also compares a plain logistic regression against a lightly-
regularized XGBoost — with this little data, a high-capacity model isn't
automatically the better choice, and it's worth checking rather than
assuming.

ground_truth.csv is used here ONLY as the training/evaluation label. This
is different from Day 2, where using it inside community detection would
have been leakage — here, supervised learning by definition trains on
labels. What must still never happen is ground truth leaking into the
*features themselves* (Day 3's feature_engineering.py doesn't touch it).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              average_precision_score, brier_score_loss)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DATA_DIR = "day1_data"


def build_labels(features: pd.DataFrame, clusters: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.Series:
    merged = clusters.merge(ground_truth, on="account_id", how="left")
    majority = merged.groupby("cluster_id")["is_ring_member"].mean()  # fraction of members that are ring
    label = (majority > 0.5).astype(int)
    return features["cluster_id"].map(label)


def evaluate(y_true, y_prob, threshold=0.5, name=""):
    y_pred = (y_prob >= threshold).astype(int)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    ap = average_precision_score(y_true, y_prob)
    print(f"{name:20s}  precision={p:.3f}  recall={r:.3f}  f1={f1:.3f}  PR-AUC={ap:.3f}")
    return p, r, f1, ap


def calibration_report(y_true, y_prob, name="", n_bins=5):
    """Are the risk scores trustworthy as probabilities, not just as a
    ranking? A Brier score near 0 and bins where predicted probability
    roughly matches actual positive rate mean yes -- important before Day 4
    picks a threshold based on assumed real-world likelihoods."""
    brier = brier_score_loss(y_true, y_prob)
    print(f"\n{name} -- Brier score: {brier:.4f}  (0=perfect, 0.25=uninformative, 1=worst)")
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        print(f"  predicted [{bins[b]:.1f}-{bins[b+1]:.1f}): n={mask.sum():3d}  "
              f"avg predicted={y_prob[mask].mean():.3f}  actual positive rate={y_true[mask].mean():.3f}")


def main():
    features = pd.read_csv("cluster_features.csv")
    clusters = pd.read_csv("clusters.csv")
    ground_truth = pd.read_csv(f"{DATA_DIR}/ground_truth.csv")

    y = build_labels(features, clusters, ground_truth).values
    X = features.drop(columns=["cluster_id"]).values
    feature_names = features.drop(columns=["cluster_id"]).columns.tolist()

    print(f"Clusters: {len(y)}  |  positive (ring): {y.sum()}  |  negative: {len(y) - y.sum()}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_logreg = np.zeros(len(y), dtype=float)
    oof_xgb = np.zeros(len(y), dtype=float)

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
        logreg.fit(X_train, y_train)
        oof_logreg[test_idx] = logreg.predict_proba(X_test)[:, 1]

        xgbc = XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42,
        )
        xgbc.fit(X_train, y_train)
        oof_xgb[test_idx] = xgbc.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 70)
    print("DAY 3 SUMMARY — out-of-fold metrics (5-fold stratified CV, threshold=0.5)")
    print("=" * 70)
    print("Day 2 baseline (flag everyone, no scoring): precision=0.738  recall=0.998")
    evaluate(y, oof_logreg, name="Logistic Regression")
    evaluate(y, oof_xgb, name="XGBoost (max_depth=3)")

    # --- Ablation: drop order-amount features, which nearly separate the
    # classes by themselves as an artifact of how Day 1 assigned order
    # amounts (ring accounts drawing from a distinctly lower mean). This
    # checks how much signal the GRAPH-STRUCTURAL features carry on their
    # own, without leaning on that shortcut. ---
    drop_cols = ["avg_order_amount", "order_amount_cv"]
    keep_idx = [i for i, n in enumerate(feature_names) if n not in drop_cols]
    X_structural = X[:, keep_idx]
    structural_names = [feature_names[i] for i in keep_idx]

    oof_logreg_struct = np.zeros(len(y), dtype=float)
    oof_xgb_struct = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in skf.split(X_structural, y):
        X_train, X_test = X_structural[train_idx], X_structural[test_idx]
        y_train = y[train_idx]

        logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
        logreg.fit(X_train, y_train)
        oof_logreg_struct[test_idx] = logreg.predict_proba(X_test)[:, 1]

        xgbc = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8,
                              eval_metric="logloss", random_state=42)
        xgbc.fit(X_train, y_train)
        oof_xgb_struct[test_idx] = xgbc.predict_proba(X_test)[:, 1]

    print("\n" + "-" * 70)
    print("ABLATION — structural + temporal features only (no order-amount features)")
    print("-" * 70)
    evaluate(y, oof_logreg_struct, name="LogReg (structural)")
    evaluate(y, oof_xgb_struct, name="XGBoost (structural)")

    # --- Second ablation: creation_span_days/creation_std_days are ALSO a
    # near-deterministic artifact of Day 1 (ring_window_days is capped at
    # 5-25 days by construction; coincidental groups are drawn from a
    # 2-year window with no such cap). Drop those too, leaving only
    # features that reflect actual resource-sharing topology. ---
    pure_drop = drop_cols + ["creation_span_days", "creation_std_days"]
    keep_idx2 = [i for i, n in enumerate(feature_names) if n not in pure_drop]
    X_pure = X[:, keep_idx2]
    pure_names = [feature_names[i] for i in keep_idx2]
    print(f"\nPure graph-topology features used: {pure_names}")

    oof_logreg_pure = np.zeros(len(y), dtype=float)
    oof_xgb_pure = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in skf.split(X_pure, y):
        X_train, X_test = X_pure[train_idx], X_pure[test_idx]
        y_train = y[train_idx]

        logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
        logreg.fit(X_train, y_train)
        oof_logreg_pure[test_idx] = logreg.predict_proba(X_test)[:, 1]

        xgbc = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8,
                              eval_metric="logloss", random_state=42)
        xgbc.fit(X_train, y_train)
        oof_xgb_pure[test_idx] = xgbc.predict_proba(X_test)[:, 1]

    print("\n" + "-" * 70)
    print("ABLATION 2 — pure graph-topology features only (no timing, no order amount)")
    print("-" * 70)
    evaluate(y, oof_logreg_pure, name="LogReg (pure graph)")
    evaluate(y, oof_xgb_pure, name="XGBoost (pure graph)")

    print("\n" + "=" * 70)
    print("CALIBRATION CHECK — are the risk scores trustworthy as probabilities?")
    print("=" * 70)
    calibration_report(y, oof_logreg, name="Logistic Regression (full)")
    calibration_report(y, oof_xgb, name="XGBoost (full)")
    calibration_report(y, oof_logreg_pure, name="Logistic Regression (pure graph)")
    calibration_report(y, oof_xgb_pure, name="XGBoost (pure graph)")

    # refit each model on ALL data to inspect feature signal (not for scoring)
    logreg_full = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    logreg_full.fit(X, y)
    coefs = logreg_full.named_steps["logisticregression"].coef_[0]

    xgb_full = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8,
                              eval_metric="logloss", random_state=42)
    xgb_full.fit(X, y)

    print("\nFeature signal (standardized logistic regression coefficients):")
    for name, c in sorted(zip(feature_names, coefs), key=lambda t: -abs(t[1])):
        print(f"  {name:26s} {c:+.3f}")

    print("\nFeature signal (XGBoost importances):")
    for name, imp in sorted(zip(feature_names, xgb_full.feature_importances_), key=lambda t: -t[1]):
        print(f"  {name:26s} {imp:.3f}")

    out = features.copy()
    out["y_true_is_ring"] = y
    out["oof_prob_logreg_full"] = oof_logreg
    out["oof_prob_xgb_full"] = oof_xgb
    out["oof_prob_logreg_structural"] = oof_logreg_struct
    out["oof_prob_xgb_structural"] = oof_xgb_struct
    out["oof_prob_logreg_pure_graph"] = oof_logreg_pure
    out["oof_prob_xgb_pure_graph"] = oof_xgb_pure
    out.to_csv("cluster_predictions.csv", index=False)
    print("\nSaved cluster_predictions.csv for Day 4 threshold/cost analysis.")


if __name__ == "__main__":
    main()
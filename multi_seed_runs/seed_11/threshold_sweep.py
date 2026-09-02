"""
Day 4 Phase 2 — threshold_sweep.py

Cost model (per-account, not per-cluster — see conversation for reasoning):

  cost(threshold) = FP_multiplier * avg_order_value * (non-ring accounts
                     wrongly caught in FLAGGED clusters)
                   + sum(order value of ring accounts sitting in UNFLAGGED
                     clusters)

Computed per-account, not per-cluster, so a bridge-linked innocent
bystander sitting inside an otherwise-correctly-flagged ring cluster still
counts as a real FP cost — and a ring member sitting inside an otherwise-
correctly-cleared coincidental cluster (via a bridge) still counts as a
real missed-fraud cost. This is more precise than assuming every cluster
is internally pure, which bridges deliberately made untrue for a few of
them.

FP_multiplier is swept, not assumed, because there's no data-grounded way
to know the real operational cost of clearing a false positive — see
Phase 1 discussion.
"""

import numpy as np
import pandas as pd

DATA_DIR = "day1_data"
FP_MULTIPLIERS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
MODEL_COLUMNS = {
    "LogReg (pure graph)": "oof_prob_logreg_pure_graph",
    "XGBoost (pure graph)": "oof_prob_xgb_pure_graph",
    "LogReg (full)": "oof_prob_logreg_full",
    "XGBoost (full)": "oof_prob_xgb_full",
}


def build_cluster_cost_inputs(clusters: pd.DataFrame, ground_truth: pd.DataFrame,
                               orders: pd.DataFrame) -> pd.DataFrame:
    """Per cluster: value of ring-member orders (cost if missed), and count
    of non-ring members (cost multiplier target if wrongly flagged)."""
    merged = clusters.merge(ground_truth, on="account_id", how="left")
    order_value = orders.groupby("account_id")["amount"].sum()
    merged["order_value"] = merged["account_id"].map(order_value).fillna(0.0)

    ring_value = (merged[merged.is_ring_member]
                  .groupby("cluster_id")["order_value"].sum()
                  .rename("ring_value_if_missed"))
    non_ring_count = (merged[~merged.is_ring_member]
                       .groupby("cluster_id").size()
                       .rename("non_ring_accounts_if_flagged"))

    out = pd.DataFrame({"cluster_id": clusters.cluster_id.unique()}).set_index("cluster_id")
    out = out.join(ring_value).join(non_ring_count).fillna(0.0).reset_index()
    return out


def sweep_thresholds(y_prob: np.ndarray, cost_inputs: pd.DataFrame,
                      fp_multiplier: float, avg_order_value: float) -> pd.DataFrame:
    thresholds = sorted(set(y_prob.tolist()) | {0.0, 1.0001})
    rows = []
    for t in thresholds:
        flagged = y_prob >= t
        fp_cost = (fp_multiplier * avg_order_value
                   * cost_inputs.loc[flagged, "non_ring_accounts_if_flagged"].sum())
        fn_cost = cost_inputs.loc[~flagged, "ring_value_if_missed"].sum()
        rows.append({"threshold": t, "n_flagged": int(flagged.sum()),
                     "fp_cost": fp_cost, "fn_cost": fn_cost, "total_cost": fp_cost + fn_cost})
    return pd.DataFrame(rows)


def main():
    features = pd.read_csv("cluster_features.csv")
    predictions = pd.read_csv("cluster_predictions.csv")
    clusters = pd.read_csv("clusters.csv")
    ground_truth = pd.read_csv(f"{DATA_DIR}/ground_truth.csv")
    orders = pd.read_csv(f"{DATA_DIR}/orders.csv")

    avg_order_value = orders["amount"].mean()
    cost_inputs = build_cluster_cost_inputs(clusters, ground_truth, orders)
    cost_inputs = cost_inputs.set_index("cluster_id").loc[predictions.cluster_id].reset_index()

    print(f"Reference avg order value: Rs.{avg_order_value:.2f}")
    print(f"Total ring value at risk if nothing were caught: "
          f"Rs.{cost_inputs.ring_value_if_missed.sum():,.2f}\n")

    best_overall = []
    for model_name, col in MODEL_COLUMNS.items():
        y_prob = predictions[col].values
        print("=" * 78)
        print(model_name)
        print("=" * 78)
        for mult in FP_MULTIPLIERS:
            sweep = sweep_thresholds(y_prob, cost_inputs, mult, avg_order_value)
            best = sweep.loc[sweep.total_cost.idxmin()]
            print(f"  FP multiplier {mult:>4}x avg order value: "
                  f"best threshold={best.threshold:.3f}  "
                  f"n_flagged={int(best.n_flagged):3d}  "
                  f"total_cost=Rs.{best.total_cost:,.0f}  "
                  f"(fp_cost=Rs.{best.fp_cost:,.0f}, fn_cost=Rs.{best.fn_cost:,.0f})")
            best_overall.append({"model": model_name, "fp_multiplier": mult,
                                  "best_threshold": best.threshold,
                                  "n_flagged": int(best.n_flagged),
                                  "total_cost": best.total_cost,
                                  "fp_cost": best.fp_cost, "fn_cost": best.fn_cost})
        print()

    pd.DataFrame(best_overall).to_csv("threshold_sweep_results.csv", index=False)
    print("Saved threshold_sweep_results.csv")


if __name__ == "__main__":
    main()

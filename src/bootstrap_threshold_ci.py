"""
bootstrap_threshold_ci.py -- Nonparametric Bootstrap Confidence Interval Engine

Runs a nonparametric percentile bootstrap (B=10,000 resamples) on the Champion
Model (pure graph + referral) at CHOSEN_THRESHOLD (~0.1333).
Also re-sweeps the cost function per resample to measure threshold stability.
Can be executed standalone or imported by final_threshold_report.py.

--- Why bootstrap, not a normal-approximation CI (see conversation) ---
precision/recall are proportions bounded on [0,1] and sit at or near the
boundary here (precision = 1.000 with FP=0) -- a Wald/normal CI on that is
degenerate. total_cost is a composite of a count-based FP term and a
continuous ring-order-value FN term, with no closed-form CI at all. A
nonparametric percentile bootstrap handles all three with one method,
which matters because the real target is the *decision* (which threshold
minimizes cost), not any one metric in isolation.

--- What gets resampled, and why ---
The resampling unit is the CLUSTER (n varies by run), with replacement,
because that is both the unit the classifier scores (one oof_prob per
cluster) and the unit the cost function aggregates over
(ring_value_if_missed / non_ring_accounts_if_flagged are already
cluster-level sums). This is a different question from the train/ring-ID-
split leakage concern elsewhere in the pipeline -- that's about leakage
during *training*; this is about sampling uncertainty in the already-
computed out-of-fold predictions, given there are only ~60 of them to
evaluate on.

Each bootstrap resample produces two things:
  (a) precision / recall / total_cost AT THE CURRENTLY LOCKED THRESHOLD
      -- how uncertain is the headline precision/recall/cost figure?
  (b) that resample's OWN cost-optimal threshold (full re-sweep) -- how
      stable is the choice of the locked threshold itself, i.e. would a
      different draw of clusters have picked a meaningfully different
      threshold?

B=10,000 resamples: at this N this is computationally trivial (no reason
to cut corners at 1-2k), and 10k gives stable 2.5th/97.5th percentiles
for a percentile bootstrap.
"""

import numpy as np
import pandas as pd

from threshold_sweep import build_cluster_cost_inputs, sweep_thresholds
from threshold_config import CHOSEN_THRESHOLD  # single source of truth -- reads
                                               # pooled_threshold_selection_summary.json.

DATA_DIR = "day1_data"
# Must match risk_scoring.py's CHAMPION_FEATURE_COLS (pure graph + referral) --
# that is the model actually shipped as final_model.joblib. Bootstrapping the
# old pure-graph-only column would validate a threshold for a model that
# isn't the one in production.
CHOSEN_COLUMN = "oof_prob_logreg_pure_graph_referral"
DEFAULT_FP_MULTIPLIER = 1.0  # matches the "1x avg order value" headline assumption in final_report.md

N_BOOTSTRAP = 10_000
SEED = 42


def load_inputs():
    """Same loading + cost-input construction as threshold_sweep.py's
    main(), factored out so this script and threshold_sweep.py can't
    silently drift apart on how cost_inputs is built."""
    predictions = pd.read_csv("cluster_predictions.csv")
    clusters = pd.read_csv("clusters.csv")
    ground_truth = pd.read_csv(f"{DATA_DIR}/ground_truth.csv")
    orders = pd.read_csv(f"{DATA_DIR}/orders.csv")

    avg_order_value = orders["amount"].mean()
    cost_inputs = build_cluster_cost_inputs(clusters, ground_truth, orders)
    cost_inputs = cost_inputs.set_index("cluster_id").loc[predictions.cluster_id].reset_index()
    return predictions, cost_inputs, avg_order_value


def _evaluate_resample(yp: np.ndarray, yt: np.ndarray, ring_val: np.ndarray, non_ring_n: np.ndarray,
                        fp_multiplier: float, avg_order_value: float, t_fixed: float) -> dict:
    """Core per-resample computation, shared by bootstrap() and the
    identity-resample correctness check in main(). Not itself the
    resampling step -- caller passes in whatever arrays it wants evaluated
    (bootstrap-resampled, or the original data unchanged)."""
    flagged_fixed = yp >= t_fixed
    tp = int(((flagged_fixed) & (yt == 1)).sum())
    fp = int(((flagged_fixed) & (yt == 0)).sum())
    fn = int(((~flagged_fixed) & (yt == 1)).sum())
    precision_fixed = tp / (tp + fp) if (tp + fp) else 0.0
    recall_fixed = tp / (tp + fn) if (tp + fn) else 0.0
    fp_cost_fixed = fp_multiplier * avg_order_value * non_ring_n[flagged_fixed].sum()
    fn_cost_fixed = ring_val[~flagged_fixed].sum()
    total_cost_fixed = fp_cost_fixed + fn_cost_fixed

    candidates = np.unique(np.concatenate([yp, [0.0, 1.0001]]))
    best_cost = np.inf
    best_t = None
    best_n_flagged = None
    for t in candidates:
        flagged = yp >= t
        fp_cost = fp_multiplier * avg_order_value * non_ring_n[flagged].sum()
        fn_cost = ring_val[~flagged].sum()
        total_cost = fp_cost + fn_cost
        if total_cost < best_cost:
            best_cost = total_cost
            best_t = t
            best_n_flagged = int(flagged.sum())

    return {
        "precision_fixed": precision_fixed, "recall_fixed": recall_fixed,
        "total_cost_fixed": total_cost_fixed, "n_flagged_fixed": int(flagged_fixed.sum()),
        "best_threshold": best_t, "best_total_cost": best_cost,
        "best_n_flagged": best_n_flagged,
    }


def bootstrap(y_prob: np.ndarray, y_true: np.ndarray,
              ring_value_if_missed: np.ndarray, non_ring_accounts_if_flagged: np.ndarray,
              fp_multiplier: float, avg_order_value: float, t_fixed: float,
              n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> pd.DataFrame:
    n = len(y_prob)
    assert len(y_true) == n and len(ring_value_if_missed) == n and len(non_ring_accounts_if_flagged) == n
    rng = np.random.default_rng(seed)
    rows = []

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        row = _evaluate_resample(y_prob[idx], y_true[idx],
                                  ring_value_if_missed[idx], non_ring_accounts_if_flagged[idx],
                                  fp_multiplier, avg_order_value, t_fixed)
        row["b"] = b
        rows.append(row)

    return pd.DataFrame(rows)


def percentile_ci(values: pd.Series, lo=2.5, hi=97.5) -> tuple[float, float]:
    return float(np.percentile(values, lo)), float(np.percentile(values, hi))


def main():
    predictions, cost_inputs, avg_order_value = load_inputs()
    y_prob = predictions[CHOSEN_COLUMN].values
    y_true = predictions["y_true_is_ring"].values
    ring_value_if_missed = cost_inputs["ring_value_if_missed"].values
    non_ring_accounts_if_flagged = cost_inputs["non_ring_accounts_if_flagged"].values
    n = len(y_prob)

    print(f"Model: {CHOSEN_COLUMN}")
    print(f"N clusters: {n}  (this is the whole population being bootstrapped)")
    print(f"FP multiplier: {DEFAULT_FP_MULTIPLIER}x avg order value (Rs.{avg_order_value:.2f})")
    print(f"Fixed/locked threshold under test: {CHOSEN_THRESHOLD}")
    print(f"Bootstrap resamples: {N_BOOTSTRAP}\n")

    real_sweep = sweep_thresholds(y_prob, cost_inputs, DEFAULT_FP_MULTIPLIER, avg_order_value)
    real_best = real_sweep.loc[real_sweep.total_cost.idxmin()]
    print(f"Reference (non-bootstrap) best threshold from threshold_sweep.py's own "
          f"function: {real_best.threshold:.4f}  (total_cost=Rs.{real_best.total_cost:,.0f})")

    identity_check = _evaluate_resample(y_prob, y_true, ring_value_if_missed, non_ring_accounts_if_flagged,
                                         DEFAULT_FP_MULTIPLIER, avg_order_value, CHOSEN_THRESHOLD)
    match = np.isclose(identity_check["best_threshold"], real_best.threshold) and \
        np.isclose(identity_check["best_total_cost"], real_best.total_cost)
    print(f"Correctness check -- this engine's own sweep on the UN-resampled data: "
          f"threshold={identity_check['best_threshold']:.4f}, "
          f"total_cost=Rs.{identity_check['best_total_cost']:,.0f}  "
          f"[{'MATCHES' if match else 'MISMATCH -- BUG'} reference above]")
    if not match:
        raise AssertionError(
            "Bootstrap engine's re-implemented sweep disagrees with threshold_sweep.py's "
            "reference sweep_thresholds() on the same (un-resampled) data -- fix before "
            "trusting any bootstrap output below."
        )

    results = bootstrap(y_prob, y_true, ring_value_if_missed, non_ring_accounts_if_flagged,
                         DEFAULT_FP_MULTIPLIER, avg_order_value, CHOSEN_THRESHOLD)

    print(f"\n{'='*70}")
    print("(a) Metrics AT the locked threshold, across bootstrap resamples")
    print(f"{'='*70}")
    for col, label in [("precision_fixed", "Precision"), ("recall_fixed", "Recall"),
                        ("total_cost_fixed", "Total cost (Rs.)")]:
        lo, hi = percentile_ci(results[col])
        median = float(results[col].median())
        if col == "total_cost_fixed":
            print(f"  {label:20s} median={median:>12,.0f}  95% CI=[{lo:>12,.0f}, {hi:>12,.0f}]")
        else:
            print(f"  {label:20s} median={median:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]")

    print(f"\n{'='*70}")
    print("(b) Distribution of each resample's OWN cost-optimal threshold")
    print(f"{'='*70}")
    bt = results["best_threshold"]
    print(f"  min={bt.min():.4f}  p25={bt.quantile(.25):.4f}  median={bt.median():.4f}  "
          f"p75={bt.quantile(.75):.4f}  max={bt.max():.4f}")
    n_unique = bt.nunique()
    print(f"  {n_unique} distinct threshold values chosen across {N_BOOTSTRAP} resamples")
    close = ((bt - CHOSEN_THRESHOLD).abs() <= 0.05).mean()
    print(f"  {close*100:.1f}% of resamples had their own optimum within +/-0.05 of "
          f"the locked threshold ({CHOSEN_THRESHOLD:.4f})")

    results.to_csv("bootstrap_threshold_ci_results.csv", index=False)
    print(f"\nSaved bootstrap_threshold_ci_results.csv ({len(results)} rows, "
          f"one per bootstrap resample)")


if __name__ == "__main__":
    main()
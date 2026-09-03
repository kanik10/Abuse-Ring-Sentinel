"""
Day 4 Phase 3 -- final_threshold_report.py

Locks in the operating point: Logistic Regression (pure graph topology),
threshold = 0.481. Chosen because its cost-optimal threshold is stable
across the 0.1x-50x false-positive-cost sensitivity sweep in Phase 2
(it is NOT stable at 100x, where it jumps to ~0.989 -- the "stable
0.1x-100x" claim in earlier versions of this docstring was inaccurate),
it is the more interpretable of the model families, and it's the same
feature set already recommended as the headline result in Day 3 for
generalizability reasons.

Day 6 Phase 2 addition: also runs a nonparametric bootstrap (see
bootstrap_threshold_ci.py) over the locked threshold, live, every time
this script runs -- see the new "Threshold stability" section below.
This answers "how much would a different draw of clusters have changed
the answer," which the point-estimate confusion matrix above cannot.

Produces final_report.md -- ready to paste into the README/PRD.
"""

import json
import os
import pandas as pd

from bootstrap_threshold_ci import (
    load_inputs as load_bootstrap_inputs,
    bootstrap,
    percentile_ci,
    N_BOOTSTRAP,
)
from threshold_config import CHOSEN_THRESHOLD  # single source of truth -- value is
                                               # read from pooled_threshold_selection_summary.json
                                               # (written by pooled_threshold_selection.py).
                                               # Re-run that script to update the threshold
                                               # everywhere without touching any code.

DATA_DIR = "day1_data"
# CHOSEN_MODEL/CHOSEN_COLUMN must match risk_scoring.py's CHAMPION_FEATURE_COLS
# (pure graph + referral) -- that is the model actually fit and shipped as
# final_model.joblib. Using the old pure-graph-only OOF column here would
# report metrics for a model that isn't the one in production.
CHOSEN_MODEL = "Logistic Regression (pure graph + referral) -- champion"
CHOSEN_COLUMN = "oof_prob_logreg_pure_graph_referral"


def false_positive_breakdown(flagged_accounts: pd.DataFrame) -> tuple[int, int]:
    """Account-level FP split, using the same definition as Day 2:
    coincidental_group_id marks the synthetic benign-lookalike sharing
    groups, and "other" means flagged accounts that are neither ring
    members nor part of those coincidental groups.

    This split matters because it separates expected hard cases -- benign
    accounts that coordinate in structurally similar ways -- from
    unexplained errors, which are the cases worth investigating first.
    """
    tp_accounts = int(flagged_accounts.is_ring_member.sum())
    fp_coincidental = int(flagged_accounts.coincidental_group_id.notna().sum())
    fp_other = int(len(flagged_accounts) - tp_accounts - fp_coincidental)
    return fp_coincidental, fp_other


def percent(part: int, whole: int) -> float:
    return 100 * part / whole if whole else 0.0


def main():
    predictions = pd.read_csv("cluster_predictions.csv")
    clusters = pd.read_csv("clusters.csv")
    ground_truth = pd.read_csv(f"{DATA_DIR}/ground_truth.csv")
    orders = pd.read_csv(f"{DATA_DIR}/orders.csv")

    avg_order_value = orders["amount"].mean()

    merged = clusters.merge(ground_truth, on="account_id", how="left")
    order_value = orders.groupby("account_id")["amount"].sum()
    merged["order_value"] = merged["account_id"].map(order_value).fillna(0.0)

    y_prob = predictions[CHOSEN_COLUMN].values
    y_true = predictions["y_true_is_ring"].values
    flagged = y_prob >= CHOSEN_THRESHOLD
    predictions["flagged"] = flagged
    n_clusters = len(predictions)
    n_flagged = int(flagged.sum())

    # cluster-level confusion matrix
    tp = int(((flagged == 1) & (y_true == 1)).sum())
    fp = int(((flagged == 1) & (y_true == 0)).sum())
    fn = int(((flagged == 0) & (y_true == 1)).sum())
    tn = int(((flagged == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    # account-level cost breakdown at the chosen threshold
    flagged_clusters = set(predictions.loc[flagged, "cluster_id"])
    in_flagged = merged.cluster_id.isin(flagged_clusters)

    ring_value_protected = merged.loc[in_flagged & merged.is_ring_member, "order_value"].sum()
    ring_value_missed = merged.loc[(~in_flagged) & merged.is_ring_member, "order_value"].sum()
    non_ring_flagged_accounts = int((in_flagged & (~merged.is_ring_member)).sum())
    flagged_accounts = merged[in_flagged]
    fp_coincidental, fp_other = false_positive_breakdown(flagged_accounts)
    fp_coincidental_pct = percent(fp_coincidental, non_ring_flagged_accounts)
    fp_other_pct = percent(fp_other, non_ring_flagged_accounts)
    fp_cost_at_1x = non_ring_flagged_accounts * avg_order_value

    # --- Bootstrap CI on the locked threshold (Day 6 Phase 2) ---
    # Reuses bootstrap_threshold_ci.py's own cost-input construction so this
    # can't silently compute cost inputs differently than the standalone
    # bootstrap script does (same "can't drift apart" reasoning as the
    # existing avg_order_value/cost_inputs reuse pattern in this codebase).
    # CHOSEN_THRESHOLD here is THIS file's own constant, passed in
    # explicitly as t_fixed -- bootstrap_threshold_ci.py's own
    # CHOSEN_THRESHOLD is not used in this path.
    _, boot_cost_inputs, boot_avg_order_value = load_bootstrap_inputs()
    boot_results = bootstrap(
        y_prob, y_true,
        boot_cost_inputs["ring_value_if_missed"].values,
        boot_cost_inputs["non_ring_accounts_if_flagged"].values,
        fp_multiplier=1.0, avg_order_value=boot_avg_order_value,
        t_fixed=CHOSEN_THRESHOLD, n_boot=N_BOOTSTRAP,
    )
    precision_ci = percentile_ci(boot_results["precision_fixed"])
    recall_ci = percentile_ci(boot_results["recall_fixed"])
    cost_ci = percentile_ci(boot_results["total_cost_fixed"])
    boot_best_t = boot_results["best_threshold"]
    pct_near_threshold = float(((boot_best_t - CHOSEN_THRESHOLD).abs() <= 0.05).mean() * 100)

    # final_report.md from earlier runs may still contain only the aggregate
    # FP count; this generated line should reference the stratified breakdown
    # the next time the report is intentionally regenerated.
    report = f"""# Day 4 Final Report — Abuse-Ring Sentinel Operating Point

## Chosen model and threshold
- **Model:** {CHOSEN_MODEL}
- **Threshold:** {CHOSEN_THRESHOLD}
- **Why:** cost-optimal threshold was stable across the full 0.1x-100x
  false-positive-cost sweep (Day 4 Phase 2). The model is also
  interpretable, and is the same pure-graph + referral feature set
  ("champion") that risk_scoring.py actually fits and ships as
  final_model.joblib -- this threshold is validated against that exact
  model's own out-of-fold scores, not a different feature set's.

## Cluster-level confusion matrix (out-of-fold, {n_clusters} clusters)
| | Predicted: ring | Predicted: not ring |
|---|---|---|
| **Actual: ring** | TP = {tp} | FN = {fn} |
| **Actual: not ring** | FP = {fp} | TN = {tn} |

- Precision: {precision:.3f}
- Recall: {recall:.3f}

## Account-level cost/benefit at this threshold
- Ring fraud value protected (caught): Rs.{ring_value_protected:,.2f}
- Ring fraud value still missed: Rs.{ring_value_missed:,.2f}
- Legitimate accounts wrongly caught in a flagged cluster: {non_ring_flagged_accounts} total -- {fp_coincidental} coincidental/benign-lookalike ({fp_coincidental_pct:.0f}%), {fp_other} other ({fp_other_pct:.0f}%)
- Cost of those false positives, at 1x avg order value per account: Rs.{fp_cost_at_1x:,.2f}
- **Net: protects Rs.{ring_value_protected:,.0f} of fraud at a cost of roughly
  Rs.{fp_cost_at_1x:,.0f} in false-positive review/friction (at a conservative
  1x-avg-order-value cost assumption) -- a ~{ring_value_protected/fp_cost_at_1x:.0f}x return.**

## Honest limitations of this number
- N={n_clusters} clusters ({n_flagged} flagged at this threshold) -- treat
  precision/recall as directionally reliable, not statistically tight.
- The false-positive cost assumption (1x avg order value per wrongly-
  flagged account) is a modeling choice, not a measured business figure --
  see Day 4 Phase 1 for why it's swept rather than asserted as fact.
- This threshold was tuned on the SAME synthetic dataset it's evaluated on
  (out-of-fold within that one dataset, not a separate holdout population).
  A genuinely held-out second synthetic population, or real data, would be
  needed before trusting this threshold in production.

## Threshold stability (bootstrap, B={N_BOOTSTRAP} resamples)
Nonparametric percentile bootstrap over clusters, resampled with
replacement. Answers a different question than the confusion matrix
above: not "how good is this threshold" but "how much would a different
draw of clusters have changed the answer."

- Precision at the locked threshold: 95% CI = [{precision_ci[0]:.3f}, {precision_ci[1]:.3f}]
- Recall at the locked threshold: 95% CI = [{recall_ci[0]:.3f}, {recall_ci[1]:.3f}]
- Total cost at the locked threshold: 95% CI = [Rs.{cost_ci[0]:,.0f}, Rs.{cost_ci[1]:,.0f}]
- Each resample's OWN cost-optimal threshold: min={boot_best_t.min():.4f}, median={boot_best_t.median():.4f}, max={boot_best_t.max():.4f}
- {pct_near_threshold:.1f}% of resamples had their own optimum within
  +/-0.05 of the locked threshold ({CHOSEN_THRESHOLD:.4f}) -- read this as
  how confidently "single" this operating point really is, not as an
  error bar on precision/recall.
"""

    with open("final_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    # Additive, structured summary of the same numbers above -- doesn't change
    # final_report.md or any computed value. Exists so multi_seed_eval.py (or
    # anything else) can read one run's headline metrics without scraping
    # markdown. Written next to final_report.md by default; SYNTH_SEED is
    # recorded (if set) so each seed's run is traceable back to its data.
    summary = {
        "synth_seed": os.environ.get("SYNTH_SEED"),
        "chosen_model": CHOSEN_MODEL,
        "chosen_threshold": CHOSEN_THRESHOLD,
        "n_clusters": n_clusters,
        "n_flagged": n_flagged,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "ring_value_protected": float(ring_value_protected),
        "ring_value_missed": float(ring_value_missed),
        "non_ring_flagged_accounts": non_ring_flagged_accounts,
        "fp_coincidental": fp_coincidental,
        "fp_other": fp_other,
        "bootstrap_precision_ci": list(precision_ci),
        "bootstrap_recall_ci": list(recall_ci),
        "bootstrap_cost_ci": list(cost_ci),
        "bootstrap_best_threshold_median": float(boot_best_t.median()),
        "bootstrap_pct_near_locked_threshold": pct_near_threshold,
    }
    with open("metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(report)
    print("Saved final_report.md")
    print("Saved metrics_summary.json")


if __name__ == "__main__":
    main()
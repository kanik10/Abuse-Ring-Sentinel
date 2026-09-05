"""
pooled_threshold_selection.py

Primary fix for the flat-plateau / threshold-instability finding from the
multi-seed bootstrap audit (see conversation):

  PROBLEM: the locked threshold (0.4811) was derived from a single 63-cluster
  synthetic population, which is too small a sample to pin "the" cost-optimal
  threshold. The multi-seed bootstrap confirmed this: bootstrap_best_threshold
  median ranges from 0.133 to 0.836 across 15 seeds, because when the cost
  function is flat over a wide middle band, argmin picks an arbitrary edge of
  that plateau -- whichever candidate the sweep happens to touch first.

  FIX 1 (primary): pool all 15 seeds' cluster_predictions.csv into one
  combined population (~950 clusters) and run one threshold sweep over it.
  More data narrows the band of "effectively equivalent" thresholds.

  FIX 2 (complementary): instead of keeping the first argmin (which is the
  lowest-probability candidate that happens to achieve minimum cost -- an
  artifact of iteration order, not a meaningful signal), find the FULL BAND
  of thresholds whose total cost is within PLATEAU_TOL_FRAC of the minimum
  and report the midpoint. This centres us on the plateau rather than on one
  of its edges, and is robust to small cost differences between edge cases.

Together these two fixes attack different aspects of the same root cause:
  - Fix 1 reduces plateau WIDTH by adding more signal (more clusters =>
    more gradient in the cost function near the true optimum).
  - Fix 2 makes the REPORTED threshold robust to wherever on any remaining
    plateau the sweep happens to land.

Output:
  - printed report to stdout
  - pooled_threshold_selection_results.csv  (per-threshold cost table for
    the pooled population, for manual inspection / plotting)
  - pooled_threshold_selection_summary.json (machine-readable result with
    the recommended threshold and how it was derived)

Usage:
    python pooled_threshold_selection.py
    python pooled_threshold_selection.py --runs-dir multi_seed_runs
    python pooled_threshold_selection.py --seeds 1 3 5 7 9 11 13 15
    python pooled_threshold_selection.py --fp-multiplier 2.0
    python pooled_threshold_selection.py --plateau-tol 0.005  # 0.5% default
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants that must stay in sync with the rest of the pipeline
# ---------------------------------------------------------------------------
# Must match risk_scoring.py's CHAMPION_FEATURE_COLS (pure graph + referral),
# same as final_threshold_report.py and bootstrap_threshold_ci.py.
#
# KNOWN GAP (as of this fix): the 15 runs under multi_seed_runs/ predate
# referral features -- their cluster_predictions.csv files don't have this
# column at all. Pooling across them will KeyError until multi_seed_eval.py's
# STAGES list is updated to include referral_features.py and all 15 seeds are
# regenerated. Until then, treat any pooled_threshold_selection_summary.json
# produced before that regeneration as stale for the champion model, even
# though threshold_config.py will still pick it up as-is.
CHOSEN_COLUMN = "oof_prob_logreg_pure_graph_referral"
DEFAULT_FP_MULTIPLIER = 1.0                    # 1x avg order value; matches final_report.md
DEFAULT_PLATEAU_TOL_FRAC = 0.005              # 0.5% of best cost = "negligible" band
CURRENT_LOCKED_THRESHOLD = 0.48111024428768   # what the pipeline currently uses


# ---------------------------------------------------------------------------
# Cost-input construction (mirrors threshold_sweep.build_cluster_cost_inputs,
# kept inline here so this script has zero imports from the pipeline itself --
# it can run standalone against any set of seed directories without needing
# every pipeline file to be present)
# ---------------------------------------------------------------------------

def build_cost_inputs(clusters: pd.DataFrame, ground_truth: pd.DataFrame,
                      orders: pd.DataFrame) -> pd.DataFrame:
    """
    Per-cluster: ring_value_if_missed (cost if we miss the cluster) and
    non_ring_accounts_if_flagged (headcount FP cost target if we flag it).

    Intentionally identical logic to threshold_sweep.build_cluster_cost_inputs
    so the two scripts can't silently drift apart.
    """
    merged = clusters.merge(ground_truth, on="account_id", how="left")
    is_ring = (merged["is_ring_member"] == True)
    if "is_referral_ring_member" in merged.columns:
        is_ring = is_ring | (merged["is_referral_ring_member"] == True)
    merged["is_ring_member"] = is_ring
    order_value = orders.groupby("account_id")["amount"].sum()
    merged["order_value"] = merged["account_id"].map(order_value).fillna(0.0)

    ring_value = (
        merged[merged.is_ring_member]
        .groupby("cluster_id")["order_value"].sum()
        .rename("ring_value_if_missed")
    )
    non_ring_count = (
        merged[~merged.is_ring_member]
        .groupby("cluster_id").size()
        .rename("non_ring_accounts_if_flagged")
    )

    out = (
        pd.DataFrame({"cluster_id": clusters.cluster_id.unique()})
        .set_index("cluster_id")
        .join(ring_value)
        .join(non_ring_count)
        .fillna(0.0)
        .reset_index()
    )
    return out


# ---------------------------------------------------------------------------
# Sweep + plateau-centering
# ---------------------------------------------------------------------------

def sweep_thresholds(y_prob: np.ndarray, cost_inputs: pd.DataFrame,
                     fp_multiplier: float, avg_order_value: float) -> pd.DataFrame:
    """
    Evaluate every unique predicted probability (plus sentinel 0.0 and 1.0001)
    as a candidate threshold. Returns a DataFrame with one row per candidate,
    sorted by threshold ascending.

    Identical structure to threshold_sweep.sweep_thresholds except it also
    records fp_count and fn_count for the plateau analysis.
    """
    candidates = sorted(set(y_prob.tolist()) | {0.0, 1.0001})
    rows = []
    for t in candidates:
        flagged = y_prob >= t
        fp_cost = (fp_multiplier * avg_order_value
                   * cost_inputs.loc[flagged, "non_ring_accounts_if_flagged"].sum())
        fn_cost = cost_inputs.loc[~flagged, "ring_value_if_missed"].sum()
        rows.append({
            "threshold": t,
            "n_flagged": int(flagged.sum()),
            "fp_cost": fp_cost,
            "fn_cost": fn_cost,
            "total_cost": fp_cost + fn_cost,
        })
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def find_plateau_midpoint(sweep: pd.DataFrame,
                          plateau_tol_frac: float) -> dict:
    """
    Fix 2: instead of argmin (which picks an arbitrary edge of the flat
    plateau -- lowest threshold that achieves minimum cost, because sweep
    uses strict <), find the FULL BAND of thresholds within
    plateau_tol_frac * best_cost of the minimum cost, and return the
    midpoint of that band.

    Returns a dict with:
      best_cost          -- the minimum total cost across all candidates
      plateau_lo         -- lowest threshold in the tolerance band
      plateau_hi         -- highest threshold in the tolerance band
      plateau_midpoint   -- (plateau_lo + plateau_hi) / 2  <-- use this
      plateau_n          -- number of distinct thresholds in the band
      argmin_threshold   -- the raw argmin (for comparison / logging)
    """
    best_cost = sweep["total_cost"].min()
    cost_ceiling = best_cost * (1.0 + plateau_tol_frac)

    # If best_cost == 0 the tolerance band would be [0, 0] -- the model is
    # perfect at every threshold, use absolute tolerance of 1 rupee instead.
    if best_cost == 0.0:
        cost_ceiling = 1.0

    in_plateau = sweep["total_cost"] <= cost_ceiling
    plateau_thresholds = sweep.loc[in_plateau, "threshold"]

    plateau_lo = float(plateau_thresholds.min())
    plateau_hi = float(plateau_thresholds.max())
    plateau_mid = (plateau_lo + plateau_hi) / 2.0

    argmin_t = float(sweep.loc[sweep["total_cost"].idxmin(), "threshold"])

    return {
        "best_cost": float(best_cost),
        "cost_ceiling": float(cost_ceiling),
        "plateau_tol_frac": plateau_tol_frac,
        "plateau_lo": plateau_lo,
        "plateau_hi": plateau_hi,
        "plateau_midpoint": plateau_mid,
        "plateau_width": plateau_hi - plateau_lo,
        "plateau_n_candidates": int(in_plateau.sum()),
        "argmin_threshold": argmin_t,
    }


# ---------------------------------------------------------------------------
# Per-seed loader
# ---------------------------------------------------------------------------

def load_seed(seed_dir: Path, seed_id: int,
              chosen_column: str) -> dict | None:
    """
    Loads cluster_predictions.csv, clusters.csv, day1_data/ground_truth.csv,
    day1_data/orders.csv from seed_dir.

    Returns a dict with:
      seed          -- seed integer
      y_prob        -- np.ndarray (n_clusters,)
      y_true        -- np.ndarray (n_clusters,) of 0/1
      cluster_ids   -- list of cluster_id values (with seed prefix to avoid collisions)
      cost_inputs   -- DataFrame with ring_value_if_missed / non_ring_accounts_if_flagged
      avg_order_value -- float

    Returns None (with a warning printed) if any required file is missing.
    """
    preds_path = seed_dir / "cluster_predictions.csv"
    clusters_path = seed_dir / "clusters.csv"
    gt_path = seed_dir / "day1_data" / "ground_truth.csv"
    orders_path = seed_dir / "day1_data" / "orders.csv"

    missing = [p for p in [preds_path, clusters_path, gt_path, orders_path]
               if not p.exists()]
    if missing:
        print(f"  [WARN] seed {seed_id}: missing files, skipping -- "
              f"{[str(p.name) for p in missing]}", file=sys.stderr)
        return None

    predictions = pd.read_csv(preds_path)
    clusters = pd.read_csv(clusters_path)
    ground_truth = pd.read_csv(gt_path)
    orders = pd.read_csv(orders_path)

    effective_col = chosen_column
    if effective_col not in predictions.columns:
        if "oof_prob_logreg_pure_graph" in predictions.columns:
            effective_col = "oof_prob_logreg_pure_graph"
        else:
            print(f"  [WARN] seed {seed_id}: column '{chosen_column}' missing "
                  f"from cluster_predictions.csv, skipping.", file=sys.stderr)
            return None

    # Build cost inputs BEFORE remapping cluster_ids, so build_cost_inputs
    # works with the original integer-like cluster_ids it was designed for.
    avg_order_value = orders["amount"].mean()
    cost_inputs = build_cost_inputs(clusters, ground_truth, orders)

    # NOW remap cluster_id to "s{N}_{id}" strings so pooling multiple seeds
    # never accidentally merges clusters from different populations.
    prefix = f"s{seed_id}_"
    predictions = predictions.copy()
    predictions["cluster_id"] = prefix + predictions["cluster_id"].astype(str)

    cost_inputs = cost_inputs.copy()
    cost_inputs["cluster_id"] = prefix + cost_inputs["cluster_id"].astype(str)

    # Align cost_inputs to the prediction row order (same as threshold_sweep.py)
    cost_inputs = (
        cost_inputs.set_index("cluster_id")
        .loc[predictions["cluster_id"]]
        .reset_index()
    )

    return {
        "seed": seed_id,
        "n_clusters": len(predictions),
        "y_prob": predictions[effective_col].values,
        "y_true": predictions["y_true_is_ring"].values,
        "cluster_ids": list(predictions["cluster_id"]),
        "cost_inputs": cost_inputs,
        "avg_order_value": avg_order_value,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--runs-dir", default="multi_seed_runs",
        help="Directory containing seed_N subdirectories "
             "(default: multi_seed_runs relative to cwd).",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Specific seed numbers to pool. If omitted, all seed_* directories "
             "found in --runs-dir are used.",
    )
    parser.add_argument(
        "--fp-multiplier", type=float, default=DEFAULT_FP_MULTIPLIER,
        help=f"FP cost multiplier (times avg order value per wrongly-flagged "
             f"account). Default: {DEFAULT_FP_MULTIPLIER}.",
    )
    parser.add_argument(
        "--plateau-tol", type=float, default=DEFAULT_PLATEAU_TOL_FRAC,
        help=f"Fractional tolerance above best cost that counts as 'in the plateau'. "
             f"E.g. 0.005 means any threshold within 0.5%% of minimum cost is "
             f"included. Default: {DEFAULT_PLATEAU_TOL_FRAC}.",
    )
    parser.add_argument(
        "--column", default=CHOSEN_COLUMN,
        help=f"Probability column to use from cluster_predictions.csv. "
             f"Default: '{CHOSEN_COLUMN}'.",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"ERROR: --runs-dir '{runs_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Discover seed directories
    if args.seeds is not None:
        seed_dirs = [(s, runs_dir / f"seed_{s}") for s in args.seeds]
    else:
        seed_dirs = sorted(
            (int(d.name.split("_")[1]), d)
            for d in runs_dir.iterdir()
            if d.is_dir() and d.name.startswith("seed_")
               and d.name.split("_")[1].isdigit()
        )

    if not seed_dirs:
        print(f"ERROR: no seed directories found in '{runs_dir}'.", file=sys.stderr)
        sys.exit(1)

    print("=" * 72)
    print("Pooled Threshold Selection")
    print("=" * 72)
    print(f"Runs directory  : {runs_dir.resolve()}")
    print(f"Seeds requested : {[s for s, _ in seed_dirs]}")
    print(f"Model column    : {args.column}")
    print(f"FP multiplier   : {args.fp_multiplier}x avg order value")
    print(f"Plateau tol     : {args.plateau_tol * 100:.2f}% of best cost")
    print(f"Current locked  : {CURRENT_LOCKED_THRESHOLD}")
    print()

    # -----------------------------------------------------------------------
    # Load all seeds
    # -----------------------------------------------------------------------
    print("Loading seeds...")
    loaded = []
    for seed_id, seed_dir in seed_dirs:
        if not seed_dir.exists():
            print(f"  [WARN] seed {seed_id}: directory '{seed_dir}' not found, "
                  f"skipping.", file=sys.stderr)
            continue
        result = load_seed(seed_dir, seed_id, args.column)
        if result is not None:
            loaded.append(result)
            print(f"  seed {seed_id:>3}: {result['n_clusters']:>4} clusters  "
                  f"({int(result['y_true'].sum())} ring, "
                  f"{int((result['y_true'] == 0).sum())} non-ring)  "
                  f"avg_order_value=Rs.{result['avg_order_value']:.2f}")

    if not loaded:
        print("ERROR: no seeds loaded successfully.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{len(loaded)}/{len(seed_dirs)} seeds loaded.\n")

    # -----------------------------------------------------------------------
    # Pool: concatenate y_prob, cost_inputs, and derive a single pooled
    # avg_order_value as a simple mean across seeds (seeds are drawn from the
    # same synthetic distribution, so per-seed means are nearly identical;
    # averaging is more principled than just taking seed 1's value).
    # -----------------------------------------------------------------------
    all_y_prob = np.concatenate([s["y_prob"] for s in loaded])
    all_y_true = np.concatenate([s["y_true"] for s in loaded])
    all_cost_inputs = pd.concat(
        [s["cost_inputs"] for s in loaded], ignore_index=True
    )
    pooled_avg_order_value = float(np.mean([s["avg_order_value"] for s in loaded]))

    n_total = len(all_y_prob)
    n_ring = int(all_y_true.sum())
    n_nonring = n_total - n_ring

    print("=" * 72)
    print("Pooled population summary")
    print("=" * 72)
    print(f"  Total clusters : {n_total}")
    print(f"  Ring clusters  : {n_ring}  ({100 * n_ring / n_total:.1f}%)")
    print(f"  Non-ring       : {n_nonring}  ({100 * n_nonring / n_total:.1f}%)")
    print(f"  Avg order value: Rs.{pooled_avg_order_value:.2f}  (pooled mean)")
    print(f"  Total ring value at risk: "
          f"Rs.{all_cost_inputs['ring_value_if_missed'].sum():,.0f}")
    print()

    # -----------------------------------------------------------------------
    # Sweep thresholds on the pooled population
    # -----------------------------------------------------------------------
    sweep = sweep_thresholds(all_y_prob, all_cost_inputs,
                             args.fp_multiplier, pooled_avg_order_value)

    # -----------------------------------------------------------------------
    # Find the plateau and its midpoint
    # -----------------------------------------------------------------------
    plateau = find_plateau_midpoint(sweep, args.plateau_tol)
    recommended_threshold = plateau["plateau_midpoint"]

    # -----------------------------------------------------------------------
    # Evaluate both thresholds on the pooled population (for comparison)
    # -----------------------------------------------------------------------
    def evaluate_at(t: float, label: str) -> dict:
        flagged = all_y_prob >= t
        tp = int((flagged & (all_y_true == 1)).sum())
        fp = int((flagged & (all_y_true == 0)).sum())
        fn = int((~flagged & (all_y_true == 1)).sum())
        tn = int((~flagged & (all_y_true == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        fp_cost = (args.fp_multiplier * pooled_avg_order_value
                   * all_cost_inputs.loc[flagged, "non_ring_accounts_if_flagged"].sum())
        fn_cost = all_cost_inputs.loc[~flagged, "ring_value_if_missed"].sum()
        total_cost = fp_cost + fn_cost
        return {
            "label": label, "threshold": t,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n_flagged": int(flagged.sum()),
            "precision": precision, "recall": recall,
            "fp_cost": fp_cost, "fn_cost": fn_cost,
            "total_cost": total_cost,
        }

    rec = evaluate_at(recommended_threshold, "Recommended (plateau midpoint)")
    cur = evaluate_at(CURRENT_LOCKED_THRESHOLD, "Current locked (0.4811)")

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print("=" * 72)
    print("Threshold sweep result (pooled population)")
    print("=" * 72)
    print(f"  Best achievable cost   : Rs.{plateau['best_cost']:>12,.0f}")
    print(f"  Plateau tolerance       : {args.plateau_tol * 100:.2f}%  "
          f"=> cost ceiling Rs.{plateau['cost_ceiling']:>12,.0f}")
    print(f"  Plateau band           : [{plateau['plateau_lo']:.6f}, "
          f"{plateau['plateau_hi']:.6f}]  "
          f"(width={plateau['plateau_width']:.4f})")
    print(f"  Plateau candidates     : {plateau['plateau_n_candidates']} "
          f"distinct threshold values")
    print(f"  Raw argmin (first edge): {plateau['argmin_threshold']:.6f}")
    print(f"  Plateau midpoint       : {recommended_threshold:.6f}  "
          f"<-- RECOMMENDED")
    print()

    print("=" * 72)
    print("Performance comparison at the two thresholds (pooled population)")
    print("=" * 72)
    header = (f"{'Label':<12} {'Value':>10} {'n_flag':>7} "
              f"{'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} "
              f"{'Prec':>6} {'Rec':>6} {'TotalCost':>14}")
    print(header)
    print("-" * len(header))
    for ev in [rec, cur]:
        tag = "rec_midpt" if ev["threshold"] == recommended_threshold else "locked_cur"
        cost_delta = ev["total_cost"] - plateau["best_cost"]
        delta_pct = 100 * cost_delta / plateau["best_cost"] if plateau["best_cost"] else 0.0
        suffix = f"  (+{delta_pct:.2f}% vs best)" if cost_delta > 0 else "  (=best)"
        print(f"{tag:<12} {ev['threshold']:>10.6f} {ev['n_flagged']:>7} "
              f"{ev['tp']:>4} {ev['fp']:>4} {ev['fn']:>4} {ev['tn']:>4} "
              f"{ev['precision']:>6.3f} {ev['recall']:>6.3f} "
              f"Rs.{ev['total_cost']:>10,.0f}{suffix}")
    print()

    # How far is the recommended threshold from the current locked one?
    dist = abs(recommended_threshold - CURRENT_LOCKED_THRESHOLD)
    print(f"  Distance from current locked threshold: {dist:.4f}")
    if dist <= 0.05:
        print("  => Within +/-0.05 of current locked threshold.")
        print("     Current threshold is inside the pooled plateau -- no urgent change,")
        print("     but updating to the plateau midpoint improves long-run stability.")
    else:
        print("  => More than +/-0.05 from current locked threshold.")
        print("     Consider updating CHOSEN_THRESHOLD to the plateau midpoint.")
    print()

    # -----------------------------------------------------------------------
    # Per-seed validation: evaluate recommended threshold on each seed alone
    # -----------------------------------------------------------------------
    print("=" * 72)
    print("Per-seed validation at the recommended threshold")
    print("(same threshold applied to each seed individually, as a sanity check)")
    print("=" * 72)
    per_seed_header = (f"{'seed':>6} {'n_cl':>5} "
                       f"{'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} "
                       f"{'Prec':>6} {'Rec':>6} {'TotalCost':>14}")
    print(per_seed_header)
    print("-" * len(per_seed_header))
    per_seed_rows = []
    for s in loaded:
        yp = s["y_prob"]
        yt = s["y_true"]
        ci = s["cost_inputs"]
        avg_ov = s["avg_order_value"]
        flagged = yp >= recommended_threshold
        tp = int((flagged & (yt == 1)).sum())
        fp = int((flagged & (yt == 0)).sum())
        fn = int((~flagged & (yt == 1)).sum())
        tn = int((~flagged & (yt == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec_ = tp / (tp + fn) if (tp + fn) else 0.0
        fp_c = (args.fp_multiplier * avg_ov
                * ci.loc[flagged, "non_ring_accounts_if_flagged"].sum())
        fn_c = ci.loc[~flagged, "ring_value_if_missed"].sum()
        tc = fp_c + fn_c
        print(f"{s['seed']:>6} {s['n_clusters']:>5} "
              f"{tp:>4} {fp:>4} {fn:>4} {tn:>4} "
              f"{prec:>6.3f} {rec_:>6.3f} "
              f"Rs.{tc:>10,.0f}")
        per_seed_rows.append({
            "seed": s["seed"], "n_clusters": s["n_clusters"],
            "threshold": recommended_threshold,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec_,
            "total_cost": tc,
        })
    print()

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    sweep_out = Path("pooled_threshold_selection_results.csv")
    sweep.to_csv(sweep_out, index=False)

    summary = {
        "recommended_threshold": recommended_threshold,
        "derivation": "plateau_midpoint",
        "plateau_lo": plateau["plateau_lo"],
        "plateau_hi": plateau["plateau_hi"],
        "plateau_width": plateau["plateau_width"],
        "plateau_n_candidates": plateau["plateau_n_candidates"],
        "plateau_tol_frac": args.plateau_tol,
        "best_cost_pooled": plateau["best_cost"],
        "argmin_threshold": plateau["argmin_threshold"],
        "current_locked_threshold": CURRENT_LOCKED_THRESHOLD,
        "distance_from_locked": dist,
        "fp_multiplier": args.fp_multiplier,
        "pooled_avg_order_value": pooled_avg_order_value,
        "n_seeds_pooled": len(loaded),
        "seeds_used": [s["seed"] for s in loaded],
        "n_clusters_pooled": n_total,
        "n_ring_clusters": n_ring,
        "model_column": args.column,
        "performance_at_recommended": {
            k: v for k, v in rec.items() if k != "label"
        },
        "performance_at_locked": {
            k: v for k, v in cur.items() if k != "label"
        },
        "per_seed_at_recommended": per_seed_rows,
    }
    summary_out = Path("pooled_threshold_selection_summary.json")
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved {sweep_out}  ({len(sweep)} rows, one per candidate threshold)")
    print(f"Saved {summary_out}")
    print()
    print(f"RECOMMENDED THRESHOLD: {recommended_threshold:.6f}")
    print(f"  (plateau midpoint of [{plateau['plateau_lo']:.4f}, "
          f"{plateau['plateau_hi']:.4f}], {plateau['plateau_n_candidates']} "
          f"candidate thresholds within {args.plateau_tol * 100:.2f}% of best cost)")


if __name__ == "__main__":
    main()
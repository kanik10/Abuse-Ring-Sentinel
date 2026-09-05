#!/usr/bin/env python3
"""
run_for_judges.py — one-command reproducibility runner for Abuse-Ring Sentinel.

Re-derives every headline claim in README.md / final_report.md from the
already-committed data and trained artifacts, in dependency order, and stops
immediately with a clear error (not a silent partial run) if any step fails.

Two modes:

  python run_for_judges.py
      Fast path (~1-2 min). Verifies the headline numbers using the already-
      committed cluster_features.csv / clusters.csv / cluster_predictions.csv
      / final_model.joblib. This is what you want to run live for judges.

  python run_for_judges.py --full
      Full pipeline regeneration. Re-runs entity resolution, graph, clusters,
      features, classifier, account scoring, and multi-seed threshold selection
      from scratch using the canonical day1_data/ files. Slow — this is for
      verifying true end-to-end pipeline reproducibility, not for a live demo.

Dependency order below was derived by reading each script's actual read/write
calls (pd.read_csv / to_csv / joblib.load / joblib.dump / open(...)), not
assumed from the README's day-by-day narrative. If a step's expected output
file doesn't appear after it runs, this script fails loudly rather than
letting a later step run against stale data.

app_interactive.py (the Streamlit cockpit) is intentionally NOT part of this
runner — it's a server, not a batch job, and never exits on its own. Launch
it separately:
    streamlit run src/app_interactive.py
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Each entry: (script, description, output files to sanity-check afterward)
FAST_STEPS = [
    ("src/final_threshold_report.py",
     "Cluster-level confusion matrix, ROI, and final_report.md",
     ["final_report.md", "metrics_summary.json"]),
    ("src/risk_scoring.py",
     "Train the Champion model, export final_model.joblib, write the "
     "hash-chained audit_log.jsonl",
     ["final_model.joblib", "audit_log.jsonl"]),
    ("src/verify_audit_log.py",
     "Verify the audit log's hash chain is intact and untampered",
     []),
    ("src/bootstrap_threshold_ci.py",
     "10,000-resample bootstrap CIs for precision/recall/cost",
     ["bootstrap_threshold_ci_results.csv"]),
    ("src/phase3_temporal_backtest.py",
     "101-snapshot zero-lookahead temporal backtest (needs final_model.joblib)",
     ["phase3_detection_latency_audit.csv", "phase3_counterfactual_summary.json"]),
    ("src/naive_baseline.py",
     "Naive shared-address baseline comparison",
     ["naive_baseline_results.csv"]),
    ("src/build_dashboard.py",
     "Build the offline dashboard.html from audit_log.jsonl",
     ["dashboard.html"]),
]

# Only run with --full, and only BEFORE the fast steps above.
# Reads canonical day1_data/ CSVs directly -- does not regenerate them so the
# canonical benchmark data and locked threshold remain completely stable.
FULL_STEPS_PIPELINE = [
    ("src/entity_resolution.py",
     "Resolve messy raw entity IDs into canonical resolved IDs "
     "(writes day1_data/resolved_account_*.csv)",
     ["day1_data/resolved_account_device.csv"]),
    ("src/community_detection.py",
     "Louvain community detection", ["clusters.csv"]),
    ("src/referral_features.py",
     "Directed referral cycle/latency features", ["referral_cluster_features.csv"]),
    ("src/feature_engineering.py",
     "17-feature graph topology extraction", ["cluster_features.csv"]),
    ("src/classifier.py",
     "5-fold CV model selection", ["cluster_predictions.csv"]),
    ("src/account_scoring.py",
     "Per-account mastermind/sleeper ranking", ["account_scores.csv"]),
    ("src/threshold_sweep.py",
     "Single-seed 0.1x-100x FP cost sweep (legacy threshold analysis)",
     ["threshold_sweep_results.csv"]),
]

# Only run with --full. Must come after FULL_STEPS_PIPELINE and before
# FAST_STEPS, since pooled_threshold_selection.py writes the JSON that
# threshold_config.py reads at import time (which risk_scoring.py imports).
FULL_STEPS_MULTISEED = [
    ("src/multi_seed_eval.py",
     "Run the full pipeline across 15 independent synthetic seeds (SLOW)", []),
    ("src/pooled_threshold_selection.py",
     "Pool all 15 seeds to select CHOSEN_THRESHOLD",
     ["pooled_threshold_selection_summary.json"]),
]


def run_step(script: str, description: str) -> float:
    print(f"\n{'=' * 70}\n>> {script}\n   {description}\n{'=' * 70}")
    start = time.time()
    result = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\nFAILED: {script} exited with code {result.returncode} "
              f"after {elapsed:.1f}s")
        sys.exit(1)
    print(f"OK: {script} completed in {elapsed:.1f}s")
    return elapsed


def check_outputs(script: str, expected_files: list[str]) -> None:
    missing = [f for f in expected_files if not (ROOT / f).exists()]
    if missing:
        print(f"\nFAILED: {script} ran but did not produce expected "
              f"output(s): {missing}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Abuse-Ring Sentinel pipeline for judges/reviewers.")
    parser.add_argument(
        "--full", action="store_true",
        help="Rebuild all pipeline artifacts from canonical day1_data/ and "
             "re-run the 15-seed multi-seed evaluation. Slow. Only needed "
             "to verify full end-to-end pipeline reproducibility.")
    parser.add_argument(
        "--skip-tests", action="store_true",
        help="Skip the pytest suite before running the pipeline.")
    args = parser.parse_args()

    overall_start = time.time()

    if not args.skip_tests:
        print(f"\n{'=' * 70}\n>> pytest tests/\n   Unit test suite\n{'=' * 70}")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v"], cwd=ROOT)
        if result.returncode != 0:
            print("\nFAILED: test suite did not pass.")
            sys.exit(1)
        print("OK: test suite passed.")

    if args.full:
        print("\n[--full] Re-running full pipeline and 15-seed multi-seed "
              "evaluation from canonical day1_data/. This is slow.")
        for script, desc, outputs in FULL_STEPS_PIPELINE:
            run_step(script, desc)
            check_outputs(script, outputs)
        for script, desc, outputs in FULL_STEPS_MULTISEED:
            run_step(script, desc)
            check_outputs(script, outputs)

    for script, desc, outputs in FAST_STEPS:
        run_step(script, desc)
        check_outputs(script, outputs)

    elapsed = time.time() - overall_start
    print(f"\n{'=' * 70}")
    print(f"ALL STEPS PASSED in {elapsed / 60:.1f} minutes.")
    print("Reproduced:")
    print("  - final_report.md / metrics_summary.json   (precision/recall/ROI)")
    print("  - audit_log.jsonl                           (hash-chain verified)")
    print("  - bootstrap_threshold_ci_results.csv         (95% bootstrap CIs)")
    print("  - phase3_detection_latency_audit.csv         (zero-lookahead backtest)")
    print("  - naive_baseline_results.csv                 (baseline comparison)")
    print("  - dashboard.html                             (open directly, no server)")
    print(f"{'=' * 70}")
    print("\nFor the interactive cockpit, run separately (not launched by this "
          "script, since Streamlit blocks and never exits):")
    print("    streamlit run src/app_interactive.py")
    print("    (or on Windows: python -m streamlit run src/app_interactive.py)")


if __name__ == "__main__":
    main()
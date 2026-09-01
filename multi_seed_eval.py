"""
multi_seed_eval.py

Runs the full detection pipeline end-to-end, once per synthetic-data seed,
and aggregates the final operating-point metrics across runs. Answers one
question: "is the headline precision/recall a fluke of one random synthetic
population, or does it hold across independently drawn ones?"

Design choices (see conversation for the full reasoning):
  - Only the DATA-GENERATION seed (SYNTH_SEED, read by generate_synthetic_
    data_v2.py) varies across runs. Every algorithmic random_state
    (Louvain, StratifiedKFold, LogisticRegression, XGBoost, betweenness
    sampling) stays pinned at 42 in every run, exactly as it already is
    in the untouched pipeline files. This isolates "does the population
    matter" from "does the model's own randomness matter" -- two
    different questions, only the first is what this script answers.
  - The classification THRESHOLD is also pinned (0.48111024428768,
    hardcoded in final_threshold_report.py, unchanged by this script).
    That means this tests something closer to a real deployment
    scenario: "if I lock in the operating point I already chose, does
    it keep working on data it wasn't tuned on" -- not "if I re-tune
    fresh every time, do I always find *some* good threshold."
  - threshold_sweep.py is intentionally NOT re-run per seed: it isn't on
    the read path of final_threshold_report.py (which uses the hardcoded
    CHOSEN_THRESHOLD constant, not threshold_sweep_results.csv), so
    running it per seed would cost time without affecting the reported
    metric. If you later change final_threshold_report.py to read a
    per-seed-optimal threshold instead, add threshold_sweep.py back into
    STAGES below.

Each seed gets its own isolated working directory under
multi_seed_runs/seed_<N>/, with its own day1_data/ subfolder -- this
replicates the same generate -> copy-to-day1_data -> downstream-scripts
flow you currently do by hand for a single run, so nothing about the
pipeline scripts' own file-reading logic has to change.

Usage:
    python3 multi_seed_eval.py                  # seeds 1..15
    python3 multi_seed_eval.py --seeds 1 2 3     # specific seeds
    python3 multi_seed_eval.py --n-seeds 5        # seeds 1..5

Requires: the pipeline .py files (generate_synthetic_data_v2.py,
community_detection.py, feature_engineering.py, graph_builder.py,
classifier.py, final_threshold_report.py) in the same directory as
this script.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "multi_seed_runs"

# Files copied into each seed's isolated directory. Keep this in sync with
# whatever the real single-run pipeline needs -- entity_resolution.py and
# risk_scoring.py/app_interactive.py are deliberately excluded: they aren't
# on the path to metrics_summary.json (entity_resolution's resolved_*.csv
# outputs aren't read by anything downstream as currently wired; risk_
# scoring/app_interactive are the demo/serving layer, not the eval chain).
PIPELINE_FILES = [
    "generate_synthetic_data_v2.py",
    "graph_builder.py",
    "community_detection.py",
    "feature_engineering.py",
    "classifier.py",
    "final_threshold_report.py",
]

DAY1_DATA_FILES = [
    "accounts.csv", "account_device.csv", "account_payment.csv",
    "account_address.csv", "account_ip.csv", "ground_truth.csv", "orders.csv",
]

# (script, description) in required execution order.
STAGES = [
    ("generate_synthetic_data_v2.py", "generating synthetic data"),
    ("community_detection.py", "clustering (Louvain)"),
    ("feature_engineering.py", "building cluster features"),
    ("classifier.py", "training / scoring classifier"),
    ("final_threshold_report.py", "applying locked threshold"),
]


def run_stage(script: str, cwd: Path, env: dict, log_path: Path) -> bool:
    """Run one pipeline stage as a subprocess. Returns True on success.
    On failure, stdout+stderr are saved to log_path so the failure is
    inspectable instead of silently skipped."""
    result = subprocess.run(
        [sys.executable, script],
        cwd=cwd, env=env,
        capture_output=True, text=True,
    )
    log_path.write_text(
        f"$ python3 {script}\n\n--- stdout ---\n{result.stdout}\n"
        f"\n--- stderr ---\n{result.stderr}\n"
    )
    return result.returncode == 0


def run_one_seed(seed: int) -> dict:
    """Runs the full pipeline for one seed in an isolated directory.
    Returns a result dict -- either the parsed metrics_summary.json plus
    seed/status, or a status='failed' dict with a pointer to the stage log."""
    seed_dir = RUNS_DIR / f"seed_{seed}"
    if seed_dir.exists():
        shutil.rmtree(seed_dir)
    seed_dir.mkdir(parents=True)
    (seed_dir / "day1_data").mkdir()

    for fname in PIPELINE_FILES:
        shutil.copy(ROOT / fname, seed_dir / fname)

    import os
    env = os.environ.copy()
    env["SYNTH_SEED"] = str(seed)

    logs_dir = seed_dir / "logs"
    logs_dir.mkdir()

    for script, description in STAGES:
        print(f"  seed {seed}: {description}...", end=" ", flush=True)
        ok = run_stage(script, cwd=seed_dir, env=env,
                        log_path=logs_dir / f"{script}.log")
        if not ok:
            print("FAILED")
            return {
                "seed": seed, "status": "failed",
                "failed_stage": script,
                "log": str(logs_dir / f"{script}.log"),
            }
        print("ok")

        # Replicate the manual day1_data copy step, but only after the
        # generator has actually run -- must happen before the next stage
        # (community_detection.py) reads from day1_data/.
        if script == "generate_synthetic_data_v2.py":
            for fname in DAY1_DATA_FILES:
                shutil.copy(seed_dir / fname, seed_dir / "day1_data" / fname)

    summary_path = seed_dir / "metrics_summary.json"
    if not summary_path.exists():
        return {"seed": seed, "status": "failed",
                "failed_stage": "final_threshold_report.py (no summary written)",
                "log": str(logs_dir / "final_threshold_report.py.log")}

    summary = json.loads(summary_path.read_text())
    summary["seed"] = seed
    summary["status"] = "ok"
    return summary


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs):
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                         help="Explicit list of seeds to run.")
    parser.add_argument("--n-seeds", type=int, default=15,
                         help="If --seeds not given, run seeds 1..N (default 15). "
                              "Seed 42 (the canonical reported run) is deliberately "
                              "excluded so this is a genuine out-of-sample check.")
    args = parser.parse_args()
    seeds = args.seeds if args.seeds else list(range(1, args.n_seeds + 1))

    RUNS_DIR.mkdir(exist_ok=True)
    print(f"Running {len(seeds)} independent seeds: {seeds}\n")

    results = []
    t0 = time.time()
    for seed in seeds:
        results.append(run_one_seed(seed))
    elapsed = time.time() - t0

    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] == "failed"]

    print(f"\n{'='*70}")
    print(f"{len(ok)}/{len(seeds)} seeds completed cleanly  ({elapsed:.0f}s total)")
    if failed:
        print(f"{len(failed)} FAILED -- see logs listed below, not silently skipped:")
        for r in failed:
            print(f"  seed {r['seed']}: failed at {r['failed_stage']} -> {r['log']}")
    print(f"{'='*70}\n")

    if ok:
        precisions = [r["precision"] for r in ok]
        recalls = [r["recall"] for r in ok]
        header = (f"{'seed':>6} {'precision':>10} {'recall':>8} "
                  f"{'tp':>4} {'fp':>4} {'fn':>4} {'tn':>4} {'n_flagged':>10}")
        print(header)
        print("-" * len(header))
        for r in ok:
            print(f"{r['seed']:>6} {r['precision']:>10.3f} {r['recall']:>8.3f} "
                  f"{r['tp']:>4} {r['fp']:>4} {r['fn']:>4} {r['tn']:>4} "
                  f"{r['n_flagged']:>10}")
        print("-" * len(header))
        print(f"{'mean':>6} {mean(precisions):>10.3f} {mean(recalls):>8.3f}")
        print(f"{'std':>6} {stdev(precisions):>10.3f} {stdev(recalls):>8.3f}\n")

        n_perfect = sum(1 for p, r in zip(precisions, recalls) if p == 1.0 and r == 1.0)
        print(f"{n_perfect}/{len(ok)} seeds: perfect precision AND recall at the "
              f"locked threshold (0.4811, tuned on seed 42, unchanged here).")

        results_path = RUNS_DIR / "multi_seed_results.csv"
        import csv
        with open(results_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ok[0].keys()))
            w.writeheader()
            w.writerows(ok)
        print(f"\nSaved {results_path}")
    else:
        print("No successful runs -- nothing to aggregate.")


if __name__ == "__main__":
    main()

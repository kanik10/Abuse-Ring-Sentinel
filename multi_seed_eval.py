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
  - The classification THRESHOLD is also pinned (whatever CHOSEN_THRESHOLD
    is currently hardcoded in final_threshold_report.py -- 0.7732484382694863
    as of the "Multi seed eval 1" / day1_data-direct-write commit). That
    means this tests something closer to a real deployment scenario: "if
    I lock in the operating point I already chose, does it keep working
    on data it wasn't tuned on" -- not "if I re-tune fresh every time, do
    I always find *some* good threshold."
  - threshold_sweep.py is intentionally NOT re-run per seed: it isn't on
    the read path of final_threshold_report.py (which uses the hardcoded
    CHOSEN_THRESHOLD constant, not threshold_sweep_results.csv), so
    running it per seed would cost time without affecting the reported
    metric. If you later change final_threshold_report.py to read a
    per-seed-optimal threshold instead, add threshold_sweep.py back into
    STAGES below -- and note that "recompute the best threshold every
    seed" answers a different question than the one this script answers
    now (see conversation).

Each seed gets its own isolated working directory under
multi_seed_runs/seed_<N>/. As of the latest generator commit,
generate_synthetic_data_v2.py writes its CSVs directly to day1_data/
(controlled by the SYNTH_OUTPUT_DIR env var, defaulting to "day1_data"
relative to cwd) -- so running each seed with cwd set to its own
directory already gives it an isolated day1_data/ with no copy step
needed. (Earlier versions of this script manually copied root-level
CSVs into day1_data/; that step is gone because the generator does it
natively now.)

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
import os
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

# (script, description) in required execution order.
STAGES = [
    ("generate_synthetic_data_v2.py", "generating synthetic data (writes day1_data/ directly)"),
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
    # No manual day1_data/ pre-creation needed -- the generator's own
    # OUTPUT_DIR.mkdir(parents=True, exist_ok=True) handles it.

    for fname in PIPELINE_FILES:
        shutil.copy(ROOT / fname, seed_dir / fname)

    env = os.environ.copy()
    env["SYNTH_SEED"] = str(seed)
    # Not strictly required (default is already "day1_data" relative to
    # cwd, and cwd is already this seed's own isolated directory) -- set
    # explicitly anyway so this script doesn't silently break if someone
    # changes the generator's default later.
    env["SYNTH_OUTPUT_DIR"] = "day1_data"

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
        thresholds_seen = {r["chosen_threshold"] for r in ok}
        threshold_note = (f"{thresholds_seen.pop():.4f}" if len(thresholds_seen) == 1
                           else f"WARNING: threshold varied across seeds: {thresholds_seen}")
        print(f"{n_perfect}/{len(ok)} seeds: perfect precision AND recall at the "
              f"locked threshold ({threshold_note}, read live from each run's "
              f"metrics_summary.json -- not hardcoded here).")

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

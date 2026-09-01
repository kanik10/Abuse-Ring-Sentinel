"""
Naive baseline -- shared resolved-address size threshold.

This baseline is fair because it starts from resolved_account_address.csv,
the same post-entity-resolution address mapping the real pipeline is meant
to consume. That keeps entity-resolution quality out of the comparison:
the only thing being tested here is whether "large shared address group"
can stand in for graph construction, community detection, feature
engineering, and classification.

It is intentionally naive: it ignores devices and payments, does not build
an account-account graph, does not use community structure, and does not
train a classifier. For each size threshold N, every account sharing any
resolved address with at least N unique accounts is predicted positive.

ground_truth.csv is read only after those threshold predictions are built,
and is used only for final scoring. The best-F1 row printed at the end is
a best-case retrospective baseline chosen with labels, not a blind
operating threshold.
"""

import re
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple

import pandas as pd

ADDRESS_PATH = Path("resolved_account_address.csv")
GROUND_TRUTH_PATH = Path("day1_data/ground_truth.csv")
FINAL_REPORT_PATH = Path("final_report.md")
RESULTS_PATH = Path("naive_baseline_results.csv")
THRESHOLDS = [3, 4, 5, 6, 8, 10, 12]


def require_columns(df: pd.DataFrame, required: Iterable[str], source: Path) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def parse_bool_labels(labels: pd.Series) -> pd.Series:
    if labels.dtype == bool:
        return labels
    normalized = labels.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        bad = sorted(normalized[~normalized.isin({"true", "false"})].unique())
        raise ValueError(f"is_ring_member contains non-boolean values: {bad}")
    return normalized.eq("true")


def build_flagged_accounts_by_threshold(address_mapping: pd.DataFrame) -> Dict[int, Set[str]]:
    """Build predictions using only resolved address sharing. No labels or
    ground-truth columns are available to this function."""
    address_accounts = (address_mapping.groupby("address_id")["account_id"]
                        .agg(lambda s: sorted(set(s))))

    out = {}
    for threshold in THRESHOLDS:
        flagged = set()
        for accounts in address_accounts:
            if len(accounts) >= threshold:
                flagged.update(accounts)
        out[threshold] = flagged
    return out


def score_predictions(ground_truth: pd.DataFrame,
                      flagged_by_threshold: Dict[int, Set[str]]) -> pd.DataFrame:
    y_true = parse_bool_labels(ground_truth["is_ring_member"])
    rows = []

    for threshold, flagged_accounts in flagged_by_threshold.items():
        y_pred = ground_truth["account_id"].isin(flagged_accounts)

        tp = int((y_pred & y_true).sum())
        fp = int((y_pred & ~y_true).sum())
        fn = int((~y_pred & y_true).sum())
        tn = int((~y_pred & ~y_true).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        accuracy = (tp + tn) / len(ground_truth) if len(ground_truth) else 0.0

        rows.append({
            "threshold_n": threshold,
            "predicted_positive_accounts": int(y_pred.sum()),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        })

    return pd.DataFrame(rows)


def parse_final_report_metrics(report_path: Path) -> Tuple[str, str, float, float]:
    text = report_path.read_text()

    model = re.search(r"- \*\*Model:\*\* (.+)", text)
    threshold = re.search(r"- \*\*Threshold:\*\* (.+)", text)
    precision = re.search(r"- Precision:\s*([0-9.]+)", text)
    recall = re.search(r"- Recall:\s*([0-9.]+)", text)

    if not (model and threshold and precision and recall):
        raise ValueError(f"Could not parse model, threshold, precision, and recall from {report_path}")

    return (model.group(1).strip(), threshold.group(1).strip(),
            float(precision.group(1)), float(recall.group(1)))


def print_results(results: pd.DataFrame) -> None:
    display_cols = ["threshold_n", "predicted_positive_accounts",
                    "precision", "recall", "f1", "accuracy"]
    printable = results[display_cols].copy()
    for col in ["precision", "recall", "f1", "accuracy"]:
        printable[col] = printable[col].map(lambda x: f"{x:.3f}")

    print("NAIVE BASELINE -- shared resolved-address groups only")
    print(printable.to_string(index=False))


def main():
    address_mapping = pd.read_csv(ADDRESS_PATH)
    require_columns(address_mapping, ["account_id", "address_id"], ADDRESS_PATH)

    flagged_by_threshold = build_flagged_accounts_by_threshold(address_mapping)

    ground_truth = pd.read_csv(GROUND_TRUTH_PATH)
    require_columns(ground_truth, ["account_id", "is_ring_member"], GROUND_TRUTH_PATH)

    results = score_predictions(ground_truth, flagged_by_threshold)
    results.to_csv(RESULTS_PATH, index=False)

    print_results(results)

    best = results.sort_values(["f1", "precision", "recall"], ascending=False).iloc[0]
    model, model_threshold, model_precision, model_recall = parse_final_report_metrics(FINAL_REPORT_PATH)

    print("\nComparison")
    print(f"Naive best-case threshold after scoring: N={int(best.threshold_n)}  "
          f"precision={best.precision:.3f}  recall={best.recall:.3f}  f1={best.f1:.3f}")
    print(f"Real pipeline from final_report.md: {model} at threshold={model_threshold}  "
          f"precision={model_precision:.3f}  recall={model_recall:.3f}")
    print("\nNote: the naive best-F1 threshold is selected retrospectively with ground truth, "
          "so it is a best-case baseline, not a blind operating threshold.")
    print(f"\nSaved {RESULTS_PATH}")


if __name__ == "__main__":
    main()

"""
account_scoring.py — per-account risk ranking within flagged clusters.

PURPOSE
-------
The cluster-level classifier (classifier.py / risk_scoring.py) flags entire
clusters as ring vs. not-ring. This module adds a second layer: given a
FLAGGED cluster, rank its member accounts by how much structural evidence
points at each one individually, so reviewers can triage the highest-
confidence accounts first without reading every member in a 30-account cluster.

LEAKAGE AND DOUBLE-COUNTING GUARDS
-----------------------------------
Two failure modes are explicitly prevented here:

  1. Double-counting cluster-level aggregates.
     entity_reuse_ratio, internal_density, cluster_size, and the cluster-level
     oof_prob_logreg_pure_graph are all IDENTICAL for every account in the same
     cluster. They carry zero per-account discriminating power and would just
     re-surface the cluster model's decision — excluded entirely.

  2. Using ground_truth.csv (is_ring_member).
     This module never reads ground_truth.csv. The 5 features below are derived
     exclusively from accounts.csv, orders.csv, and the account-resource mapping
     tables (device/payment/address/ip) — all of which were available before any
     labelling decision. ground_truth.csv is used ONLY in the __main__ block's
     post-hoc correlation check (explicitly labelled as evaluation, not feature
     construction).

FEATURES (all account-local; differ across members of the same cluster)
-----------------------------------------------------------------------
  1. n_shared_resources
     Count of distinct (resource_type, resource_id) pairs this account shares
     with at least one OTHER member of the same cluster. Higher = more tightly
     embedded in the ring's sharing pattern.

  2. within_cluster_degree
     Number of distinct cluster-member neighbours in the account-account graph G
     (i.e., degree in the cluster's subgraph). Higher = more central.

  3. within_cluster_edge_weight_sum
     Sum of edge weights to other cluster members. Edge weight = 1/resource_degree
     (from graph_builder.py), so sharing a resource used by only 2 accounts weights
     more than sharing one used by 20. Higher = more exclusive sharing partnerships.

  4. creation_date_centrality
     1 - (|creation_date - cluster_median_creation_date| / max_within_cluster_deviation)
     Accounts created at the same time as the cluster's burst core score higher.
     Sleeper accounts (created months early) score lower. Computed from within-
     cluster dates only — NOT the cluster-level creation_span_days aggregate.

  5. order_amount_centrality
     1 - (|account_avg_order - cluster_mean_order| / max_within_cluster_deviation)
     Accounts whose spending pattern matches the cluster mean score higher.
     Note: RING_BLEND_IN_PROB=0.45 means 45% of ring accounts deliberately
     match the global order distribution, making this the WEAKEST of the 5
     features. It is included for completeness but contributes equally to the
     composite; if it proves uninformative on real data, it can be dropped or
     down-weighted without changing the other 4.

COMPOSITE SCORE
---------------
Each feature is min-max normalized to [0, 1] within the cluster (so scores
are comparable across clusters of different sizes). Composite = arithmetic mean
of the 5 normalized features, then rounded to 4 decimal places.

Edge cases:
  - Cluster of size 1: all features are 0 → composite = 0.0 (lowest priority,
    since there are no other members to compare against anyway).
  - All members tied on one feature: that feature's normalized value = 0.5 for
    all (neither suspicious nor not — neutral, not missing).
  - Account with no orders: order_amount_centrality = 0.5 (neutral).
"""

from __future__ import annotations

import sys
from typing import List

import networkx as nx
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core feature computation
# ---------------------------------------------------------------------------

def _minmax_normalize(series: pd.Series) -> pd.Series:
    """Min-max normalize a series to [0, 1]. When all values are equal,
    return 0.5 for all (neutral, not arbitrary 0 or 1)."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def compute_account_features(
    cluster_id: int,
    member_ids: List[str],
    G: nx.Graph,
    account_device: pd.DataFrame,
    account_payment: pd.DataFrame,
    account_address: pd.DataFrame,
    account_ip: pd.DataFrame,
    accounts: pd.DataFrame,
    orders: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute 5 account-local features for each member of one cluster.

    Parameters
    ----------
    cluster_id : int
        Cluster identifier (carried through to output for traceability).
    member_ids : list of str
        account_id values belonging to this cluster.
    G : nx.Graph
        Full account-account graph (from graph_builder.build_account_graph).
        Only the subgraph restricted to member_ids is used here.
    account_device, account_payment, account_address, account_ip : pd.DataFrame
        Resolved account-resource mapping tables (must have been entity-resolved
        already — pass the resolved_account_*.csv files, not the raw ones).
    accounts : pd.DataFrame
        Must have columns: account_id, creation_date.
    orders : pd.DataFrame
        Must have columns: account_id, amount.

    Returns
    -------
    pd.DataFrame with columns:
        account_id, cluster_id,
        n_shared_resources, within_cluster_degree,
        within_cluster_edge_weight_sum,
        creation_date_centrality, order_amount_centrality
    """
    member_set = set(member_ids)
    n = len(member_ids)

    # ------------------------------------------------------------------ #
    # Feature 1 & 2 & 3: graph-structural features (within-cluster only)
    # ------------------------------------------------------------------ #
    # Build the cluster subgraph once; use it for degree and edge weight.
    subgraph = G.subgraph(member_ids)

    # Shared resource counts: for each account, count distinct (type, id) pairs
    # that it shares with at least one other cluster member.
    resource_tables = [
        ("device",  account_device,  "device_id"),
        ("payment", account_payment, "payment_id"),
        ("address", account_address, "address_id"),
        ("ip",      account_ip,      "ip_id"),
    ]
    # Map account_id → set of (resource_type, resource_id) shared with another member
    shared_by_account: dict[str, set] = {acc: set() for acc in member_ids}
    for rtype, df, rcol in resource_tables:
        if df is None or df.empty:
            continue
        sub = df[df.account_id.isin(member_set)]
        # For each resource_id used by ≥2 cluster members, record it for each member
        resource_members = sub.groupby(rcol)["account_id"].apply(set)
        for resource_id, users in resource_members.items():
            cluster_users = users & member_set
            if len(cluster_users) >= 2:
                for acc in cluster_users:
                    shared_by_account[acc].add((rtype, resource_id))

    rows = []
    for acc in member_ids:
        # Feature 1: n_shared_resources
        n_shared = len(shared_by_account[acc])

        # Features 2 & 3: graph-structural (within-cluster subgraph only)
        if subgraph.has_node(acc):
            neighbors_in_cluster = [v for v in subgraph.neighbors(acc) if v in member_set]
            within_degree = len(neighbors_in_cluster)
            within_weight_sum = sum(
                float(subgraph[acc][v].get("weight", 1.0))
                for v in neighbors_in_cluster
            )
        else:
            within_degree = 0
            within_weight_sum = 0.0

        rows.append({
            "account_id": acc,
            "cluster_id": cluster_id,
            "n_shared_resources": n_shared,
            "within_cluster_degree": within_degree,
            "within_cluster_edge_weight_sum": within_weight_sum,
        })

    df_out = pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # Feature 4: creation_date_centrality
    # ------------------------------------------------------------------ #
    accts_sub = accounts[accounts.account_id.isin(member_set)][["account_id", "creation_date"]].copy()
    accts_sub["creation_date"] = pd.to_datetime(accts_sub["creation_date"])
    accts_sub["creation_ordinal"] = accts_sub["creation_date"].apply(lambda d: d.toordinal())
    df_out = df_out.merge(accts_sub[["account_id", "creation_ordinal"]], on="account_id", how="left")

    if n > 1 and df_out["creation_ordinal"].notna().any():
        median_ord = df_out["creation_ordinal"].median()
        abs_dev = (df_out["creation_ordinal"] - median_ord).abs()
        max_dev = abs_dev.max()
        if max_dev > 0:
            df_out["creation_date_centrality"] = 1.0 - abs_dev / max_dev
        else:
            df_out["creation_date_centrality"] = 0.5  # all created same day → neutral
    else:
        df_out["creation_date_centrality"] = 0.5

    df_out = df_out.drop(columns=["creation_ordinal"])

    # ------------------------------------------------------------------ #
    # Feature 5: order_amount_centrality
    # ------------------------------------------------------------------ #
    # Per-account average order amount (not the cluster-level aggregate!)
    acct_avg_order = (
        orders[orders.account_id.isin(member_set)]
        .groupby("account_id")["amount"].mean()
        .rename("avg_order_amount")
    )
    df_out = df_out.merge(
        acct_avg_order.reset_index(), on="account_id", how="left"
    )
    df_out["avg_order_amount"] = df_out["avg_order_amount"].fillna(np.nan)

    if n > 1 and df_out["avg_order_amount"].notna().any():
        cluster_mean_order = df_out["avg_order_amount"].mean(skipna=True)
        abs_dev = (df_out["avg_order_amount"] - cluster_mean_order).abs()
        max_dev = abs_dev.max(skipna=True)
        if max_dev > 0:
            df_out["order_amount_centrality"] = (1.0 - abs_dev / max_dev).fillna(0.5)
        else:
            df_out["order_amount_centrality"] = 0.5
    else:
        df_out["order_amount_centrality"] = 0.5

    df_out = df_out.drop(columns=["avg_order_amount"])

    return df_out


def score_accounts_in_cluster(
    cluster_id: int,
    member_ids: List[str],
    G: nx.Graph,
    account_device: pd.DataFrame,
    account_payment: pd.DataFrame,
    account_address: pd.DataFrame,
    account_ip: pd.DataFrame,
    accounts: pd.DataFrame,
    orders: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute and normalize per-account risk scores for one cluster.

    Returns a DataFrame sorted by account_risk_score descending, with columns:
        account_id, cluster_id, account_risk_score,
        n_shared_resources, within_cluster_degree,
        within_cluster_edge_weight_sum,
        creation_date_centrality, order_amount_centrality
    """
    df = compute_account_features(
        cluster_id, member_ids, G,
        account_device, account_payment, account_address, account_ip,
        accounts, orders,
    )

    # Min-max normalize the 3 monotone features (higher raw = more suspicious)
    for col in ["n_shared_resources", "within_cluster_degree", "within_cluster_edge_weight_sum"]:
        df[col + "_norm"] = _minmax_normalize(df[col])

    # creation_date_centrality and order_amount_centrality are already in [0, 1]
    # (centrality = 1 - normalized_deviation from cluster center)
    df["creation_date_centrality_norm"] = df["creation_date_centrality"]
    df["order_amount_centrality_norm"] = df["order_amount_centrality"]

    norm_cols = [
        "n_shared_resources_norm",
        "within_cluster_degree_norm",
        "within_cluster_edge_weight_sum_norm",
        "creation_date_centrality_norm",
        "order_amount_centrality_norm",
    ]
    df["account_risk_score"] = df[norm_cols].mean(axis=1).round(4)

    # Drop the _norm columns — the raw features are already in the output
    df = df.drop(columns=norm_cols)

    return df.sort_values("account_risk_score", ascending=False).reset_index(drop=True)


def score_all_flagged_accounts(
    flagged_cluster_ids: list,
    clusters: pd.DataFrame,
    G: nx.Graph,
    account_device: pd.DataFrame,
    account_payment: pd.DataFrame,
    account_address: pd.DataFrame,
    account_ip: pd.DataFrame,
    accounts: pd.DataFrame,
    orders: pd.DataFrame,
) -> pd.DataFrame:
    """
    Score all accounts in all flagged clusters.

    Parameters
    ----------
    flagged_cluster_ids : list of int
        Only these clusters are scored.
    clusters : pd.DataFrame
        Must have columns: cluster_id, account_id.
    (remaining parameters as in score_accounts_in_cluster)

    Returns
    -------
    pd.DataFrame sorted by (cluster_id, account_risk_score DESC), one row per
    account in a flagged cluster.
    """
    all_rows = []
    for cluster_id in flagged_cluster_ids:
        member_ids = clusters.loc[clusters.cluster_id == cluster_id, "account_id"].tolist()
        if not member_ids:
            continue
        scored = score_accounts_in_cluster(
            cluster_id, member_ids, G,
            account_device, account_payment, account_address, account_ip,
            accounts, orders,
        )
        all_rows.append(scored)

    if not all_rows:
        return pd.DataFrame(columns=[
            "account_id", "cluster_id", "account_risk_score",
            "n_shared_resources", "within_cluster_degree",
            "within_cluster_edge_weight_sum",
            "creation_date_centrality", "order_amount_centrality",
        ])

    return pd.concat(all_rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Standalone entry point — leakage checks + quick inspection
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from pathlib import Path
    from graph_builder import build_account_graph
    from threshold_config import CHOSEN_THRESHOLD

    DATA_DIR = "day1_data"
    print("account_scoring.py — standalone validation\n")

    # Load all inputs (same pattern as risk_scoring.py)
    clusters = pd.read_csv("clusters.csv")
    predictions = pd.read_csv("cluster_predictions.csv")
    accounts = pd.read_csv(f"{DATA_DIR}/accounts.csv")
    orders = pd.read_csv(f"{DATA_DIR}/orders.csv")
    account_device = pd.read_csv(f"{DATA_DIR}/resolved_account_device.csv")
    account_payment = pd.read_csv(f"{DATA_DIR}/resolved_account_payment.csv")
    account_address = pd.read_csv(f"{DATA_DIR}/resolved_account_address.csv")
    account_ip = pd.read_csv(f"{DATA_DIR}/resolved_account_ip.csv")

    G = build_account_graph(account_device, account_payment, account_address, account_ip)

    flagged_ids = predictions.loc[
        predictions["oof_prob_logreg_pure_graph"] >= CHOSEN_THRESHOLD, "cluster_id"
    ].tolist()
    print(f"Flagged clusters (threshold={CHOSEN_THRESHOLD:.4f}): {len(flagged_ids)}")

    result = score_all_flagged_accounts(
        flagged_ids, clusters, G,
        account_device, account_payment, account_address, account_ip,
        accounts, orders,
    )
    print(f"Total accounts scored: {len(result)}\n")

    # ------------------------------------------------------------------ #
    # Leakage assertions
    # ------------------------------------------------------------------ #
    FORBIDDEN_COLUMNS = {
        "is_ring_member", "ring_id", "coincidental_group_id",  # ground truth columns
        "entity_reuse_ratio", "internal_density",               # cluster-level aggregates
        "oof_prob_logreg_pure_graph", "oof_prob_xgb_pure_graph",
        "oof_prob_logreg_full", "oof_prob_xgb_full",
    }
    bad_cols = set(result.columns) & FORBIDDEN_COLUMNS
    assert not bad_cols, (
        f"LEAKAGE DETECTED: forbidden columns in output: {bad_cols}"
    )
    print("[OK] No forbidden columns in account feature output (no label leakage).")

    # All scores in [0, 1]
    assert result["account_risk_score"].between(0.0, 1.0).all(), \
        "ASSERTION FAILED: account_risk_score out of [0, 1]"
    print("[OK] All account_risk_score values in [0, 1].")

    # Scores differ within clusters (scorer is non-trivial)
    # For clusters with >=3 members, at least one should have a different score
    large_clusters = result.groupby("cluster_id").filter(lambda g: len(g) >= 3)
    if not large_clusters.empty:
        non_trivial = large_clusters.groupby("cluster_id")["account_risk_score"].nunique()
        assert (non_trivial > 1).any(), (
            "ASSERTION FAILED: all accounts in every large cluster have identical scores "
            "-- the scorer is trivial (all features are tied everywhere)"
        )
        print("[OK] Scores differ within clusters (non-trivial ranking confirmed).")

    # Every account_id in output is also in clusters.csv
    cluster_account_ids = set(clusters["account_id"])
    orphan = set(result["account_id"]) - cluster_account_ids
    assert not orphan, f"ASSERTION FAILED: {len(orphan)} output accounts not in clusters.csv"
    print("[OK] All output accounts are present in clusters.csv.\n")

    # ------------------------------------------------------------------ #
    # Quick inspection — top/bottom 3 accounts per sample cluster
    # ------------------------------------------------------------------ #
    sample_clusters = result["cluster_id"].unique()[:3]
    for cid in sample_clusters:
        sub = result[result.cluster_id == cid]
        print(f"Cluster {cid}  ({len(sub)} members)")
        print(sub[[
            "account_id", "account_risk_score",
            "n_shared_resources", "within_cluster_degree",
            "within_cluster_edge_weight_sum",
            "creation_date_centrality", "order_amount_centrality",
        ]].head(5).to_string(index=False))
        print()

    # ------------------------------------------------------------------ #
    # Post-hoc correlation with ground truth (evaluation only — not features)
    # ------------------------------------------------------------------ #
    gt_path = Path(DATA_DIR) / "ground_truth.csv"
    if gt_path.exists():
        ground_truth = pd.read_csv(gt_path)
        merged = result.merge(
            ground_truth[["account_id", "is_ring_member"]], on="account_id", how="left"
        )
        merged["is_ring_member"] = merged["is_ring_member"].fillna(False).astype(bool)
        corr = merged["account_risk_score"].corr(merged["is_ring_member"].astype(float))
        print(f"Post-hoc correlation (account_risk_score vs is_ring_member): {corr:.3f}")
        print("(This is evaluation only — ground_truth was NOT used in feature construction.)")

        # Within flagged clusters only, precision at top-1 per cluster
        top1 = merged.groupby("cluster_id").first().reset_index()
        top1_precision = top1["is_ring_member"].mean()
        print(f"Top-1 account per flagged cluster is ring member: {top1_precision:.1%}")
    else:
        print("(ground_truth.csv not found — skipping post-hoc evaluation)")

    result.to_csv("account_scores.csv", index=False)
    print(f"\nSaved account_scores.csv  ({len(result)} rows)")

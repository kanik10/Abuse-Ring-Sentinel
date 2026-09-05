"""
referral_features.py -- cluster-level referral-graph features.

PURPOSE
-------
The resource-sharing classifier detects rings through shared devices, IPs,
payment instruments, and addresses.  Referral rings deliberately avoid
sharing these resources to evade that detector -- they share only one IP
(creating a weak Louvain cluster with low entity_reuse_ratio) while farming
referral bonuses through a directed referral chain.

This module computes 4 cluster-level features from the directed referral
graph that characterise referral-ring structure, then joins them to the
cluster feature table.  The combined feature set (3 structural + 4 referral)
lets the classifier flag referral-ring clusters that the structural features
alone would miss.

FEATURES
--------
1. referral_cycle_ratio
   Directed cycles within the cluster referral subgraph, normalised by
   cluster size.  Organic referral trees are DAGs; fraud rings deliberately
   create cycles to maximise bonus extraction.

2. referral_resource_overlap_ratio
   Fraction of referral edges (A->B) where both A and B belong to the same
   resource-sharing cluster.

3. median_referral_activation_days
   Median days between referral_date and referred account first order,
   for all accounts referred BY any cluster member.

4. within_cluster_referral_density
   Fraction of cluster member pairs with a referral edge in either direction.

LEAKAGE GUARD
-------------
Reads: clusters.csv, referrals.csv (NO is_ring_referral column), orders.csv
Never reads: ground_truth.csv, referral_ground_truth.csv during feature
construction.  Post-hoc eval block in __main__ reads labels for evaluation
only.
"""

from __future__ import annotations
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd


def compute_referral_features(
    clusters: pd.DataFrame,
    referrals: pd.DataFrame,
    orders: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute 4 referral-graph features for each cluster.

    Parameters
    ----------
    clusters  : cluster_id, account_id (one row per account)
    referrals : referrer_id, referred_id, referral_date, bonus_amount
                Must NOT contain is_ring_referral.
    orders    : account_id, amount, timestamp

    Returns
    -------
    DataFrame with one row per cluster that has >= 1 referral edge.
    Caller should left-join on cluster_id and fill NaN with 0.
    """
    assert "is_ring_referral" not in referrals.columns, (
        "LEAKAGE: referrals DataFrame contains is_ring_referral. "
        "Pass referrals.csv (pipeline input), not referral_ground_truth.csv."
    )

    acct_to_cluster = dict(zip(clusters["account_id"], clusters["cluster_id"]))

    G_ref = nx.DiGraph()
    referrals["referral_date"] = pd.to_datetime(referrals["referral_date"])
    for _, row in referrals.iterrows():
        G_ref.add_edge(row["referrer_id"], row["referred_id"],
                       referral_date=row["referral_date"])

    orders["timestamp"] = pd.to_datetime(orders["timestamp"])
    first_order = orders.groupby("account_id")["timestamp"].min().rename("first_order_date")

    rows = []
    for cid in clusters["cluster_id"].unique():
        member_ids = set(clusters.loc[clusters["cluster_id"] == cid, "account_id"])
        n = len(member_ids)
        n_pairs = n * (n - 1) / 2

        sub = G_ref.subgraph(member_ids)
        n_edges = sub.number_of_edges()
        if n_edges == 0:
            continue

        # Feature 1: referral_cycle_ratio
        sccs = list(nx.strongly_connected_components(sub))
        n_cycle_nodes = sum(len(s) for s in sccs if len(s) >= 2)
        referral_cycle_ratio = n_cycle_nodes / n if n > 0 else 0.0

        # Feature 2: referral_resource_overlap_ratio
        total_involving = sum(
            1 for u, v in G_ref.edges()
            if u in member_ids or v in member_ids
        )
        referral_resource_overlap_ratio = (
            n_edges / total_involving if total_involving > 0 else 0.0
        )

        # Feature 3: median_referral_activation_days
        activation_days = []
        for referrer in member_ids:
            if referrer not in G_ref:
                continue  # this cluster member has no referral edges at all
            for referred in G_ref.successors(referrer):
                ref_date = G_ref[referrer][referred]["referral_date"]
                if referred in first_order.index:
                    delta = (first_order[referred] - ref_date).days
                    if delta >= 0:
                        activation_days.append(delta)
        median_referral_activation_days = (
            float(np.median(activation_days)) if activation_days else 60.0
        )

        # Feature 4: within_cluster_referral_density
        if n_pairs > 0:
            undirected_pairs = len(set(
                (min(u, v), max(u, v)) for u, v in sub.edges()
            ))
            within_cluster_referral_density = undirected_pairs / n_pairs
        else:
            within_cluster_referral_density = 0.0

        rows.append({
            "cluster_id": cid,
            "referral_cycle_ratio": round(referral_cycle_ratio, 4),
            "referral_resource_overlap_ratio": round(referral_resource_overlap_ratio, 4),
            "median_referral_activation_days": round(median_referral_activation_days, 2),
            "within_cluster_referral_density": round(within_cluster_referral_density, 4),
        })

    return pd.DataFrame(rows)


REFERRAL_FEATURE_COLS = [
    "referral_cycle_ratio",
    "referral_resource_overlap_ratio",
    "median_referral_activation_days",
    "within_cluster_referral_density",
]


if __name__ == "__main__":
    DATA_DIR = Path("day1_data")
    print("referral_features.py -- standalone validation\n")

    clusters = pd.read_csv("clusters.csv")
    referrals = pd.read_csv(DATA_DIR / "referrals.csv")
    orders = pd.read_csv(DATA_DIR / "orders.csv")

    assert "is_ring_referral" not in referrals.columns, \
        "LEAKAGE: referrals.csv must not contain is_ring_referral"
    print("[OK] referrals.csv has no label column.")

    ref_features = compute_referral_features(clusters, referrals, orders)
    print(f"Clusters with referral activity: {len(ref_features)} "
          f"of {clusters['cluster_id'].nunique()}")
    print(f"\nFeature stats:\n{ref_features[REFERRAL_FEATURE_COLS].describe().round(3)}\n")

    # Sanity assertions
    assert ref_features["referral_cycle_ratio"].between(0.0, 1.0).all()
    print("[OK] referral_cycle_ratio in [0, 1].")
    assert ref_features["referral_resource_overlap_ratio"].between(0.0, 1.0).all()
    print("[OK] referral_resource_overlap_ratio in [0, 1].")
    assert (ref_features["median_referral_activation_days"] >= 0).all()
    print("[OK] median_referral_activation_days >= 0.")
    assert ref_features["within_cluster_referral_density"].between(0.0, 1.0).all()
    print("[OK] within_cluster_referral_density in [0, 1].")
    assert "is_ring_referral" not in ref_features.columns
    print("[OK] No label column in feature output.\n")

    n_cycles = (ref_features["referral_cycle_ratio"] > 0).sum()
    print(f"Clusters with directed cycles: {n_cycles} "
          f"(expected ~{int(0.35 * 8)} from RING_REFERRAL_CYCLE_PROB=0.35)")

    # Post-hoc evaluation (eval only -- labels NOT used in features)
    rgt_path = DATA_DIR / "referral_ground_truth.csv"
    gt_path = DATA_DIR / "ground_truth.csv"
    if rgt_path.exists() and gt_path.exists():
        gt = pd.read_csv(gt_path)
        rref_members = set(gt.loc[gt["is_referral_ring_member"] == True, "account_id"])
        cluster_is_rref = clusters.groupby("cluster_id")["account_id"].apply(
            lambda ids: ids.isin(rref_members).mean() > 0.5
        )
        rref_cluster_ids = set(cluster_is_rref[cluster_is_rref].index)
        print(f"\n--- Post-hoc evaluation ---")
        print(f"Referral-ring clusters (majority members): {len(rref_cluster_ids)}")
        rf_idx = ref_features.set_index("cluster_id")
        for feat in REFERRAL_FEATURE_COLS:
            rref_vals = rf_idx.loc[rf_idx.index.isin(rref_cluster_ids), feat]
            other_vals = rf_idx.loc[~rf_idx.index.isin(rref_cluster_ids), feat]
            print(f"  {feat}: ring mean={rref_vals.mean():.3f}  "
                  f"other mean={other_vals.mean():.3f}")

    ref_features.to_csv("referral_cluster_features.csv", index=False)
    print(f"\nSaved referral_cluster_features.csv  ({len(ref_features)} rows)")

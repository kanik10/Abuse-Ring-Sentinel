"""
Day 3 — feature_engineering.py

Turns each Day-2 cluster (size >= 2) into a row of features describing
*how it looks structurally and behaviorally* — not who's in it. These
features are what the Day 3 classifier learns from; ground_truth.csv is
merged in separately, only as the training/evaluation label, never as an
input feature.
"""

import networkx as nx
import pandas as pd

from graph_builder import (CENTRALITY_COLUMNS, build_account_graph,
                           compute_account_centrality_features)

DATA_DIR = "day1_data"


def compute_cluster_features(clusters: pd.DataFrame,
                              accounts: pd.DataFrame,
                              orders: pd.DataFrame,
                              account_device: pd.DataFrame,
                              account_payment: pd.DataFrame,
                              account_address: pd.DataFrame,
                              account_ip: pd.DataFrame = None,
                              G: nx.Graph = None,
                              referral_features: pd.DataFrame = None) -> pd.DataFrame:
    """Compute cluster-level features.  If referral_features is supplied
    (a DataFrame from referral_features.py with cluster_id + 4 referral
    columns), it is left-joined onto the output and NaN filled with 0.
    Clusters with no referral activity get 0 for all referral features."""
    if G is None and isinstance(account_ip, nx.Graph):
        G = account_ip
        account_ip = None
    if G is None:
        raise ValueError("G must be provided for structural cluster features.")

    accounts = accounts.copy()
    accounts["creation_date"] = pd.to_datetime(accounts["creation_date"])
    account_centrality = (
        compute_account_centrality_features(G)
        .set_index("account_id")
        .reindex(columns=CENTRALITY_COLUMNS)
    )

    flagged = clusters[clusters.cluster_size >= 2]
    rows = []

    for cluster_id, group in flagged.groupby("cluster_id"):
        members = group["account_id"].tolist()
        n = len(members)

        # --- creation-time burstiness ---
        member_creation = accounts[accounts.account_id.isin(members)]["creation_date"]
        creation_span_days = (member_creation.max() - member_creation.min()).days
        creation_std_days = member_creation.astype("int64").std() / 1e9 / 86400 if n > 1 else 0.0

        # --- order behavior ---
        member_orders = orders[orders.account_id.isin(members)]
        avg_order_amount = member_orders["amount"].mean() if len(member_orders) else 0.0
        std_order_amount = member_orders["amount"].std() if len(member_orders) > 1 else 0.0
        order_amount_cv = (std_order_amount / avg_order_amount) if avg_order_amount else 0.0

        # --- resource usage within this cluster ---
        dev_rows = account_device[account_device.account_id.isin(members)]
        pay_rows = account_payment[account_payment.account_id.isin(members)]
        addr_rows = account_address[account_address.account_id.isin(members)]
        ip_rows = (account_ip[account_ip.account_id.isin(members)]
                   if account_ip is not None else pd.DataFrame(columns=["account_id", "ip_id"]))

        total_usages = len(dev_rows) + len(pay_rows) + len(addr_rows) + len(ip_rows)
        distinct_resources = (dev_rows.device_id.nunique()
                               + pay_rows.payment_id.nunique()
                               + addr_rows.address_id.nunique()
                               + ip_rows.ip_id.nunique())
        entity_reuse_ratio = 1 - (distinct_resources / total_usages) if total_usages else 0.0

        # NOTE: avg_resource_degree and fraction_unique_payment were dropped
        # after diagnostic analysis showed they correlate 0.7-0.94 with
        # entity_reuse_ratio/internal_density/cluster_size -- real signal,
        # but almost entirely redundant with features already in the model.
        # Kept out to avoid a bloated, harder-to-defend feature set.

        # --- structural density within the cluster ---
        subgraph = G.subgraph(members)
        density = nx.density(subgraph) if n > 1 else 0.0
        member_centrality = account_centrality.reindex(members).fillna(0.0)

        rows.append({
            "cluster_id": cluster_id,
            "cluster_size": n,
            "entity_reuse_ratio": entity_reuse_ratio,
            "internal_density": density,
            "mean_degree_centrality": member_centrality["degree_centrality"].mean(),
            "max_degree_centrality": member_centrality["degree_centrality"].max(),
            "mean_pagerank": member_centrality["pagerank"].mean(),
            "max_pagerank": member_centrality["pagerank"].max(),
            "mean_betweenness_centrality": member_centrality["betweenness_centrality"].mean(),
            "max_betweenness_centrality": member_centrality["betweenness_centrality"].max(),
            "creation_span_days": creation_span_days,
            "creation_std_days": creation_std_days,
            "avg_order_amount": avg_order_amount,
            "order_amount_cv": order_amount_cv,
        })

    df = pd.DataFrame(rows)

    # Optionally join referral features (4 new columns, 0 for clusters with
    # no referral activity so the feature set stays consistent)
    if referral_features is not None and not referral_features.empty:
        from referral_features import REFERRAL_FEATURE_COLS
        df = df.merge(
            referral_features[["cluster_id"] + REFERRAL_FEATURE_COLS],
            on="cluster_id", how="left",
        )
        df[REFERRAL_FEATURE_COLS] = df[REFERRAL_FEATURE_COLS].fillna(0.0)

    return df


if __name__ == "__main__":
    from pathlib import Path
    clusters = pd.read_csv("clusters.csv")
    accounts = pd.read_csv(f"{DATA_DIR}/accounts.csv")
    orders = pd.read_csv(f"{DATA_DIR}/orders.csv")
    account_device = pd.read_csv(f"{DATA_DIR}/resolved_account_device.csv")
    account_payment = pd.read_csv(f"{DATA_DIR}/resolved_account_payment.csv")
    account_address = pd.read_csv(f"{DATA_DIR}/resolved_account_address.csv")
    account_ip = pd.read_csv(f"{DATA_DIR}/resolved_account_ip.csv")

    G = build_account_graph(account_device, account_payment, account_address, account_ip)

    # Load referral features if available (run referral_features.py first)
    ref_features = None
    if Path("referral_cluster_features.csv").exists():
        ref_features = pd.read_csv("referral_cluster_features.csv")
        print(f"Loaded referral_cluster_features.csv ({len(ref_features)} rows)")

    features = compute_cluster_features(
        clusters, accounts, orders,
        account_device, account_payment, account_address,
        account_ip=account_ip, G=G,
        referral_features=ref_features,
    )
    features.to_csv("cluster_features.csv", index=False)
    print(f"Wrote cluster_features.csv -- {len(features)} clusters, "
          f"{features.shape[1] - 1} features each.")
    print(features.describe().T[["mean", "std", "min", "max"]])

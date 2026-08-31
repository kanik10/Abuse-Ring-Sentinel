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

from graph_builder import build_account_graph

DATA_DIR = "day1_data"


def compute_cluster_features(clusters: pd.DataFrame,
                              accounts: pd.DataFrame,
                              orders: pd.DataFrame,
                              account_device: pd.DataFrame,
                              account_payment: pd.DataFrame,
                              account_address: pd.DataFrame,
                              G: nx.Graph) -> pd.DataFrame:
    accounts = accounts.copy()
    accounts["creation_date"] = pd.to_datetime(accounts["creation_date"])

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

        total_usages = len(dev_rows) + len(pay_rows) + len(addr_rows)
        distinct_resources = (dev_rows.device_id.nunique()
                               + pay_rows.payment_id.nunique()
                               + addr_rows.address_id.nunique())
        entity_reuse_ratio = 1 - (distinct_resources / total_usages) if total_usages else 0.0

        # NOTE: avg_resource_degree and fraction_unique_payment were dropped
        # after diagnostic analysis showed they correlate 0.7-0.94 with
        # entity_reuse_ratio/internal_density/cluster_size -- real signal,
        # but almost entirely redundant with features already in the model.
        # Kept out to avoid a bloated, harder-to-defend feature set.

        # --- structural density within the cluster ---
        subgraph = G.subgraph(members)
        density = nx.density(subgraph) if n > 1 else 0.0

        rows.append({
            "cluster_id": cluster_id,
            "cluster_size": n,
            "entity_reuse_ratio": entity_reuse_ratio,
            "internal_density": density,
            "creation_span_days": creation_span_days,
            "creation_std_days": creation_std_days,
            "avg_order_amount": avg_order_amount,
            "order_amount_cv": order_amount_cv,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    clusters = pd.read_csv("clusters.csv")
    accounts = pd.read_csv(f"{DATA_DIR}/accounts.csv")
    orders = pd.read_csv(f"{DATA_DIR}/orders.csv")
    account_device = pd.read_csv(f"{DATA_DIR}/account_device.csv")
    account_payment = pd.read_csv(f"{DATA_DIR}/account_payment.csv")
    account_address = pd.read_csv(f"{DATA_DIR}/account_address.csv")

    G = build_account_graph(account_device, account_payment, account_address)

    features = compute_cluster_features(
        clusters, accounts, orders,
        account_device, account_payment, account_address, G,
    )
    features.to_csv("cluster_features.csv", index=False)
    print(f"Wrote cluster_features.csv — {len(features)} clusters, "
          f"{features.shape[1] - 1} features each.")
    print(features.describe().T[["mean", "std", "min", "max"]])
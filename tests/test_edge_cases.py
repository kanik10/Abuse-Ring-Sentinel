"""
test_edge_cases.py -- Edge case and resilience tests across the pipeline.

Verifies:
  - Empty graph inputs to centrality and graph construction functions.
  - Single-account clusters (cluster_size == 1) in feature engineering and account scoring.
  - Duplicate resource entries (e.g. repeated logs for the same account and device).
  - Account with no orders gets neutral order amount centrality (0.5).
  - Cluster members with identical creation dates get neutral creation date centrality (0.5).
"""

import networkx as nx
import pandas as pd
import pytest

from account_scoring import compute_account_features, score_accounts_in_cluster
from feature_engineering import compute_cluster_features
from graph_builder import build_account_graph, compute_account_centrality_features


def test_empty_graph_input():
    """Empty graph input returns empty features DataFrame without raising exceptions."""
    G_empty = nx.Graph()
    features = compute_account_centrality_features(G_empty)
    assert len(features) == 0
    assert "account_id" in features.columns
    assert "degree_centrality" in features.columns


def test_single_account_cluster_excluded_from_features():
    """Clusters of size < 2 are strictly excluded in feature_engineering."""
    clusters = pd.DataFrame([{"cluster_id": 999, "account_id": "LONE_WOLF", "cluster_size": 1}])
    accounts = pd.DataFrame([{"account_id": "LONE_WOLF", "creation_date": "2024-01-01"}])
    orders = pd.DataFrame([{"account_id": "LONE_WOLF", "amount": 50.0}])
    empty = pd.DataFrame(columns=["account_id", "res_id"])
    G = nx.Graph()
    G.add_node("LONE_WOLF")

    features_df = compute_cluster_features(
        clusters=clusters,
        accounts=accounts,
        orders=orders,
        account_device=empty.rename(columns={"res_id": "device_id"}),
        account_payment=empty.rename(columns={"res_id": "payment_id"}),
        account_address=empty.rename(columns={"res_id": "address_id"}),
        G=G,
    )
    # Output must be empty since cluster_size < 2
    assert len(features_df) == 0


def test_single_account_cluster_account_scoring():
    """Single-account cluster in account_scoring safely produces neutral scores."""
    G = nx.Graph()
    G.add_node("SOLO_ACC")

    accounts = pd.DataFrame([{"account_id": "SOLO_ACC", "creation_date": "2024-01-01"}])
    orders = pd.DataFrame([{"account_id": "SOLO_ACC", "amount": 100.0}])
    empty = pd.DataFrame(columns=["account_id", "res_id"])

    df = compute_account_features(
        cluster_id=1,
        member_ids=["SOLO_ACC"],
        G=G,
        account_device=empty.rename(columns={"res_id": "device_id"}),
        account_payment=empty.rename(columns={"res_id": "payment_id"}),
        account_address=empty.rename(columns={"res_id": "address_id"}),
        account_ip=empty.rename(columns={"res_id": "ip_id"}),
        accounts=accounts,
        orders=orders,
    )

    assert len(df) == 1
    assert df.iloc[0]["creation_date_centrality"] == 0.5
    assert df.iloc[0]["order_amount_centrality"] == 0.5
    assert df.iloc[0]["n_shared_resources"] == 0


def test_duplicate_resource_entries():
    """Duplicate rows in mapping table (e.g. repeated logins on same device) do not create self-loops."""
    # A logged in 3 times on DEV_1, B logged in 2 times on DEV_1
    device_df = pd.DataFrame([
        {"account_id": "A", "device_id": "DEV_1"},
        {"account_id": "A", "device_id": "DEV_1"},
        {"account_id": "A", "device_id": "DEV_1"},
        {"account_id": "B", "device_id": "DEV_1"},
        {"account_id": "B", "device_id": "DEV_1"},
    ])

    empty = pd.DataFrame(columns=["account_id", "res_id"])
    G = build_account_graph(
        account_device=device_df,
        account_payment=empty.rename(columns={"res_id": "payment_id"}),
        account_address=empty.rename(columns={"res_id": "address_id"}),
    )

    # Must contain edge between A and B, and strictly no self loops
    assert G.has_edge("A", "B")
    assert not G.has_edge("A", "A")
    assert not G.has_edge("B", "B")


def test_account_with_no_orders_neutral_centrality():
    """An account with no orders receives neutral 0.5 order_amount_centrality."""
    members = ["ACC_HAS_ORDERS", "ACC_NO_ORDERS"]
    clusters = pd.DataFrame([{"cluster_id": 1, "account_id": m} for m in members])
    accounts = pd.DataFrame([{"account_id": m, "creation_date": "2024-01-01"} for m in members])
    orders = pd.DataFrame([{"account_id": "ACC_HAS_ORDERS", "amount": 100.0}])
    empty = pd.DataFrame(columns=["account_id", "res_id"])
    G = nx.Graph()
    G.add_edge("ACC_HAS_ORDERS", "ACC_NO_ORDERS", weight=1.0)

    df = compute_account_features(
        cluster_id=1,
        member_ids=members,
        G=G,
        account_device=empty.rename(columns={"res_id": "device_id"}),
        account_payment=empty.rename(columns={"res_id": "payment_id"}),
        account_address=empty.rename(columns={"res_id": "address_id"}),
        account_ip=empty.rename(columns={"res_id": "ip_id"}),
        accounts=accounts,
        orders=orders,
    )

    no_order_row = df[df.account_id == "ACC_NO_ORDERS"].iloc[0]
    assert no_order_row["order_amount_centrality"] == 0.5


def test_identical_creation_dates_neutral_centrality():
    """When all accounts in a cluster share identical creation dates, creation centrality is 0.5."""
    members = ["ACC_A", "ACC_B"]
    accounts = pd.DataFrame([{"account_id": m, "creation_date": "2024-01-01 12:00:00"} for m in members])
    orders = pd.DataFrame([{"account_id": m, "amount": 100.0} for m in members])
    empty = pd.DataFrame(columns=["account_id", "res_id"])
    G = nx.Graph()
    G.add_edge("ACC_A", "ACC_B", weight=1.0)

    df = compute_account_features(
        cluster_id=1,
        member_ids=members,
        G=G,
        account_device=empty.rename(columns={"res_id": "device_id"}),
        account_payment=empty.rename(columns={"res_id": "payment_id"}),
        account_address=empty.rename(columns={"res_id": "address_id"}),
        account_ip=empty.rename(columns={"res_id": "ip_id"}),
        accounts=accounts,
        orders=orders,
    )

    assert (df["creation_date_centrality"] == 0.5).all()

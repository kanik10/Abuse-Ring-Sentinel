"""
test_account_scoring.py -- Unit tests for account_scoring.py.

Verifies:
  - Synthetic "hub" account ranks strictly above synthetic "peripheral" account in the same cluster.
  - _minmax_normalize assigns 0.5 (neutral) when all values in a series are identical.
  - _minmax_normalize correctly scales varying series to [0, 1].
  - Single-account cluster edge case is handled safely without division-by-zero errors.
  - score_all_flagged_accounts correctly processes multiple clusters.
"""

import networkx as nx
import pandas as pd
import pytest

from account_scoring import (
    _minmax_normalize,
    compute_account_features,
    score_accounts_in_cluster,
    score_all_flagged_accounts,
)


def test_hub_account_ranked_above_peripheral(synthetic_hub_cluster_data):
    """
    In a cluster with a central hub (sharing devices/payments with all others)
    and peripheral members (connected only to the hub), the hub account must
    rank strictly higher in account_risk_score than peripheral accounts.
    """
    d = synthetic_hub_cluster_data

    scored = score_accounts_in_cluster(
        cluster_id=d["cluster_id"],
        member_ids=d["members"],
        G=d["G"],
        account_device=d["device"],
        account_payment=d["payment"],
        account_address=d["address"],
        account_ip=d["ip"],
        accounts=d["accounts"],
        orders=d["orders"],
    )

    # Top-ranked account must be ACC_HUB
    assert scored.iloc[0]["account_id"] == "ACC_HUB"

    hub_score = scored.loc[scored.account_id == "ACC_HUB", "account_risk_score"].iloc[0]
    periph_score = scored.loc[scored.account_id == "ACC_PERIPH_1", "account_risk_score"].iloc[0]

    assert hub_score > periph_score
    assert scored.loc[scored.account_id == "ACC_HUB", "within_cluster_degree"].iloc[0] == 3
    assert scored.loc[scored.account_id == "ACC_PERIPH_1", "within_cluster_degree"].iloc[0] == 1


def test_minmax_normalize_constant_series():
    """When all series values are equal, returns 0.5 for all elements (neutral, not 0/1/NaN)."""
    series = pd.Series([42.0, 42.0, 42.0])
    normalized = _minmax_normalize(series)
    assert (normalized == 0.5).all()


def test_minmax_normalize_varying_series():
    """Varying series is scaled to [0, 1] with min -> 0 and max -> 1."""
    series = pd.Series([10.0, 20.0, 30.0])
    normalized = _minmax_normalize(series)
    assert normalized.iloc[0] == 0.0
    assert normalized.iloc[1] == 0.5
    assert normalized.iloc[2] == 1.0


def test_single_account_cluster_edge_case():
    """Cluster with 1 member must compute neutral/safe scores without ZeroDivisionError."""
    G = nx.Graph()
    G.add_node("SOLO_ACC")

    empty = pd.DataFrame(columns=["account_id", "res_id"])
    accounts = pd.DataFrame([{"account_id": "SOLO_ACC", "creation_date": "2024-01-01"}])
    orders = pd.DataFrame([{"account_id": "SOLO_ACC", "amount": 100.0}])

    scored = score_accounts_in_cluster(
        cluster_id=99,
        member_ids=["SOLO_ACC"],
        G=G,
        account_device=empty.rename(columns={"res_id": "device_id"}),
        account_payment=empty.rename(columns={"res_id": "payment_id"}),
        account_address=empty.rename(columns={"res_id": "address_id"}),
        account_ip=empty.rename(columns={"res_id": "ip_id"}),
        accounts=accounts,
        orders=orders,
    )

    assert len(scored) == 1
    assert scored.iloc[0]["account_id"] == "SOLO_ACC"
    assert scored.iloc[0]["n_shared_resources"] == 0
    assert scored.iloc[0]["within_cluster_degree"] == 0
    assert scored.iloc[0]["account_risk_score"] >= 0.0


def test_score_all_flagged_accounts(synthetic_hub_cluster_data):
    """score_all_flagged_accounts returns scores for all members across flagged clusters."""
    d = synthetic_hub_cluster_data
    all_scored = score_all_flagged_accounts(
        flagged_cluster_ids=[20],
        clusters=d["clusters"],
        G=d["G"],
        account_device=d["device"],
        account_payment=d["payment"],
        account_address=d["address"],
        account_ip=d["ip"],
        accounts=d["accounts"],
        orders=d["orders"],
    )

    assert len(all_scored) == 4
    assert set(all_scored["account_id"]) == set(d["members"])
    assert all_scored.iloc[0]["account_id"] == "ACC_HUB"

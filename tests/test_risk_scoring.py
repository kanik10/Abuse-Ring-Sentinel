"""
test_risk_scoring.py -- Unit tests for risk_scoring.py.

Verifies:
  - Defense-only structural guarantee: RecommendedAction enum has strictly ONE member (FLAG_FOR_REVIEW).
  - No offensive capabilities (BLOCK, FREEZE, CANCEL, RATE_LIMIT) exist in RecommendedAction.
  - ClusterRiskOutput dataclass serialization to dictionary/JSON.
  - find_shared_resources detects resources shared by >= 2 cluster members.
  - score_all_clusters filters out clusters below CHOSEN_THRESHOLD.
  - train_final_model trains a valid scikit-learn Pipeline.
"""

from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest

from risk_scoring import (
    CHAMPION_FEATURE_COLS,
    ClusterRiskOutput,
    RecommendedAction,
    find_shared_resources,
    score_all_clusters,
    train_final_model,
)
from threshold_config import CHOSEN_THRESHOLD


def test_recommended_action_structurally_constrained():
    """
    Defense-only architectural enforcement:
    RecommendedAction must have EXACTLY ONE member (FLAG_FOR_REVIEW).
    There must be no BLOCK, FREEZE, CANCEL, or RATE_LIMIT actions anywhere in the project.
    """
    members = list(RecommendedAction)
    assert len(members) == 1
    assert members[0] == RecommendedAction.FLAG_FOR_REVIEW
    assert RecommendedAction.FLAG_FOR_REVIEW.value == "flag_for_review"

    offensive_actions = ["BLOCK", "FREEZE", "CANCEL", "RATE_LIMIT", "TERMINATE"]
    for act in offensive_actions:
        assert act not in RecommendedAction.__members__


def test_cluster_risk_output_json_serialization():
    """ClusterRiskOutput serializes cleanly to dict with string action."""
    output = ClusterRiskOutput(
        cluster_id=1,
        risk_score=0.9542,
        member_account_ids=["A", "B"],
        shared_resources=["device:DEV_01 (used by 2 members)"],
        contributing_features={"cluster_size": 2, "entity_reuse_ratio": 0.5},
    )

    d = output.to_json_dict()
    assert d["cluster_id"] == 1
    assert d["risk_score"] == 0.9542
    assert d["recommended_action"] == "flag_for_review"
    assert isinstance(d["shared_resources"], list)
    assert isinstance(d["contributing_features"], dict)


def test_find_shared_resources():
    """find_shared_resources must detect resources shared by >= 2 cluster members."""
    clusters = pd.DataFrame([
        {"cluster_id": 5, "account_id": "MEMBER_A"},
        {"cluster_id": 5, "account_id": "MEMBER_B"},
        {"cluster_id": 5, "account_id": "MEMBER_C"},
    ])

    device_df = pd.DataFrame([
        {"account_id": "MEMBER_A", "device_id": "DEV_SHARED"},
        {"account_id": "MEMBER_B", "device_id": "DEV_SHARED"},
        {"account_id": "MEMBER_C", "device_id": "DEV_UNIQUE_C"},
    ])

    empty_df = pd.DataFrame(columns=["account_id", "resource_id"])

    shared = find_shared_resources(
        cluster_id=5,
        clusters=clusters,
        account_device=device_df,
        account_payment=empty_df.rename(columns={"resource_id": "payment_id"}),
        account_address=empty_df.rename(columns={"resource_id": "address_id"}),
        account_ip=empty_df.rename(columns={"resource_id": "ip_id"}),
    )

    assert len(shared) == 1
    assert "device:DEV_SHARED (used by 2 members)" in shared[0]


def test_score_all_clusters_filters_below_threshold():
    """Clusters with risk score below CHOSEN_THRESHOLD must be excluded from output."""
    clusters = pd.DataFrame([
        {"cluster_id": 1, "account_id": "A1"},
        {"cluster_id": 1, "account_id": "A2"},
        {"cluster_id": 2, "account_id": "B1"},
        {"cluster_id": 2, "account_id": "B2"},
    ])

    features = pd.DataFrame([
        {"cluster_id": 1, **{col: 1.0 for col in CHAMPION_FEATURE_COLS}},
        {"cluster_id": 2, **{col: 0.0 for col in CHAMPION_FEATURE_COLS}},
    ])

    # Mock model: cluster 1 has score 0.85 (>= threshold), cluster 2 has score 0.02 (< threshold)
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([
        [0.15, 0.85],
        [0.98, 0.02],
    ])

    empty = pd.DataFrame(columns=["account_id", "res_id"])
    outputs = score_all_clusters(
        features=features,
        clusters=clusters,
        model=mock_model,
        account_device=empty.rename(columns={"res_id": "device_id"}),
        account_payment=empty.rename(columns={"res_id": "payment_id"}),
        account_address=empty.rename(columns={"res_id": "address_id"}),
        account_ip=empty.rename(columns={"res_id": "ip_id"}),
    )

    # Only cluster 1 should be returned
    assert len(outputs) == 1
    assert outputs[0].cluster_id == 1
    assert outputs[0].risk_score == 0.85


def test_train_final_model():
    """train_final_model returns a trained Pipeline on CHAMPION_FEATURE_COLS."""
    # Synthetic feature rows: 2 positive, 2 negative
    data = []
    for i in range(4):
        val = 1.0 if i < 2 else 0.0
        data.append({"cluster_id": i + 1, **{col: val for col in CHAMPION_FEATURE_COLS}})
    features = pd.DataFrame(data)
    labels = pd.Series([1, 1, 0, 0])

    model = train_final_model(features, labels)
    preds = model.predict(features[CHAMPION_FEATURE_COLS].values)
    assert len(preds) == 4

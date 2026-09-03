"""
test_classifier.py -- Unit tests for classifier.py.

Verifies:
  - build_labels follows the majority rule (> 0.5 members positive -> cluster labeled 1).
  - Backward compatibility with and without is_referral_ring_member in ground truth.
  - evaluate computes precision, recall, f1, and PR-AUC.
"""

import numpy as np
import pandas as pd
import pytest

from classifier import build_labels, evaluate


def test_build_labels_majority_rule():
    """Cluster is labeled 1 if >50% of its members are ring members, else 0."""
    features = pd.DataFrame([{"cluster_id": 1}, {"cluster_id": 2}, {"cluster_id": 3}])

    clusters = pd.DataFrame([
        # Cluster 1: 2 out of 3 are ring members (2/3 > 0.5 -> 1)
        {"cluster_id": 1, "account_id": "C1_A"},
        {"cluster_id": 1, "account_id": "C1_B"},
        {"cluster_id": 1, "account_id": "C1_C"},

        # Cluster 2: 1 out of 3 is ring member (1/3 <= 0.5 -> 0)
        {"cluster_id": 2, "account_id": "C2_A"},
        {"cluster_id": 2, "account_id": "C2_B"},
        {"cluster_id": 2, "account_id": "C2_C"},

        # Cluster 3: 0 out of 2 are ring members (0.0 <= 0.5 -> 0)
        {"cluster_id": 3, "account_id": "C3_A"},
        {"cluster_id": 3, "account_id": "C3_B"},
    ])

    ground_truth = pd.DataFrame([
        {"account_id": "C1_A", "is_ring_member": True},
        {"account_id": "C1_B", "is_ring_member": True},
        {"account_id": "C1_C", "is_ring_member": False},
        {"account_id": "C2_A", "is_ring_member": True},
        {"account_id": "C2_B", "is_ring_member": False},
        {"account_id": "C2_C", "is_ring_member": False},
        {"account_id": "C3_A", "is_ring_member": False},
        {"account_id": "C3_B", "is_ring_member": False},
    ])

    labels = build_labels(features, clusters, ground_truth)
    assert list(labels) == [1, 0, 0]


def test_build_labels_with_referral_ring_member():
    """is_referral_ring_member is combined with is_ring_member via boolean OR."""
    features = pd.DataFrame([{"cluster_id": 1}])
    clusters = pd.DataFrame([
        {"cluster_id": 1, "account_id": "A1"},
        {"cluster_id": 1, "account_id": "A2"},
    ])

    ground_truth = pd.DataFrame([
        {"account_id": "A1", "is_ring_member": False, "is_referral_ring_member": True},
        {"account_id": "A2", "is_ring_member": True, "is_referral_ring_member": False},
    ])

    labels = build_labels(features, clusters, ground_truth)
    assert list(labels) == [1]


def test_evaluate_metrics():
    """evaluate returns (precision, recall, f1, pr_auc) consistent with ground truth."""
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.1])

    p, r, f1, ap = evaluate(y_true, y_prob, threshold=0.5, name="test_model")
    assert p == 1.0
    assert r == 1.0
    assert f1 == 1.0
    assert ap == 1.0

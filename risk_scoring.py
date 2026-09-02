"""
Day 5 Phase 1 — risk_scoring.py

Structural defense-only guarantee: RecommendedAction below has exactly ONE
member. There is no BLOCK, FREEZE, CANCEL, or RATE_LIMIT anywhere in this
codebase. Adding one would be a visible, one-line change in any code
review -- this system cannot silently gain an offensive capability, because
the type system doesn't have room for one.

Every flagged cluster carries evidence: the feature values that drove its
score, and the actual shared resources connecting its members. A human
reviewer acts on the evidence; this module never acts on anything itself.
"""

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from graph_builder import build_account_graph
from account_scoring import score_all_flagged_accounts
from threshold_config import CHOSEN_THRESHOLD  # reads pooled_threshold_selection_summary.json;
                                               # run pooled_threshold_selection.py to update.

DATA_DIR = "day1_data"
PURE_GRAPH_FEATURES = ["cluster_size", "entity_reuse_ratio", "internal_density"]

class RecommendedAction(Enum):
    """Deliberately the ONLY member this enum will ever have in this
    project. This is the structural enforcement of defense-only: there is
    no code path, anywhere, that can assign a cluster anything other than
    a request for human review."""
    FLAG_FOR_REVIEW = "flag_for_review"


@dataclass
class ClusterRiskOutput:
    cluster_id: int
    risk_score: float
    member_account_ids: List[str]
    shared_resources: List[str]
    contributing_features: dict
    # Per-account scores within this cluster, sorted by account_risk_score DESC.
    # Each entry is a dict with keys: account_id, account_risk_score, and the 5
    # raw account-local features. Populated by risk_scoring.main() after the
    # cluster-level pass; empty list if account scoring is unavailable.
    account_scores: List[dict] = field(default_factory=list)
    recommended_action: RecommendedAction = RecommendedAction.FLAG_FOR_REVIEW

    def to_json_dict(self) -> dict:
        d = asdict(self)
        d["recommended_action"] = self.recommended_action.value
        return d


def train_final_model(features: pd.DataFrame, labels: pd.Series):
    """The model actually shipped -- fit on all current clusters, not a CV fold.
    CV in Day 3/4 was for honest evaluation; this is the artifact a real
    deployment would load."""
    X = features[PURE_GRAPH_FEATURES].values
    y = labels.values
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    model.fit(X, y)
    return model


def find_shared_resources(cluster_id: int, clusters: pd.DataFrame,
                           account_device: pd.DataFrame, account_payment: pd.DataFrame,
                           account_address: pd.DataFrame,
                           account_ip: pd.DataFrame) -> List[str]:
    members = clusters.loc[clusters.cluster_id == cluster_id, "account_id"].tolist()
    shared = []
    for name, df, col in [("device", account_device, "device_id"),
                           ("payment", account_payment, "payment_id"),
                           ("address", account_address, "address_id"),
                           ("ip", account_ip, "ip_id")]:
        sub = df[df.account_id.isin(members)]
        counts = sub.groupby(col)["account_id"].nunique()
        for resource_id, n in counts[counts >= 2].items():
            shared.append(f"{name}:{resource_id} (used by {n} members)")
    return shared


def score_all_clusters(features: pd.DataFrame, clusters: pd.DataFrame, model,
                        account_device, account_payment, account_address,
                        account_ip, G=None,
                        accounts: pd.DataFrame = None,
                        orders: pd.DataFrame = None) -> List[ClusterRiskOutput]:
    """Cluster-level pass: score every cluster and keep those above threshold.
    If G, accounts, and orders are supplied, also runs per-account scoring
    (account_scoring.score_all_flagged_accounts) and attaches the results to
    each ClusterRiskOutput.account_scores."""
    X = features[PURE_GRAPH_FEATURES].values
    risk_scores = model.predict_proba(X)[:, 1]

    outputs = []
    for i, row in features.iterrows():
        score = float(risk_scores[i])
        if score < CHOSEN_THRESHOLD:
            continue  # below threshold -> no output at all, not even a low-priority flag

        cluster_id = int(row["cluster_id"])
        members = clusters.loc[clusters.cluster_id == cluster_id, "account_id"].tolist()
        shared = find_shared_resources(
            cluster_id, clusters, account_device, account_payment, account_address, account_ip
        )

        outputs.append(ClusterRiskOutput(
            cluster_id=cluster_id,
            risk_score=round(score, 4),
            member_account_ids=members,
            shared_resources=shared,
            contributing_features={f: round(float(row[f]), 4) for f in PURE_GRAPH_FEATURES},
        ))

    # Per-account scoring pass (runs only if graph + data are available)
    if G is not None and accounts is not None and orders is not None and outputs:
        flagged_ids = [o.cluster_id for o in outputs]
        account_scored = score_all_flagged_accounts(
            flagged_ids, clusters, G,
            account_device, account_payment, account_address, account_ip,
            accounts, orders,
        )
        # Index by cluster_id for fast lookup
        acct_by_cluster = {
            cid: grp.drop(columns=["cluster_id"]).to_dict(orient="records")
            for cid, grp in account_scored.groupby("cluster_id")
        }
        for o in outputs:
            o.account_scores = acct_by_cluster.get(o.cluster_id, [])

    return outputs


def main():
    features = pd.read_csv("cluster_features.csv")
    clusters = pd.read_csv("clusters.csv")
    predictions = pd.read_csv("cluster_predictions.csv")
    accounts = pd.read_csv(f"{DATA_DIR}/accounts.csv")
    orders = pd.read_csv(f"{DATA_DIR}/orders.csv")
    account_device = pd.read_csv(f"{DATA_DIR}/resolved_account_device.csv")
    account_payment = pd.read_csv(f"{DATA_DIR}/resolved_account_payment.csv")
    account_address = pd.read_csv(f"{DATA_DIR}/resolved_account_address.csv")
    account_ip = pd.read_csv(f"{DATA_DIR}/resolved_account_ip.csv")

    G = build_account_graph(account_device, account_payment, account_address, account_ip)

    labels = predictions.set_index("cluster_id").loc[features.cluster_id, "y_true_is_ring"].reset_index(drop=True)

    model = train_final_model(features, labels)
    joblib.dump(model, "final_model.joblib")

    outputs = score_all_clusters(
        features, clusters, model, account_device, account_payment, account_address,
        account_ip, G=G, accounts=accounts, orders=orders,
    )

    print(f"Flagged {len(outputs)} clusters for review (threshold={CHOSEN_THRESHOLD:.4f})\n")
    for o in outputs[:3]:
        print(json.dumps(o.to_json_dict(), indent=2))
        print()

    with open("audit_log.jsonl", "w", encoding="utf-8") as f:
        for o in outputs:
            f.write(json.dumps(o.to_json_dict()) + "\n")

    print(f"Saved final_model.joblib and audit_log.jsonl ({len(outputs)} flagged clusters)")


if __name__ == "__main__":
    main()

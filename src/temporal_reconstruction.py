"""
temporal_reconstruction.py -- Point-in-time / as-of-T reconstruction.

PHASE 1:
Entity-based ring<->cluster lookup (find_dominant_cluster).
Re-derives "which cluster currently contains ring X's members" fresh at each
snapshot using ground-truth ring membership (eval-only -- never fed into features).

PHASE 2:
Point-in-Time Temporal Reconstruction Engine.
Filters all input tables by timestamp <= T before building the graph, running
Louvain community detection, computing features, and scoring with final_model.joblib.
Demonstrates zero lookahead bias and measures detection latency in days and
fraud order volume prevented before execution.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import community as community_louvain
import joblib
import networkx as nx
import numpy as np
import pandas as pd

from feature_engineering import compute_cluster_features
from graph_builder import build_account_graph
from referral_features import REFERRAL_FEATURE_COLS, compute_referral_features
from risk_scoring import CHAMPION_FEATURE_COLS
from threshold_config import CHOSEN_THRESHOLD

DATA_DIR = Path("day1_data")


def _resolve_data_path(filename: str) -> Path:
    p = DATA_DIR / filename
    if p.exists():
        return p
    fallback = Path(filename)
    if fallback.exists():
        warnings.warn(
            f"File '{p}' not found; falling back to root-level '{fallback}'. "
            "Canonical data should reside in day1_data/.",
            UserWarning,
            stacklevel=2,
        )
        return fallback
    return p


def find_dominant_cluster(
    clusters: pd.DataFrame, target_member_ids: set, min_capture: float = 0.5
) -> dict | None:
    """
    Finds whichever cluster captures the largest share of target_member_ids.

    Parameters
    ----------
    clusters : DataFrame with columns cluster_id, account_id (one row per account).
    target_member_ids : the account_ids we're trying to locate.
    min_capture : minimum fraction of target_member_ids that must land in
        one cluster for that cluster to count as 'dominant' (default 0.5).

    Returns
    -------
    None if not found or threshold not met; otherwise dict with cluster metadata.
    """
    if not target_member_ids:
        return None

    sub = clusters[clusters["account_id"].isin(target_member_ids)]
    if sub.empty:
        return None

    counts = sub.groupby("cluster_id")["account_id"].nunique()
    best_cluster_id = counts.idxmax()
    members_captured = int(counts.loc[best_cluster_id])
    capture_fraction = members_captured / len(target_member_ids)
    if capture_fraction < min_capture:
        return None

    cluster_size = int((clusters["cluster_id"] == best_cluster_id).sum())
    return {
        "cluster_id": best_cluster_id,
        "cluster_size": cluster_size,
        "members_captured": members_captured,
        "target_member_count": len(target_member_ids),
        "capture_fraction": round(capture_fraction, 4),
        "cluster_purity": round(members_captured / cluster_size, 4) if cluster_size else 0.0,
    }


def reconstruct_snapshot(
    as_of_date: pd.Timestamp,
    accounts: pd.DataFrame,
    orders: pd.DataFrame,
    resolved_device: pd.DataFrame,
    resolved_payment: pd.DataFrame,
    resolved_address: pd.DataFrame,
    resolved_ip: pd.DataFrame,
    referrals: Optional[pd.DataFrame],
    model,
) -> Optional[dict]:
    """
    Point-in-time reconstruction for historical timestamp as_of_date (Zero Lookahead Bias).
    Filters input datasets, reconstructs graph, runs Louvain, extracts features, and scores clusters.
    """
    d_sub = resolved_device[resolved_device["first_seen_date"] <= as_of_date].copy()
    p_sub = resolved_payment[resolved_payment["first_seen_date"] <= as_of_date].copy()
    a_sub = resolved_address[resolved_address["first_seen_date"] <= as_of_date].copy()
    i_sub = resolved_ip[resolved_ip["first_seen_date"] <= as_of_date].copy()
    acc_sub = accounts[accounts["creation_date"] <= as_of_date].copy()
    ord_sub = orders[orders["timestamp"] <= as_of_date].copy()
    ref_sub = (
        referrals[referrals["referral_date"] <= as_of_date].copy()
        if referrals is not None and not referrals.empty
        else None
    )

    if len(acc_sub) < 5:
        return None

    G_t = build_account_graph(d_sub, p_sub, a_sub, i_sub)
    if G_t.number_of_nodes() < 2 or G_t.number_of_edges() == 0:
        return None

    part = community_louvain.best_partition(G_t, random_state=42)
    c_df = pd.DataFrame(list(part.items()), columns=["account_id", "cluster_id"])
    c_sizes = c_df.groupby("cluster_id").size().rename("cluster_size")
    c_df = c_df.merge(c_sizes, on="cluster_id")

    c_candidates = c_df[c_df["cluster_size"] >= 2].copy()
    if c_candidates.empty:
        return None

    ref_feat = (
        compute_referral_features(c_candidates, ref_sub, ord_sub)
        if ref_sub is not None and not ref_sub.empty
        else None
    )

    feat_t = compute_cluster_features(
        c_candidates, acc_sub, ord_sub, d_sub, p_sub, a_sub, i_sub, G_t, ref_feat
    )
    if feat_t.empty:
        return None

    for col in CHAMPION_FEATURE_COLS:
        if col not in feat_t.columns:
            feat_t[col] = 0.0
        else:
            feat_t[col] = feat_t[col].fillna(0.0)

    X_t = feat_t[CHAMPION_FEATURE_COLS].values
    feat_t["risk_score"] = model.predict_proba(X_t)[:, 1]

    return {
        "as_of_date": as_of_date,
        "graph": G_t,
        "clusters": c_df,
        "features": feat_t,
        "risk_scores": dict(zip(feat_t["cluster_id"], feat_t["risk_score"])),
        "n_nodes": G_t.number_of_nodes(),
        "n_edges": G_t.number_of_edges(),
    }


def run_temporal_backtest(
    snapshot_freq_days: int = 14,
    threshold: float = CHOSEN_THRESHOLD,
    save_csv: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Executes full point-in-time temporal reconstruction backtest across all fraud rings.
    Measures detection latency in days and fraud volume prevented before execution.
    """
    # Load required data
    dev_path = _resolve_data_path("resolved_account_device.csv")
    pay_path = _resolve_data_path("resolved_account_payment.csv")
    addr_path = _resolve_data_path("resolved_account_address.csv")
    ip_path = _resolve_data_path("resolved_account_ip.csv")
    acc_path = _resolve_data_path("accounts.csv")
    orders_path = _resolve_data_path("orders.csv")
    ref_path = _resolve_data_path("referrals.csv")
    gt_path = _resolve_data_path("ground_truth.csv")

    dev = pd.read_csv(dev_path)
    pay = pd.read_csv(pay_path)
    addr = pd.read_csv(addr_path)
    ip = pd.read_csv(ip_path)
    accounts = pd.read_csv(acc_path)
    orders = pd.read_csv(orders_path)
    referrals = pd.read_csv(ref_path) if ref_path.exists() else None
    gt = pd.read_csv(gt_path)
    model = joblib.load("final_model.joblib")

    # Format datetime columns
    accounts["creation_date"] = pd.to_datetime(accounts["creation_date"])
    orders["timestamp"] = pd.to_datetime(orders["timestamp"])
    dev["first_seen_date"] = pd.to_datetime(dev["first_seen_date"])
    pay["first_seen_date"] = pd.to_datetime(pay["first_seen_date"])
    addr["first_seen_date"] = pd.to_datetime(addr["first_seen_date"])
    ip["first_seen_date"] = pd.to_datetime(ip["first_seen_date"])
    if referrals is not None and not referrals.empty:
        referrals["referral_date"] = pd.to_datetime(referrals["referral_date"])

    acc_dates = dict(zip(accounts["account_id"], accounts["creation_date"]))

    # Date range
    start_date = pd.Timestamp("2024-10-01")
    end_date = pd.Timestamp("2026-08-30")
    snapshot_dates = pd.date_range(start=start_date, end=end_date, freq=f"{snapshot_freq_days}D").tolist()
    if snapshot_dates[-1] < end_date:
        snapshot_dates.append(end_date)

    # Gather ground-truth ring definitions
    resource_rings = sorted(gt.loc[gt["is_ring_member"] == True, "ring_id"].dropna().unique())
    referral_rings = sorted(
        gt.loc[gt.get("is_referral_ring_member", False) == True, "referral_ring_id"].dropna().unique()
    )

    all_rings = []
    for rid in resource_rings:
        members = set(gt.loc[gt["ring_id"] == rid, "account_id"])
        all_rings.append({"ring_id": rid, "ring_type": "resource_sharing", "members": members})
    for rref_id in referral_rings:
        members = set(gt.loc[gt["referral_ring_id"] == rref_id, "account_id"])
        all_rings.append({"ring_id": rref_id, "ring_type": "referral_chain", "members": members})

    ring_records = {}
    for r in all_rings:
        rid = r["ring_id"]
        mem = r["members"]
        c_dates = [acc_dates[a] for a in mem if a in acc_dates]
        form_date = min(c_dates) if c_dates else None
        comp_date = max(c_dates) if c_dates else None
        tot_orders = orders[orders["account_id"].isin(mem)]
        tot_vol = float(tot_orders["amount"].sum())

        ring_records[rid] = {
            "ring_id": rid,
            "ring_type": r["ring_type"],
            "member_count": len(mem),
            "formation_date": form_date.strftime("%Y-%m-%d") if form_date else None,
            "completion_date": comp_date.strftime("%Y-%m-%d") if comp_date else None,
            "first_clustered_date": None,
            "first_flagged_date": None,
            "dominant_cluster_id": None,
            "risk_score_at_flag": None,
            "detection_latency_days": None,
            "total_fraud_volume": round(tot_vol, 2),
            "pre_flag_fraud_volume": 0.0,
            "post_flag_fraud_volume": 0.0,
            "volume_prevented_pct": 0.0,
            "flagged": False,
        }

    t0 = time.time()
    for snap_date in snapshot_dates:
        snap_res = reconstruct_snapshot(
            snap_date, accounts, orders, dev, pay, addr, ip, referrals, model
        )
        if snap_res is None:
            continue

        c_df = snap_res["clusters"]
        risk_score_lookup = snap_res["risk_scores"]

        for r in all_rings:
            rid = r["ring_id"]
            rec = ring_records[rid]
            if rec["flagged"]:
                continue

            mem = r["members"]
            mem_as_of_t = {a for a in mem if acc_dates.get(a, snap_date + pd.Timedelta(days=1)) <= snap_date}
            if len(mem_as_of_t) < 2:
                continue

            dom = find_dominant_cluster(c_df, mem_as_of_t, min_capture=0.5)
            if dom is not None:
                cid = dom["cluster_id"]
                if rec["first_clustered_date"] is None:
                    rec["first_clustered_date"] = snap_date.strftime("%Y-%m-%d")

                score = risk_score_lookup.get(cid, 0.0)
                if score >= threshold:
                    rec["flagged"] = True
                    rec["first_flagged_date"] = snap_date.strftime("%Y-%m-%d")
                    rec["dominant_cluster_id"] = int(cid)
                    rec["risk_score_at_flag"] = round(float(score), 4)

                    form_dt = pd.to_datetime(rec["formation_date"])
                    latency = (snap_date - form_dt).days
                    rec["detection_latency_days"] = max(int(latency), 0)

                    r_orders = orders[orders["account_id"].isin(mem)]
                    pre_orders = r_orders[r_orders["timestamp"] <= snap_date]
                    post_orders = r_orders[r_orders["timestamp"] > snap_date]
                    pre_vol = float(pre_orders["amount"].sum())
                    post_vol = float(post_orders["amount"].sum())
                    rec["pre_flag_fraud_volume"] = round(pre_vol, 2)
                    rec["post_flag_fraud_volume"] = round(post_vol, 2)
                    tot = pre_vol + post_vol
                    rec["volume_prevented_pct"] = round(100.0 * post_vol / tot, 1) if tot > 0 else 0.0

    res_df = pd.DataFrame(list(ring_records.values()))
    flagged_df = res_df[res_df["flagged"] == True]

    if flagged_df.empty:
        summary = {
            "snapshots_evaluated": len(snapshot_dates),
            "snapshot_frequency_days": snapshot_freq_days,
            "backtest_duration_seconds": round(time.time() - t0, 1),
            "total_rings": len(res_df),
            "rings_flagged": 0,
            "detection_rate_pct": 0.0,
            "median_detection_latency_days": None,
            "mean_detection_latency_days": None,
            "min_detection_latency_days": None,
            "max_detection_latency_days": None,
            "average_volume_prevented_pct": 0.0,
            "total_prevented_fraud_amount": 0.0,
        }
    else:
        summary = {
            "snapshots_evaluated": len(snapshot_dates),
            "snapshot_frequency_days": snapshot_freq_days,
            "backtest_duration_seconds": round(time.time() - t0, 1),
            "total_rings": len(res_df),
            "rings_flagged": int(res_df["flagged"].sum()),
            "detection_rate_pct": round(100.0 * res_df["flagged"].mean(), 1),
            "median_detection_latency_days": round(float(flagged_df["detection_latency_days"].median()), 1),
            "mean_detection_latency_days": round(float(flagged_df["detection_latency_days"].mean()), 1),
            "min_detection_latency_days": int(flagged_df["detection_latency_days"].min()),
            "max_detection_latency_days": int(flagged_df["detection_latency_days"].max()),
            "average_volume_prevented_pct": round(float(flagged_df["volume_prevented_pct"].mean()), 1),
            "total_prevented_fraud_amount": round(float(flagged_df["post_flag_fraud_volume"].sum()), 2),
        }

    if save_csv:
        res_df.to_csv("temporal_detection_latencies.csv", index=False)
        Path("temporal_backtest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return res_df, summary


if __name__ == "__main__":
    print("=" * 60)
    print("UNIT TESTS (synthetic, no file I/O)")
    print("=" * 60)

    toy = pd.DataFrame({
        "cluster_id": [0, 0, 0, 0, 1, 1, 1, 2, 2],
        "account_id": ["A1", "A2", "A3", "X1", "A4", "A5", "X2", "X3", "X4"],
    })

    assert find_dominant_cluster(toy, set()) is None
    assert find_dominant_cluster(toy, {"NOTPRESENT1", "NOTPRESENT2"}) is None
    res = find_dominant_cluster(toy, {"A1", "A2", "A3", "A4"})
    assert res is not None and res["cluster_id"] == 0
    print("[OK] Synthetic unit tests passed.")

    print("\n" + "=" * 60)
    print("RUNNING AS-OF-T TEMPORAL RECONSTRUCTION BACKTEST")
    print("=" * 60)
    res_df, summary = run_temporal_backtest(snapshot_freq_days=14)
    print(f"Evaluated {summary['snapshots_evaluated']} snapshots in {summary['backtest_duration_seconds']}s")
    print(f"Detection Rate: {summary['rings_flagged']}/{summary['total_rings']} ({summary['detection_rate_pct']}%)")
    print(f"Median Detection Latency: {summary['median_detection_latency_days']} days")
    print(f"Average Fraud Volume Prevented: {summary['average_volume_prevented_pct']}%")
    print("\nSaved temporal_detection_latencies.csv and temporal_backtest_summary.json.")

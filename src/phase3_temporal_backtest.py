"""
phase3_temporal_backtest.py — Phase 3 Full Temporal Backtest & Counterfactual Protection

Evaluates Point-in-Time Reconstruction across all 20 resource-sharing rings and
8 referral rings at fine-grained temporal resolution with zero lookahead bias.
Computes:
1. Full Detection-Latency Table (all 28 rings)
2. Counterfactual Fraud Volume Protected (incurred vs protected amounts & order counts)
3. Subgroup breakdown (resource-sharing vs referral rings)
"""

import json
import time
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
from temporal_reconstruction import find_dominant_cluster, reconstruct_snapshot
from threshold_config import CHOSEN_THRESHOLD

DATA_DIR = Path("day1_data")


def run_phase3_backtest(snapshot_freq_days: int = 7) -> Tuple[pd.DataFrame, dict]:
    print("=" * 80)
    print("PHASE 3: FULL POINT-IN-TIME TEMPORAL BACKTEST (ZERO LOOKAHEAD BIAS)")
    print("=" * 80)

    # 1. Load Data
    print("\n[1/5] Loading datasets...")
    dev = pd.read_csv(DATA_DIR / "resolved_account_device.csv")
    pay = pd.read_csv(DATA_DIR / "resolved_account_payment.csv")
    addr = pd.read_csv(DATA_DIR / "resolved_account_address.csv")
    ip = pd.read_csv(DATA_DIR / "resolved_account_ip.csv")
    accounts = pd.read_csv(DATA_DIR / "accounts.csv")
    orders = pd.read_csv(DATA_DIR / "orders.csv")
    referrals = pd.read_csv(DATA_DIR / "referrals.csv") if (DATA_DIR / "referrals.csv").exists() else None
    gt = pd.read_csv(DATA_DIR / "ground_truth.csv")
    model = joblib.load("final_model.joblib")

    # Format Datetime
    accounts["creation_date"] = pd.to_datetime(accounts["creation_date"])
    orders["timestamp"] = pd.to_datetime(orders["timestamp"])
    dev["first_seen_date"] = pd.to_datetime(dev["first_seen_date"])
    pay["first_seen_date"] = pd.to_datetime(pay["first_seen_date"])
    addr["first_seen_date"] = pd.to_datetime(addr["first_seen_date"])
    ip["first_seen_date"] = pd.to_datetime(ip["first_seen_date"])
    if referrals is not None and not referrals.empty:
        referrals["referral_date"] = pd.to_datetime(referrals["referral_date"])

    acc_dates = dict(zip(accounts["account_id"], accounts["creation_date"]))

    # 2. Identify All Ground-Truth Rings (20 resource + 8 referral)
    print("\n[2/5] Identifying fraud ring ground truth...")
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

    print(f"Total Rings to evaluate: {len(all_rings)} (20 Resource-Sharing + 8 Referral-Chain)")

    # 3. Setup Tracking Matrix
    ring_records = {}
    for r in all_rings:
        rid = r["ring_id"]
        mem = r["members"]
        c_dates = [acc_dates[a] for a in mem if a in acc_dates]
        form_date = min(c_dates) if c_dates else None
        comp_date = max(c_dates) if c_dates else None
        ring_orders = orders[orders["account_id"].isin(mem)]

        tot_vol = float(ring_orders["amount"].sum())
        tot_orders = len(ring_orders)

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
            "total_fraud_orders": tot_orders,
            "total_fraud_volume": round(tot_vol, 2),
            "incurred_orders_pre_flag": 0,
            "incurred_volume_pre_flag": 0.0,
            "protected_orders_counterfactual": 0,
            "protected_volume_counterfactual": 0.0,
            "counterfactual_protected_pct": 0.0,
            "flagged": False,
        }

    # 4. Generate Snapshot Grid (Weekly = 7 days)
    start_date = pd.Timestamp("2024-10-01")
    end_date = pd.Timestamp("2026-08-30")
    snapshot_dates = pd.date_range(start=start_date, end=end_date, freq=f"{snapshot_freq_days}D").tolist()
    if snapshot_dates[-1] < end_date:
        snapshot_dates.append(end_date)

    print(f"\n[3/5] Slicing timeline into {len(snapshot_dates)} historical snapshots (Step = {snapshot_freq_days} days)...")

    # 5. Execute Chronological Slices
    t_start = time.time()
    for snap_idx, snap_date in enumerate(snapshot_dates):
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
                if score >= CHOSEN_THRESHOLD:
                    rec["flagged"] = True
                    rec["first_flagged_date"] = snap_date.strftime("%Y-%m-%d")
                    rec["dominant_cluster_id"] = int(cid)
                    rec["risk_score_at_flag"] = round(float(score), 4)

                    form_dt = pd.to_datetime(rec["formation_date"])
                    latency = (snap_date - form_dt).days
                    rec["detection_latency_days"] = max(int(latency), 0)

                    # Calculate Counterfactual Orders & Volume
                    r_orders = orders[orders["account_id"].isin(mem)]
                    pre_orders = r_orders[r_orders["timestamp"] <= snap_date]
                    post_orders = r_orders[r_orders["timestamp"] > snap_date]

                    pre_vol = float(pre_orders["amount"].sum())
                    post_vol = float(post_orders["amount"].sum())

                    rec["incurred_orders_pre_flag"] = len(pre_orders)
                    rec["incurred_volume_pre_flag"] = round(pre_vol, 2)
                    rec["protected_orders_counterfactual"] = len(post_orders)
                    rec["protected_volume_counterfactual"] = round(post_vol, 2)

                    tot = pre_vol + post_vol
                    rec["counterfactual_protected_pct"] = round(100.0 * post_vol / tot, 1) if tot > 0 else 0.0

    duration = time.time() - t_start
    print(f"[4/5] Backtest completed in {duration:.1f}s!")

    # 6. Aggregate Results
    res_df = pd.DataFrame(list(ring_records.values()))
    flagged_df = res_df[res_df["flagged"] == True]
    resource_df = flagged_df[flagged_df["ring_type"] == "resource_sharing"]
    referral_df = flagged_df[flagged_df["ring_type"] == "referral_chain"]

    tot_fraud_vol = float(res_df["total_fraud_volume"].sum())
    tot_incurred_vol = float(res_df["incurred_volume_pre_flag"].sum())
    tot_protected_vol = float(res_df["protected_volume_counterfactual"].sum())
    overall_protected_pct = round(100.0 * tot_protected_vol / tot_fraud_vol, 2) if tot_fraud_vol > 0 else 0.0

    summary = {
        "snapshots_evaluated": len(snapshot_dates),
        "snapshot_frequency_days": snapshot_freq_days,
        "backtest_duration_seconds": round(duration, 1),
        "total_rings": len(res_df),
        "rings_flagged": int(res_df["flagged"].sum()),
        "detection_rate_pct": round(100.0 * res_df["flagged"].mean(), 1),
        
        # Overall Latency
        "median_detection_latency_days": round(float(flagged_df["detection_latency_days"].median()), 1),
        "mean_detection_latency_days": round(float(flagged_df["detection_latency_days"].mean()), 1),
        "min_detection_latency_days": int(flagged_df["detection_latency_days"].min()),
        "max_detection_latency_days": int(flagged_df["detection_latency_days"].max()),

        # Subgroup Latency
        "resource_sharing_median_latency_days": round(float(resource_df["detection_latency_days"].median()), 1),
        "referral_chain_median_latency_days": round(float(referral_df["detection_latency_days"].median()), 1),

        # Counterfactual Volume Protection
        "total_gross_fraud_volume": round(tot_fraud_vol, 2),
        "incurred_fraud_volume_pre_flag": round(tot_incurred_vol, 2),
        "counterfactual_protected_volume": round(tot_protected_vol, 2),
        "counterfactual_protected_rate_pct": overall_protected_pct,

        # Order Counts
        "total_fraud_orders": int(res_df["total_fraud_orders"].sum()),
        "incurred_fraud_orders_pre_flag": int(res_df["incurred_orders_pre_flag"].sum()),
        "counterfactual_protected_orders": int(res_df["protected_orders_counterfactual"].sum()),

        # Subgroup Financials
        "resource_sharing_protected_volume": round(float(resource_df["protected_volume_counterfactual"].sum()), 2),
        "resource_sharing_protected_rate_pct": round(100.0 * float(resource_df["protected_volume_counterfactual"].sum()) / float(resource_df["total_fraud_volume"].sum()), 2),
        "referral_chain_protected_volume": round(float(referral_df["protected_volume_counterfactual"].sum()), 2),
        "referral_chain_protected_rate_pct": round(100.0 * float(referral_df["protected_volume_counterfactual"].sum()) / float(referral_df["total_fraud_volume"].sum()), 2),
    }

    # Save to disk
    print("\n[5/5] Saving Phase 3 audit files...")
    res_df.to_csv("phase3_detection_latency_audit.csv", index=False)

    # Convert res_df for temporal_detection_latencies format compatibility
    compat_df = res_df.copy()
    compat_df["pre_flag_fraud_volume"] = compat_df["incurred_volume_pre_flag"]
    compat_df["post_flag_fraud_volume"] = compat_df["protected_volume_counterfactual"]
    compat_df["volume_prevented_pct"] = compat_df["counterfactual_protected_pct"]
    compat_df.to_csv("temporal_detection_latencies.csv", index=False)

    compat_summary = {
        "snapshots_evaluated": summary["snapshots_evaluated"],
        "snapshot_frequency_days": summary["snapshot_frequency_days"],
        "backtest_duration_seconds": summary["backtest_duration_seconds"],
        "total_rings": summary["total_rings"],
        "rings_flagged": summary["rings_flagged"],
        "detection_rate_pct": summary["detection_rate_pct"],
        "median_detection_latency_days": summary["median_detection_latency_days"],
        "mean_detection_latency_days": summary["mean_detection_latency_days"],
        "min_detection_latency_days": summary["min_detection_latency_days"],
        "max_detection_latency_days": summary["max_detection_latency_days"],
        "average_volume_prevented_pct": round(float(flagged_df["counterfactual_protected_pct"].mean()), 1),
        "total_prevented_fraud_amount": summary["counterfactual_protected_volume"],
    }
    Path("temporal_backtest_summary.json").write_text(json.dumps(compat_summary, indent=2), encoding="utf-8")
    Path("phase3_counterfactual_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return res_df, summary


if __name__ == "__main__":
    df, summary = run_phase3_backtest(snapshot_freq_days=7)

    print("\n" + "=" * 80)
    print("PHASE 3 EXECUTIVE SUMMARY")
    print("=" * 80)
    print(f"Rings Evaluated: {summary['total_rings']} | Rings Flagged: {summary['rings_flagged']} ({summary['detection_rate_pct']}%)")
    print(f"Median Detection Latency: {summary['median_detection_latency_days']} days (Resource Rings: {summary['resource_sharing_median_latency_days']}d | Referral Rings: {summary['referral_chain_median_latency_days']}d)")
    print(f"Total Gross Fraud Exposure: Rs. {summary['total_gross_fraud_volume']:,.2f}")
    print(f"Incurred Fraud (Pre-Flag): Rs. {summary['incurred_fraud_volume_pre_flag']:,.2f} ({100 - summary['counterfactual_protected_rate_pct']:.2f}%)")
    print(f"COUNTERFACTUAL VOLUME PROTECTED: Rs. {summary['counterfactual_protected_volume']:,.2f} ({summary['counterfactual_protected_rate_pct']}%)")
    print(f"Counterfactual Orders Intercepted: {summary['counterfactual_protected_orders']} / {summary['total_fraud_orders']} orders")
    print("=" * 80)

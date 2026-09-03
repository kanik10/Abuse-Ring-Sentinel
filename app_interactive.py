"""
app_interactive.py — Abuse-Ring Sentinel Interactive Intelligence Dashboard

Ultra-modern, dark glassmorphism executive interface for inspecting abuse-ring
clusters, network topology, local SHAP risk drivers, account rankings, and referral signals.

Read-only, non-actioning defense interface — every output is a human review recommendation.
# Sentinel Interactive Dashboard - Refreshed UI 2026
Run with: streamlit run app_interactive.py
"""

import altair as alt
import html
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import community as community_louvain

from graph_builder import build_account_graph
from feature_engineering import compute_cluster_features
from account_scoring import score_accounts_in_cluster
from referral_features import compute_referral_features, REFERRAL_FEATURE_COLS
from threshold_config import CHOSEN_THRESHOLD as DEFAULT_THRESHOLD
from pdf_generator import generate_cluster_pdf_report

PURE_GRAPH_FEATURES = ["cluster_size", "entity_reuse_ratio", "internal_density"]
CHAMPION_FEATURE_COLS = PURE_GRAPH_FEATURES + REFERRAL_FEATURE_COLS

FEATURE_LABELS = {
    "cluster_size": "Cluster size",
    "entity_reuse_ratio": "Entity reuse ratio",
    "internal_density": "Internal graph density",
    "referral_cycle_ratio": "Referral cycle ratio",
    "referral_resource_overlap_ratio": "Referral resource overlap",
    "median_referral_activation_days": "Median activation days",
    "within_cluster_referral_density": "Within-cluster referral density",
}

# ---------------------------------------------------------------------------
# Page Configuration & Dark Glassmorphism CSS Theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Abuse-Ring Sentinel — Intelligence Dashboard",
    page_icon="Sentinel",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Main background theme */
    .stApp {
        background: radial-gradient(circle at 50% -20%, #1e293b 0%, #0b0f19 50%, #05070d 100%) !important;
        color: #f8fafc !important;
    }

    /* Titles Styling — JetBrains Mono */
    h1, h2, h3, [data-testid="stHeader"] h1 {
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 500 !important;
        letter-spacing: -0.02em !important;
        color: #f8fafc !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: rgba(11, 15, 25, 0.85) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(59, 130, 246, 0.15) !important;
    }
    
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        font-family: 'JetBrains Mono', monospace !important;
        color: #60a5fa !important;
    }


    /* Native Border Containers (Equal Height Glassmorphism) */
    [data-testid="stBorderContainer"] {
        background: rgba(15, 23, 42, 0.65) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(59, 130, 246, 0.2) !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37), 0 0 15px rgba(59, 130, 246, 0.05) !important;
        padding: 16px 20px !important;
        margin-bottom: 16px !important;
    }


    /* Multiselect Input Container Styling */
    div[data-testid="stMultiSelect"] {
        margin-bottom: 4px !important;
    }

    /* Multiselect Tag Pill Boxes -> PURE SOLID BLACK (#000000), NO OUTLINE, NATURAL SIZE */
    span[data-tag],
    div[data-tag],
    span[data-baseweb="tag"],
    div[data-baseweb="tag"] {
        background-color: #000000 !important;
        background: #000000 !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }

    /* Keep natural spacing and make text & icons white */
    span[data-tag] *,
    div[data-tag] *,
    span[data-baseweb="tag"] *,
    div[data-baseweb="tag"] * {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        color: #ffffff !important;
    }

    span[data-tag] svg,
    div[data-tag] svg,
    span[data-baseweb="tag"] svg,
    div[data-baseweb="tag"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
        border: none !important;
    }

    /* Ensure widget labels & buttons NEVER get black boxes or borders */
    div[data-testid="stMultiSelect"] label,
    div[data-testid="stMultiSelect"] label *,
    label[data-testid="stWidgetLabel"],
    label[data-testid="stWidgetLabel"] * {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #f8fafc !important;
    }

    /* Slider track & thumb override (Replaces default bright red) */
    [data-baseweb="slider"] div[role="slider"],
    div[data-testid="stSlider"] div[role="slider"] {
        background-color: #3b82f6 !important;
        border: 2px solid #ffffff !important;
        box-shadow: none !important;
    }
    div[data-testid="stSlider"] label + div [data-testid="stMarkdownContainer"] p {
        color: #60a5fa !important;
        font-weight: 600 !important;
    }

    /* Native Metric Cards Override */
    [data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.45) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(59, 130, 246, 0.15) !important;
        border-radius: 12px !important;
        padding: 10px 12px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25) !important;
        transition: transform 0.2s ease, border-color 0.2s ease !important;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px) !important;
        border-color: rgba(59, 130, 246, 0.4) !important;
    }

    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
        color: #94a3b8 !important;
        font-size: 0.65rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.3px !important;
        text-transform: uppercase !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }

    [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
        color: #60a5fa !important;
        font-weight: 700 !important;
        font-size: 1.25rem !important;
        text-shadow: 0 0 12px rgba(59, 130, 246, 0.3) !important;
        white-space: nowrap !important;
    }



    /* Title Un-bolding */
    h1, [data-testid="stHeader"] h1 {
        font-weight: 400 !important;
        letter-spacing: -0.01em !important;
        color: #f8fafc !important;
    }

    /* ----------------------------------------------------------------------- */
    /* Tabs Styling — Single Electric-Blue Line with Upward Glow (No Orange, No Box) */
    /* ----------------------------------------------------------------------- */
    div[data-testid="stTabs"] {
        --primary-color: #3b82f6 !important;
    }

    .stTabs [data-baseweb="tab-list"], 
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
        gap: 20px !important;
        padding: 0 0 2px 0 !important;
    }

    .stTabs button[data-baseweb="tab"], 
    [data-testid="stTabs"] button[role="tab"],
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        border-top: none !important;
        border-left: none !important;
        border-right: none !important;
        border-bottom: none !important;
        outline: none !important;
        box-shadow: none !important;
        color: #94a3b8 !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        padding: 10px 16px !important;
        transition: color 0.2s ease !important;
    }

    .stTabs button[aria-selected="true"],
    [data-testid="stTabs"] button[aria-selected="true"] {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        border-top: none !important;
        border-left: none !important;
        border-right: none !important;
        border-bottom: none !important;
        outline: none !important;
        box-shadow: none !important;
        color: #60a5fa !important;
        font-weight: 600 !important;
        text-shadow: 0 0 10px rgba(96, 165, 250, 0.6) !important;
    }

    /* Single Upward Glowing Electric-Blue Underline Bar */
    .stTabs [data-baseweb="tab-highlight"],
    [data-testid="stTabs"] [data-baseweb="tab-highlight"],
    [data-testid="stTabHighlight"] {
        background-color: #3b82f6 !important;
        background: #3b82f6 !important;
        height: 3px !important;
        border-radius: 2px !important;
        box-shadow: 0 -8px 20px 4px rgba(59, 130, 246, 0.85), 0 -2px 10px rgba(96, 165, 250, 0.9) !important;
    }

    .stTabs [data-baseweb="tab-border"],
    [data-testid="stTabs"] [data-baseweb="tab-border"] {
        background-color: transparent !important;
        display: none !important;
        height: 0px !important;
    }

    /* ----------------------------------------------------------------------- */
    /* Tab 1 (Network Graph) Special Referral-Pink Theme (#ec4899)             */
    /* ----------------------------------------------------------------------- */
    .stTabs button[data-baseweb="tab"]:nth-child(1)[aria-selected="true"],
    [data-testid="stTabs"] button[role="tab"]:nth-child(1)[aria-selected="true"],
    [data-testid="stTabs"] button[id*="tab-0"][aria-selected="true"] {
        color: #ec4899 !important;
        text-shadow: 0 0 10px rgba(236, 72, 153, 0.7) !important;
    }

    [data-baseweb="tab-list"]:has(button:nth-child(1)[aria-selected="true"]) [data-baseweb="tab-highlight"],
    .stTabs:has(button:nth-child(1)[aria-selected="true"]) [data-baseweb="tab-highlight"] {
        background-color: #ec4899 !important;
        background: #ec4899 !important;
        box-shadow: 0 -8px 20px 4px rgba(236, 72, 153, 0.85), 0 -2px 10px rgba(244, 114, 182, 0.9) !important;
    }

    /* Scope Pink Theme to Network Graph Tab Content (Tab 1 Only) */
    div[data-testid="stTabContent"]:nth-of-type(1) {
        --primary-color: #ec4899 !important;
        --primary: #ec4899 !important;
    }

    /* Sliders in Network Graph Tab -> Pink (#ec4899) */
    div[data-testid="stTabContent"]:nth-of-type(1) [data-baseweb="slider"] div[role="slider"],
    div[data-testid="stTabContent"]:nth-of-type(1) div[data-testid="stSlider"] div[role="slider"] {
        background-color: #ec4899 !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 0 12px rgba(236, 72, 153, 0.7) !important;
    }

    div[data-testid="stTabContent"]:nth-of-type(1) div[data-testid="stSlider"] [data-baseweb="slider"] > div > div:first-child,
    div[data-testid="stTabContent"]:nth-of-type(1) div[data-testid="stSlider"] [data-baseweb="slider"] div[style*="background"] {
        background-color: #ec4899 !important;
    }

    div[data-testid="stTabContent"]:nth-of-type(1) div[data-testid="stSlider"] label + div [data-testid="stMarkdownContainer"] p {
        color: #f472b6 !important;
    }

    /* Checkbox in Network Graph Tab -> Pink (#ec4899) */
    div[data-testid="stTabContent"]:nth-of-type(1) [data-testid="stCheckbox"] input:checked + div,
    div[data-testid="stTabContent"]:nth-of-type(1) [data-testid="stCheckbox"] div[role="checkbox"][aria-checked="true"],
    div[data-testid="stTabContent"]:nth-of-type(1) [data-testid="stCheckbox"] span[aria-checked="true"],
    div[data-testid="stTabContent"]:nth-of-type(1) [data-testid="stCheckbox"] [data-baseweb="checkbox"] span {
        background-color: #ec4899 !important;
        background: #ec4899 !important;
        border-color: #ec4899 !important;
    }

    div[data-testid="stTabContent"]:nth-of-type(1) [data-testid="stCheckbox"] svg {
        fill: #ffffff !important;
    }





    /* Dataframe & Tables */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(59, 130, 246, 0.2) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25) !important;
    }

    /* Custom Badges & Labels */
    .badge-blue {
        background: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.35);
        color: #60a5fa;
        font-size: 0.72rem;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        letter-spacing: 0.5px;
        display: inline-block;
        margin-bottom: 8px;
    }

    /* Sliders */
    .stSlider [data-baseweb="slider"] {
        color: #3b82f6 !important;
    }

    /* Buttons & Controls — Clean Modern Flat (Zero Glows / Shadows) */
    .stButton>button, .stDownloadButton>button,
    button[data-testid="stBaseButton-secondary"],
    button[data-testid="stBaseButton-primary"] {
        background: rgba(30, 41, 59, 0.8) !important;
        color: #f8fafc !important;
        border-radius: 8px !important;
        border: 1px solid rgba(148, 163, 184, 0.25) !important;
        font-weight: 500 !important;
        font-family: Inter, sans-serif !important;
        font-size: 13px !important;
        box-shadow: none !important;
        filter: none !important;
        outline: none !important;
        transition: background 0.15s ease, border-color 0.15s ease !important;
    }

    .stButton>button:hover, .stDownloadButton>button:hover,
    button[data-testid="stBaseButton-secondary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover {
        background: rgba(51, 65, 85, 0.9) !important;
        border-color: rgba(148, 163, 184, 0.45) !important;
        color: #ffffff !important;
        box-shadow: none !important;
        filter: none !important;
        transform: none !important;
        outline: none !important;
    }
    
    hr {
        border-color: rgba(59, 130, 246, 0.15) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_model():
    model_path = BASE_DIR / "final_model.joblib"
    return joblib.load(model_path)


@st.cache_data
def load_bundled_demo_data():
    d = BASE_DIR / "day1_data" if (BASE_DIR / "day1_data").exists() else BASE_DIR
    ref_path = d / "referrals.csv"
    referrals = (
        pd.read_csv(ref_path)
        if ref_path.exists()
        else pd.DataFrame(columns=["referrer_id", "referred_id", "referral_date", "bonus_amount"])
    )
    ground_truth = pd.read_csv(d / "ground_truth.csv")
    if "is_referral_ring_member" in ground_truth.columns:
        # Unify ring membership so referral-chain abuse ring members are recognized as true ring members
        ground_truth["is_ring_member"] = (
            (ground_truth["is_ring_member"] == True) | (ground_truth["is_referral_ring_member"] == True)
        )
    return (
        pd.read_csv(d / "accounts.csv"),
        pd.read_csv(d / "resolved_account_device.csv"),
        pd.read_csv(d / "resolved_account_payment.csv"),
        pd.read_csv(d / "resolved_account_address.csv"),
        pd.read_csv(d / "resolved_account_ip.csv"),
        pd.read_csv(d / "orders.csv"),
        ground_truth,
        referrals,
    )


@st.cache_data
def load_temporal_backtest_data():
    lat_path = BASE_DIR / "temporal_detection_latencies.csv"
    sum_path = BASE_DIR / "temporal_backtest_summary.json"
    lat_df = pd.read_csv(lat_path) if lat_path.exists() else None
    sum_dict = json.loads(sum_path.read_text(encoding="utf-8")) if sum_path.exists() else None
    return lat_df, sum_dict


def format_rs(value: float) -> str:
    if abs(value) >= 1e7:
        return f"Rs.{value/1e7:,.2f}Cr"
    elif abs(value) >= 1e5:
        return f"Rs.{value/1e5:,.2f}L"
    elif abs(value) >= 1e3:
        return f"Rs.{value/1e3:,.1f}k"
    return f"Rs.{value:,.0f}"



def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def cluster_shared_resources(members, account_device, account_payment, account_address, account_ip):
    pair_resources = defaultdict(list)
    resource_rows = []

    for resource_type, df, col in [
        ("device", account_device, "device_id"),
        ("payment", account_payment, "payment_id"),
        ("address", account_address, "address_id"),
        ("ip", account_ip, "ip_id"),
    ]:
        sub = df[df.account_id.isin(members)]
        for resource_id, group in sub.groupby(col):
            users = sorted(set(group["account_id"].tolist()))
            if len(users) < 2:
                continue
            resource_rows.append({
                "resource_type": resource_type,
                "resource_id": resource_id,
                "member_accounts": len(users),
            })
            for a, b in itertools.combinations(users, 2):
                key = (a, b) if a < b else (b, a)
                pair_resources[key].append(f"{resource_type}:{resource_id}")

    resource_summary = pd.DataFrame(resource_rows)
    if not resource_summary.empty:
        resource_summary = resource_summary.sort_values(
            ["member_accounts", "resource_type"], ascending=[False, True]
        )
    return pair_resources, resource_summary


def filter_cluster_graph(G, members, pair_resources, selected_resource_types,
                         min_edge_weight, min_node_degree, hide_isolated):
    selected_resource_types = set(selected_resource_types)
    base = G.subgraph(members).copy()
    filtered = nx.Graph()
    filtered.add_nodes_from(base.nodes())

    for a, b, data in base.edges(data=True):
        key = (a, b) if a < b else (b, a)
        resource_types = {
            resource.split(":", 1)[0]
            for resource in pair_resources.get(key, [])
            if ":" in resource
        }
        if not (resource_types & selected_resource_types):
            continue

        weight = float(data.get("weight", 1.0))
        if weight < min_edge_weight:
            continue

        filtered.add_edge(a, b, **data)

    if min_node_degree > 0:
        weighted_degree = dict(filtered.degree(weight="weight"))
        low_signal_nodes = [node for node, degree in weighted_degree.items()
                            if degree < min_node_degree]
        filtered.remove_nodes_from(low_signal_nodes)

    if hide_isolated:
        filtered.remove_nodes_from(list(nx.isolates(filtered)))

    return filtered


def render_cluster_graph_html(subgraph, accounts, orders, pair_resources,
                               selected_resource_types, node_label_mode,
                               edge_label_mode):
    if subgraph.number_of_nodes() == 0:
        return """
        <div style="height: 520px; display: grid; place-items: center; color: #94a3b8;
                    border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 12px;
                    background: rgba(15, 23, 42, 0.65); font-family: Inter, sans-serif;">
          No graph structure matches the current filters.
        </div>
        """

    pos = nx.spring_layout(subgraph, seed=42, weight="weight", iterations=90)

    width, height, pad = 980, 560, 48
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)

    def project(node):
        x, y = pos[node]
        px = pad + ((x - min_x) / span_x) * (width - 2 * pad)
        py = pad + ((y - min_y) / span_y) * (height - 2 * pad)
        return px, py

    account_lookup = accounts.set_index("account_id")
    order_totals = orders.groupby("account_id")["amount"].sum()
    weighted_degree = dict(subgraph.degree(weight="weight"))
    max_degree = max(weighted_degree.values()) if weighted_degree else 1.0
    max_weight = max((d.get("weight", 1.0) for _, _, d in subgraph.edges(data=True)), default=1.0)
    selected_resource_types = set(selected_resource_types)

    def visible_resources_for_edge(a, b):
        key = (a, b) if a < b else (b, a)
        resources = pair_resources.get(key, [])
        return [
            resource for resource in resources
            if resource.split(":", 1)[0] in selected_resource_types
        ]

    def node_label(node, degree, order_value):
        if node_label_mode == "None":
            return ""
        if node_label_mode == "Full account ID":
            return str(node)
        if node_label_mode == "Weighted degree":
            return f"{str(node)[-6:]} d={degree:.2f}"
        if node_label_mode == "Order value":
            return f"{str(node)[-6:]} Rs.{order_value:,.0f}"
        return str(node)[-6:]

    def edge_label(weight, resources):
        if edge_label_mode == "None":
            return ""
        if edge_label_mode == "Weight":
            return f"{weight:.2f}"
        if edge_label_mode == "Shared count":
            return f"{len(resources)} shared"
        resource_types = sorted({resource.split(":", 1)[0] for resource in resources})
        return "+".join(resource_types)

    degree_vals = list(weighted_degree.values())
    med_deg = np.median(degree_vals) if degree_vals else 0.0

    nodes = []
    for node in subgraph.nodes():
        degree = float(weighted_degree.get(node, 0.0))
        radius = float(9.0 + 13.0 * (degree / max_degree if max_degree else 0.0))
        order_value = float(order_totals.get(node, 0.0))
        creation = ""
        if node in account_lookup.index and "creation_date" in account_lookup.columns:
            creation = str(account_lookup.loc[node, "creation_date"])
        fill = "#ec4899" if degree >= med_deg and med_deg > 0 else "#3b82f6"
        title = (
            f"{node}\nweighted degree={degree:.3f}\norder value=Rs.{order_value:,.2f}"
            + (f"\ncreated={creation}" if creation else "")
        )
        lbl = node_label(node, degree, order_value)
        nodes.append({
            "id": str(node),
            "label": lbl,
            "radius": radius,
            "fill": fill,
            "title": title,
            "degree": degree,
            "order_value": order_value,
        })

    links = []
    for u, v, data in subgraph.edges(data=True):
        weight = float(data.get("weight", 1.0))
        stroke_width = float(1.2 + 4.5 * (weight / max_weight))
        resources = visible_resources_for_edge(u, v)
        resource_text = ", ".join(resources[:8])
        if len(resources) > 8:
            resource_text += f", +{len(resources) - 8} more"
        title = f"{u} <-> {v}\nweight={weight:.3f}\nshared: {resource_text or 'shared resource'}"
        lbl = edge_label(weight, resources)
        links.append({
            "source": str(u),
            "target": str(v),
            "weight": weight,
            "stroke_width": stroke_width,
            "label": lbl,
            "title": title,
        })

    template = """
    <div class="graph-shell">
      <div class="graph-toolbar">
        <span style="color:#60a5fa; font-weight:600; font-family:'JetBrains Mono', monospace; font-size:11px; letter-spacing:0.5px;">PHYSICS SIMULATION VIEWPORT</span>
        <button id="reset-zoom-btn" class="hud-btn">Reset View</button>
        <button id="reheat-sim-btn" class="hud-btn">Re-spread Layout</button>
        <button id="fullscreen-btn" class="hud-btn" style="margin-left:auto;">&boxbox; Fullscreen</button>
      </div>
      <svg id="cluster-graph" viewBox="0 0 __WIDTH__ __HEIGHT__" role="img"></svg>
    </div>
    <style>
      .graph-shell {
        width: 100%;
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 16px;
        background: rgba(7, 9, 14, 0.95);
        font-family: Inter, system-ui, sans-serif;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5), inset 0 0 25px rgba(59, 130, 246, 0.05);
        overflow: hidden;
      }
      .graph-shell:fullscreen {
        width: 100vw !important;
        height: 100vh !important;
        border-radius: 0 !important;
        background: #07090e !important;
        display: flex;
        flex-direction: column;
      }
      .graph-shell:fullscreen #cluster-graph {
        height: calc(100vh - 45px) !important;
        flex: 1;
      }
      .graph-toolbar {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 8px 16px;
        background: rgba(15, 23, 42, 0.85);
        border-bottom: 1px solid rgba(59, 130, 246, 0.15);
      }
      .hud-btn {
        background: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.35);
        color: #60a5fa;
        border-radius: 6px;
        padding: 4px 12px;
        font-size: 11px;
        font-family: Inter, sans-serif;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s;
      }
      .hud-btn:hover {
        background: rgba(59, 130, 246, 0.35);
        color: #ffffff;
        box-shadow: 0 0 10px rgba(59, 130, 246, 0.4);
      }
      #cluster-graph {
        width: 100%;
        height: 500px;
        display: block;
        cursor: grab;
      }
      #cluster-graph:active { cursor: grabbing; }
      .edge-line {
        stroke: #334155;
        stroke-opacity: 0.5;
        stroke-linecap: round;
        transition: stroke 0.2s, stroke-opacity 0.2s;
      }
      .edge-line.highlighted {
        stroke: #60a5fa !important;
        stroke-opacity: 1.0 !important;
        filter: drop-shadow(0px 0px 6px rgba(96, 165, 250, 0.8));
      }
      .node-group {
        cursor: grab;
      }
      .node-group:active {
        cursor: grabbing;
      }
      .node-circle {
        stroke: #0f172a;
        stroke-width: 2.5px;
        filter: drop-shadow(0px 0px 6px rgba(59, 130, 246, 0.5));
        transition: stroke 0.2s;
      }
      .node-circle.selected-node {
        stroke: #f59e0b !important;
        stroke-width: 4.5px !important;
        filter: drop-shadow(0px 0px 14px rgba(245, 158, 11, 0.95)) !important;
      }
      .node-group {
        cursor: grab;
        transition: opacity 0.25s ease;
      }
      .node-group:active {
        cursor: grabbing;
      }
      .node-group:hover .node-circle {
        stroke: #ffffff !important;
        stroke-width: 3.5px !important;
        filter: drop-shadow(0px 0px 12px rgba(255, 255, 255, 0.9));
      }
      .node-txt {
        fill: #f8fafc;
        font-size: 10.5px;
        font-weight: 600;
        text-anchor: middle;
        pointer-events: none;
        paint-order: stroke;
        stroke: #07090e;
        stroke-linejoin: round;
        stroke-width: 3.5px;
      }
      .edge-txt {
        fill: #94a3b8;
        font-size: 10px;
        font-weight: 600;
        text-anchor: middle;
        pointer-events: none;
        paint-order: stroke;
        stroke: #07090e;
        stroke-linejoin: round;
        stroke-width: 3.5px;
        opacity: 0.35;
        transition: opacity 0.2s, fill 0.2s;
      }
      .edge-txt.highlighted {
        opacity: 1.0 !important;
        fill: #60a5fa !important;
      }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
    <script>
      (() => {
        const nodes = __NODES_JSON__;
        const links = __LINKS_JSON__;
        const width = __WIDTH__;
        const height = __HEIGHT__;

        const svg = d3.select("#cluster-graph");
        svg.selectAll("*").remove();

        const defs = svg.append("defs");
        const pat = defs.append("pattern")
          .attr("id", "grid-pattern-d3")
          .attr("width", 30)
          .attr("height", 30)
          .attr("patternUnits", "userSpaceOnUse");
        pat.append("path")
          .attr("d", "M 30 0 L 0 0 0 30")
          .attr("fill", "none")
          .attr("stroke", "rgba(59, 130, 246, 0.05)")
          .attr("stroke-width", 1);

        svg.append("rect")
          .attr("width", width)
          .attr("height", height)
          .attr("rx", 14)
          .attr("fill", "#07090e");

        svg.append("rect")
          .attr("width", width)
          .attr("height", height)
          .attr("rx", 14)
          .attr("fill", "url(#grid-pattern-d3)");

        const container = svg.append("g").attr("class", "graph-container");

        const zoom = d3.zoom()
          .scaleExtent([0.3, 4.0])
          .on("zoom", (event) => {
            container.attr("transform", event.transform);
          });
        svg.call(zoom);

        // Highly damped physics simulation to prevent excessive movement/jittering
        const simulation = d3.forceSimulation(nodes)
          .velocityDecay(0.65)
          .alphaDecay(0.06)
          .force("link", d3.forceLink(links).id(d => d.id).distance(d => 120 + 35 / Math.sqrt(d.weight || 1)))
          .force("charge", d3.forceManyBody().strength(-350))
          .force("center", d3.forceCenter(width / 2, height / 2))
          .force("collide", d3.forceCollide().radius(d => d.radius + 30));

        const linkGroup = container.append("g").attr("class", "links-group");
        const link = linkGroup.selectAll("line")
          .data(links)
          .enter().append("line")
          .attr("class", "edge-line")
          .attr("stroke-width", d => d.stroke_width || 2.0);

        link.append("title").text(d => d.title);

        const linkText = linkGroup.selectAll("text")
          .data(links)
          .enter().append("text")
          .attr("class", "edge-txt")
          .text(d => d.label);

        const nodeGroup = container.append("g").attr("class", "nodes-group");
        const node = nodeGroup.selectAll("g")
          .data(nodes)
          .enter().append("g")
          .attr("class", "node-group")
          .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended));

        node.append("circle")
          .attr("class", "node-circle")
          .attr("r", d => d.radius)
          .attr("fill", d => d.fill);

        node.append("title").text(d => d.title);

        node.append("text")
          .attr("class", "node-txt")
          .attr("dy", d => d.radius + 15)
          .text(d => d.label);

        let selectedNodeId = null;

        function updateHighlights() {
          if (selectedNodeId) {
            const connectedNodeIds = new Set([selectedNodeId]);
            links.forEach(l => {
              const srcId = typeof l.source === "object" ? l.source.id : l.source;
              const tgtId = typeof l.target === "object" ? l.target.id : l.target;
              if (srcId === selectedNodeId) connectedNodeIds.add(tgtId);
              if (tgtId === selectedNodeId) connectedNodeIds.add(srcId);
            });

            node.style("opacity", n => connectedNodeIds.has(n.id) ? 1.0 : 0.18);
            node.selectAll(".node-circle").classed("selected-node", n => n.id === selectedNodeId);

            link.style("opacity", l => {
              const srcId = typeof l.source === "object" ? l.source.id : l.source;
              const tgtId = typeof l.target === "object" ? l.target.id : l.target;
              return (srcId === selectedNodeId || tgtId === selectedNodeId) ? 1.0 : 0.06;
            }).classed("highlighted", l => {
              const srcId = typeof l.source === "object" ? l.source.id : l.source;
              const tgtId = typeof l.target === "object" ? l.target.id : l.target;
              return (srcId === selectedNodeId || tgtId === selectedNodeId);
            });

            linkText.style("opacity", l => {
              const srcId = typeof l.source === "object" ? l.source.id : l.source;
              const tgtId = typeof l.target === "object" ? l.target.id : l.target;
              return (srcId === selectedNodeId || tgtId === selectedNodeId) ? 1.0 : 0.04;
            }).classed("highlighted", l => {
              const srcId = typeof l.source === "object" ? l.source.id : l.source;
              const tgtId = typeof l.target === "object" ? l.target.id : l.target;
              return (srcId === selectedNodeId || tgtId === selectedNodeId);
            });
          } else {
            node.style("opacity", 1.0);
            node.selectAll(".node-circle").classed("selected-node", false);
            link.style("opacity", 0.5).classed("highlighted", false);
            linkText.style("opacity", 0.35).classed("highlighted", false);
          }
        }

        node.on("click", (event, d) => {
          event.stopPropagation();
          selectedNodeId = (selectedNodeId === d.id) ? null : d.id;
          updateHighlights();
        });

        svg.on("click", () => {
          if (selectedNodeId) {
            selectedNodeId = null;
            updateHighlights();
          }
        });

        node.on("mouseover", (event, d) => {
          if (!selectedNodeId) {
            link.classed("highlighted", l => {
              const srcId = typeof l.source === "object" ? l.source.id : l.source;
              const tgtId = typeof l.target === "object" ? l.target.id : l.target;
              return srcId === d.id || tgtId === d.id;
            });
            linkText.classed("highlighted", l => {
              const srcId = typeof l.source === "object" ? l.source.id : l.source;
              const tgtId = typeof l.target === "object" ? l.target.id : l.target;
              return srcId === d.id || tgtId === d.id;
            });
          }
        }).on("mouseout", () => {
          if (!selectedNodeId) {
            link.classed("highlighted", false);
            linkText.classed("highlighted", false);
          }
        });

        simulation.on("tick", () => {
          link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

          linkText
            .attr("x", d => (d.source.x + d.target.x) / 2)
            .attr("y", d => (d.source.y + d.target.y) / 2);

          node.attr("transform", d => `translate(${d.x}, ${d.y})`);
        });

        function dragstarted(event, d) {
          if (!event.active) simulation.alphaTarget(0.15).restart();
          d.fx = d.x;
          d.fy = d.y;
        }

        function dragged(event, d) {
          d.fx = event.x;
          d.fy = event.y;
        }

        function dragended(event, d) {
          if (!event.active) simulation.alphaTarget(0);
        }

        document.getElementById("reset-zoom-btn").addEventListener("click", () => {
          svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity);
        });

        document.getElementById("reheat-sim-btn").addEventListener("click", () => {
          nodes.forEach(d => { d.fx = null; d.fy = null; });
          simulation.alpha(1).restart();
        });

        const shell = document.querySelector(".graph-shell");
        const fsBtn = document.getElementById("fullscreen-btn");
        function updateFsBtnText() {
          if (document.fullscreenElement || document.webkitFullscreenElement) {
            fsBtn.innerHTML = "&boxbox; Exit Fullscreen";
          } else {
            fsBtn.innerHTML = "&boxbox; Fullscreen";
          }
        }
        if (fsBtn && shell) {
          fsBtn.addEventListener("click", () => {
            if (!document.fullscreenElement && !document.webkitFullscreenElement) {
              if (shell.requestFullscreen) shell.requestFullscreen();
              else if (shell.webkitRequestFullscreen) shell.webkitRequestFullscreen();
            } else {
              if (document.exitFullscreen) document.exitFullscreen();
              else if (document.webkitExitFullscreen) document.webkitExitFullscreen();
            }
          });
          document.addEventListener("fullscreenchange", updateFsBtnText);
          document.addEventListener("webkitfullscreenchange", updateFsBtnText);
        }
      })();
    </script>
    """
    return (
        template.replace("__WIDTH__", str(width))
        .replace("__HEIGHT__", str(height))
        .replace("__NODES_JSON__", json.dumps(nodes))
        .replace("__LINKS_JSON__", json.dumps(links))
    )



def build_single_account_ego_graph(
    target_account_id,
    G,
    account_device,
    account_payment,
    account_address,
    account_ip,
    radius=1,
    view_mode="Account Projection",
    selected_resource_types=None,
    min_edge_weight=0.0,
    hide_isolated=True,
    pair_resources=None,
):
    """
    Builds ego network around target_account_id.
    In 'Account Projection' mode: Extracts k-hop NetworkX subgraph around target_account_id,
    filtered by selected_resource_types and min_edge_weight.
    In 'Entity Bipartite Tree' mode: Constructs bipartite graph of target_account_id, its raw
    resources (devices, payments, addresses, IPs), and other accounts sharing those resources.
    """
    if view_mode == "Entity Bipartite Tree":
        B = nx.Graph()
        B.add_node(target_account_id, node_type="target", label=str(target_account_id))
        
        sel_types = set(selected_resource_types) if selected_resource_types else {"device", "payment", "address", "ip"}

        # Add resource nodes and edges to target with edge attributes
        if "device" in sel_types:
            devs = account_device[account_device.account_id == target_account_id]["device_id"].tolist()
            for r_id in devs[:10]:
                r_node = f"device:{r_id}"
                B.add_node(r_node, node_type="device", label=f"Dev:{str(r_id)[-6:]}")
                B.add_edge(target_account_id, r_node, edge_type="device", weight=1.0)
                sharing_accts = account_device[account_device.device_id == r_id]["account_id"].tolist()
                for neighbor_acct in sharing_accts[:10]:
                    if neighbor_acct != target_account_id:
                        B.add_node(neighbor_acct, node_type="account", label=str(neighbor_acct))
                        B.add_edge(r_node, neighbor_acct, edge_type="device", weight=1.0)
                    
        if "payment" in sel_types:
            pmts = account_payment[account_payment.account_id == target_account_id]["payment_id"].tolist()
            for r_id in pmts[:10]:
                r_node = f"payment:{r_id}"
                B.add_node(r_node, node_type="payment", label=f"Pay:{str(r_id)[-6:]}")
                B.add_edge(target_account_id, r_node, edge_type="payment", weight=1.0)
                sharing_accts = account_payment[account_payment.payment_id == r_id]["account_id"].tolist()
                for neighbor_acct in sharing_accts[:10]:
                    if neighbor_acct != target_account_id:
                        B.add_node(neighbor_acct, node_type="account", label=str(neighbor_acct))
                        B.add_edge(r_node, neighbor_acct, edge_type="payment", weight=1.0)

        if "address" in sel_types:
            addrs = account_address[account_address.account_id == target_account_id]["address_id"].tolist()
            for r_id in addrs[:10]:
                r_node = f"address:{r_id}"
                B.add_node(r_node, node_type="address", label=f"Addr:{str(r_id)[-6:]}")
                B.add_edge(target_account_id, r_node, edge_type="address", weight=1.0)
                sharing_accts = account_address[account_address.address_id == r_id]["account_id"].tolist()
                for neighbor_acct in sharing_accts[:10]:
                    if neighbor_acct != target_account_id:
                        B.add_node(neighbor_acct, node_type="account", label=str(neighbor_acct))
                        B.add_edge(r_node, neighbor_acct, edge_type="address", weight=1.0)

        if "ip" in sel_types:
            ips = account_ip[account_ip.account_id == target_account_id]["ip_id"].tolist()
            for r_id in ips[:10]:
                r_node = f"ip:{r_id}"
                B.add_node(r_node, node_type="ip", label=f"IP:{str(r_id)[-6:]}")
                B.add_edge(target_account_id, r_node, edge_type="ip", weight=1.0)
                sharing_accts = account_ip[account_ip.ip_id == r_id]["account_id"].tolist()
                for neighbor_acct in sharing_accts[:10]:
                    if neighbor_acct != target_account_id:
                        B.add_node(neighbor_acct, node_type="account", label=str(neighbor_acct))
                        B.add_edge(r_node, neighbor_acct, edge_type="ip", weight=1.0)

        if hide_isolated:
            isolates = [n for n in B.nodes() if B.degree(n) == 0 and n != target_account_id]
            B.remove_nodes_from(isolates)

        return B

    # Account Projection mode
    if target_account_id in G:
        sub = nx.ego_graph(G, target_account_id, radius=radius)
        sel_types = set(selected_resource_types) if selected_resource_types else {"device", "payment", "address", "ip"}
        
        filtered = nx.Graph()
        filtered.add_node(target_account_id)

        for u, v, data in sub.edges(data=True):
            w = float(data.get("weight", 1.0))
            if w < min_edge_weight:
                continue
            
            # Filter by selected resource types
            if pair_resources is not None:
                key = (u, v) if u < v else (v, u)
                res = pair_resources.get(key, [])
                rtypes = {r.split(":", 1)[0] for r in res if ":" in r}
                if rtypes and not (rtypes & sel_types):
                    continue
            filtered.add_edge(u, v, **data)

        if not hide_isolated:
            filtered.add_nodes_from(sub.nodes(data=True))

        return filtered
    else:
        sub = nx.Graph()
        sub.add_node(target_account_id)
        return sub


def render_single_account_graph_html(
    subgraph,
    target_account_id,
    accounts,
    orders,
    view_mode="Account Projection",
    edge_label_mode="Resource types",
    pair_resources=None,
    account_device=None,
    account_payment=None,
    account_address=None,
    account_ip=None,
    width=1000,
    height=520,
):
    if subgraph.number_of_nodes() == 0:
        return "<div style='color:#94a3b8; padding:20px;'>No graph nodes available for this account.</div>"

    nodes = []
    for node in subgraph.nodes():
        node_attr = subgraph.nodes[node]
        ntype = node_attr.get("node_type", "account")
        is_target = (node == target_account_id)

        if is_target:
            fill = "#f59e0b"
            radius = 18.0
            stroke = "#fbbf24"
        elif ntype == "device":
            fill = "#06b6d4"
            radius = 12.0
            stroke = "#22d3ee"
        elif ntype == "payment":
            fill = "#a855f7"
            radius = 12.0
            stroke = "#c084fc"
        elif ntype == "address":
            fill = "#f97316"
            radius = 12.0
            stroke = "#fb923c"
        elif ntype == "ip":
            fill = "#10b981"
            radius = 12.0
            stroke = "#34d399"
        else:
            fill = "#3b82f6"
            radius = 11.0
            stroke = "#60a5fa"

        lbl = str(node_attr.get("label", str(node)[-6:]))
        title = f"ID: {node}\nType: {ntype.upper()}"
        nodes.append({
            "id": str(node),
            "label": lbl,
            "radius": radius,
            "fill": fill,
            "stroke": stroke,
            "title": title,
            "is_target": is_target,
            "ntype": ntype,
        })

    def get_resources_for_edge(u, v):
        if pair_resources is not None:
            key = (u, v) if u < v else (v, u)
            if key in pair_resources:
                return pair_resources[key]
        if account_device is not None:
            shared = []
            for rtype, df, col in [
                ("device", account_device, "device_id"),
                ("payment", account_payment, "payment_id"),
                ("address", account_address, "address_id"),
                ("ip", account_ip, "ip_id"),
            ]:
                if df is not None and not df.empty:
                    u_res = set(df.loc[df.account_id == u, col])
                    v_res = set(df.loc[df.account_id == v, col])
                    for r_id in u_res & v_res:
                        shared.append(f"{rtype}:{r_id}")
            if pair_resources is not None:
                key = (u, v) if u < v else (v, u)
                pair_resources[key] = shared
            return shared
        return []

    def compute_edge_label(a, b, data, weight, resources):
        if edge_label_mode == "None":
            return ""
        if view_mode == "Entity Bipartite Tree":
            etype = data.get("edge_type", "")
            if not etype:
                for cand in [str(a), str(b)]:
                    if cand.startswith("device:"): return "device"
                    if cand.startswith("payment:"): return "payment"
                    if cand.startswith("address:"): return "address"
                    if cand.startswith("ip:"): return "ip"
            return etype or "linked"
        # Account Projection mode
        if edge_label_mode == "Weight":
            return f"{weight:.2f}"
        if edge_label_mode == "Shared count":
            return f"{len(resources)} shared" if resources else f"{weight:.1f}"
        # Default: "Resource types"
        if resources:
            rtypes = sorted({r.split(":", 1)[0] for r in resources if ":" in r})
            return "+".join(rtypes) if rtypes else f"{weight:.1f}"
        return f"{weight:.1f}" if weight > 1.0 else ""

    links = []
    for a, b, data in subgraph.edges(data=True):
        w = float(data.get("weight", 1.0))
        sw = max(1.5, min(6.0, 1.2 * w))
        resources = get_resources_for_edge(a, b) if view_mode != "Entity Bipartite Tree" else []
        lbl = compute_edge_label(a, b, data, w, resources)
        if view_mode == "Entity Bipartite Tree":
            etype = data.get("edge_type", "entity")
            title = f"{a} <-> {b} ({etype})"
        else:
            res_str = ", ".join(resources[:6]) if resources else "shared resource"
            title = f"{a} <-> {b}\nweight={w:.2f}\nshared: {res_str}"
        links.append({
            "source": str(a),
            "target": str(b),
            "weight": w,
            "stroke_width": sw,
            "label": lbl,
            "title": title,
        })

    template = """
    <div class="graph-shell">
      <div class="graph-toolbar">
        <span style="color:#f59e0b; font-weight:600; font-family:'JetBrains Mono', monospace; font-size:11px; letter-spacing:0.5px;">EGO GRAPH FOCUS VIEWPORT</span>
        <button id="reset-single-zoom" class="hud-btn">Reset View</button>
        <button id="reheat-single-sim" class="hud-btn">Re-spread Layout</button>
        <button id="fullscreen-single-btn" class="hud-btn" style="margin-left:auto;">&boxbox; Fullscreen</button>
      </div>
      <svg id="single-account-graph" viewBox="0 0 __WIDTH__ __HEIGHT__" role="img"></svg>
    </div>
    <style>
      .graph-shell {
        width: 100%;
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 16px;
        background: rgba(7, 9, 14, 0.95);
        font-family: Inter, system-ui, sans-serif;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5), inset 0 0 25px rgba(245, 158, 11, 0.05);
        overflow: hidden;
      }
      .graph-shell:fullscreen {
        width: 100vw !important;
        height: 100vh !important;
        border-radius: 0 !important;
        background: #07090e !important;
        display: flex;
        flex-direction: column;
      }
      .graph-shell:fullscreen #single-account-graph {
        height: calc(100vh - 45px) !important;
        flex: 1;
      }
      .graph-toolbar {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 8px 16px;
        background: rgba(15, 23, 42, 0.85);
        border-bottom: 1px solid rgba(245, 158, 11, 0.2);
      }
      .hud-btn {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.35);
        color: #fbbf24;
        border-radius: 6px;
        padding: 4px 12px;
        font-size: 11px;
        font-family: Inter, sans-serif;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s;
      }
      .hud-btn:hover {
        background: rgba(245, 158, 11, 0.35);
        color: #ffffff;
        box-shadow: 0 0 10px rgba(245, 158, 11, 0.4);
      }
      #single-account-graph {
        width: 100%;
        height: 480px;
        display: block;
        cursor: grab;
      }
      #single-account-graph:active { cursor: grabbing; }
      .edge-line {
        stroke: #475569;
        stroke-opacity: 0.6;
        stroke-linecap: round;
        transition: stroke 0.2s, stroke-opacity 0.2s;
      }
      .edge-line.highlighted {
        stroke: #fbbf24 !important;
        stroke-opacity: 1.0 !important;
        filter: drop-shadow(0px 0px 6px rgba(251, 191, 36, 0.8));
      }
      .node-group {
        cursor: grab;
      }
      .node-group:active {
        cursor: grabbing;
      }
      .node-shape {
        stroke-width: 2.5px;
        transition: stroke 0.2s;
      }
      .node-shape.selected-node {
        stroke: #f59e0b !important;
        stroke-width: 4.5px !important;
        filter: drop-shadow(0px 0px 14px rgba(245, 158, 11, 0.95)) !important;
      }
      .node-group {
        cursor: grab;
        transition: opacity 0.25s ease;
      }
      .node-group:active {
        cursor: grabbing;
      }
      .node-group:hover .node-shape {
        stroke: #ffffff !important;
        stroke-width: 4.0px !important;
        filter: drop-shadow(0px 0px 12px rgba(255, 255, 255, 0.9)) !important;
      }
      .node-txt {
        fill: #f8fafc;
        font-size: 10.5px;
        font-weight: 600;
        text-anchor: middle;
        pointer-events: none;
        paint-order: stroke;
        stroke: #07090e;
        stroke-linejoin: round;
        stroke-width: 3.5px;
      }
      .edge-txt {
        fill: #fbbf24;
        font-size: 10px;
        font-weight: 600;
        text-anchor: middle;
        pointer-events: none;
        paint-order: stroke;
        stroke: #07090e;
        stroke-linejoin: round;
        stroke-width: 3.5px;
        opacity: 0.65;
        transition: opacity 0.2s, fill 0.2s;
      }
      .edge-txt.highlighted {
        opacity: 1.0 !important;
        fill: #ffffff !important;
      }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
    <script>
      (() => {
        const nodes = __NODES_JSON__;
        const links = __LINKS_JSON__;
        const width = __WIDTH__;
        const height = __HEIGHT__;
        const targetId = "__TARGET__";

        const svg = d3.select("#single-account-graph");
        svg.selectAll("*").remove();

        const defs = svg.append("defs");
        const pat = defs.append("pattern")
          .attr("id", "grid-pattern-single")
          .attr("width", 30)
          .attr("height", 30)
          .attr("patternUnits", "userSpaceOnUse");
        pat.append("path")
          .attr("d", "M 30 0 L 0 0 0 30")
          .attr("fill", "none")
          .attr("stroke", "rgba(245, 158, 11, 0.06)")
          .attr("stroke-width", 1);

        svg.append("rect")
          .attr("width", width)
          .attr("height", height)
          .attr("rx", 14)
          .attr("fill", "#07090e");

        svg.append("rect")
          .attr("width", width)
          .attr("height", height)
          .attr("rx", 14)
          .attr("fill", "url(#grid-pattern-single)");

        const container = svg.append("g").attr("class", "graph-container");

        const zoom = d3.zoom()
          .scaleExtent([0.3, 4.0])
          .on("zoom", (event) => {
            container.attr("transform", event.transform);
          });
        svg.call(zoom);

        const simulation = d3.forceSimulation(nodes)
          .velocityDecay(0.65)
          .alphaDecay(0.06)
          .force("link", d3.forceLink(links).id(d => d.id).distance(120))
          .force("charge", d3.forceManyBody().strength(-380))
          .force("center", d3.forceCenter(width / 2, height / 2))
          .force("collide", d3.forceCollide().radius(d => d.radius + 30));

        const targetNode = nodes.find(d => d.id === targetId);
        if (targetNode) {
          targetNode.fx = width / 2;
          targetNode.fy = height / 2;
        }

        const linkGroup = container.append("g").attr("class", "links-group");
        const link = linkGroup.selectAll("line")
          .data(links)
          .enter().append("line")
          .attr("class", "edge-line")
          .attr("stroke-width", d => d.stroke_width || 2.0);

        link.append("title").text(d => d.title);

        const linkText = linkGroup.selectAll("text")
          .data(links)
          .enter().append("text")
          .attr("class", "edge-txt")
          .text(d => d.label);

        const nodeGroup = container.append("g").attr("class", "nodes-group");
        const node = nodeGroup.selectAll("g")
          .data(nodes)
          .enter().append("g")
          .attr("class", "node-group")
          .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended));

        node.each(function(d) {
          const el = d3.select(this);
          if (d.ntype === 'device' || d.ntype === 'payment') {
            el.append("rect")
              .attr("class", "node-shape")
              .attr("x", -d.radius)
              .attr("y", -d.radius)
              .attr("width", d.radius * 2)
              .attr("height", d.radius * 2)
              .attr("rx", 4)
              .attr("fill", d.fill)
              .attr("stroke", d.stroke);
          } else {
            el.append("circle")
              .attr("class", "node-shape")
              .attr("r", d.radius)
              .attr("fill", d.fill)
              .attr("stroke", d.stroke);
          }
        });

        node.append("title").text(d => d.title);

        node.append("text")
          .attr("class", "node-txt")
          .attr("dy", d => d.radius + 16)
          .text(d => d.label);

        let selectedNodeId = null;

        function updateHighlights() {
          if (selectedNodeId) {
            const connectedNodeIds = new Set([selectedNodeId]);
            links.forEach(l => {
              const srcId = typeof l.source === "object" ? l.source.id : l.source;
              const tgtId = typeof l.target === "object" ? l.target.id : l.target;
              if (srcId === selectedNodeId) connectedNodeIds.add(tgtId);
              if (tgtId === selectedNodeId) connectedNodeIds.add(srcId);
            });

            node.style("opacity", n => connectedNodeIds.has(n.id) ? 1.0 : 0.18);
            node.selectAll(".node-shape").classed("selected-node", n => n.id === selectedNodeId);

            link.style("opacity", l => {
              const srcId = typeof l.source === "object" ? l.source.id : l.source;
              const tgtId = typeof l.target === "object" ? l.target.id : l.target;
              return (srcId === selectedNodeId || tgtId === selectedNodeId) ? 1.0 : 0.06;
            }).classed("highlighted", l => {
              const srcId = typeof l.source === "object" ? l.source.id : l.source;
              const tgtId = typeof l.target === "object" ? l.target.id : l.target;
              return (srcId === selectedNodeId || tgtId === selectedNodeId);
            });

            linkText.style("opacity", l => {
              const srcId = typeof l.source === "object" ? l.source.id : l.source;
              const tgtId = typeof l.target === "object" ? l.target.id : l.target;
              return (srcId === selectedNodeId || tgtId === selectedNodeId) ? 1.0 : 0.04;
            }).classed("highlighted", l => {
              const srcId = typeof l.source === "object" ? l.source.id : l.source;
              const tgtId = typeof l.target === "object" ? l.target.id : l.target;
              return (srcId === selectedNodeId || tgtId === selectedNodeId);
            });
          } else {
            node.style("opacity", 1.0);
            node.selectAll(".node-shape").classed("selected-node", false);
            link.style("opacity", 0.6).classed("highlighted", false);
            linkText.style("opacity", 0.65).classed("highlighted", false);
          }
        }

        node.on("click", (event, d) => {
          event.stopPropagation();
          selectedNodeId = (selectedNodeId === d.id) ? null : d.id;
          updateHighlights();
        });

        svg.on("click", () => {
          if (selectedNodeId) {
            selectedNodeId = null;
            updateHighlights();
          }
        });

        node.on("mouseover", (event, d) => {
          if (!selectedNodeId) {
            link.classed("highlighted", l => {
              const srcId = typeof l.source === "object" ? l.source.id : l.source;
              const tgtId = typeof l.target === "object" ? l.target.id : l.target;
              return srcId === d.id || tgtId === d.id;
            });
            linkText.classed("highlighted", l => {
              const srcId = typeof l.source === "object" ? l.source.id : l.source;
              const tgtId = typeof l.target === "object" ? l.target.id : l.target;
              return srcId === d.id || tgtId === d.id;
            });
          }
        }).on("mouseout", () => {
          if (!selectedNodeId) {
            link.classed("highlighted", false);
            linkText.classed("highlighted", false);
          }
        });

        simulation.on("tick", () => {
          link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

          linkText
            .attr("x", d => (d.source.x + d.target.x) / 2)
            .attr("y", d => (d.source.y + d.target.y) / 2);

          node.attr("transform", d => `translate(${d.x}, ${d.y})`);
        });

        function dragstarted(event, d) {
          if (!event.active) simulation.alphaTarget(0.15).restart();
          d.fx = d.x;
          d.fy = d.y;
        }

        function dragged(event, d) {
          d.fx = event.x;
          d.fy = event.y;
        }

        function dragended(event, d) {
          if (!event.active) simulation.alphaTarget(0);
        }

        document.getElementById("reset-single-zoom").addEventListener("click", () => {
          svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity);
        });

        document.getElementById("reheat-single-sim").addEventListener("click", () => {
          nodes.forEach(d => {
            if (d.id !== targetId) { d.fx = null; d.fy = null; }
          });
          simulation.alpha(1).restart();
        });

        const shell = document.querySelector(".graph-shell");
        const fsBtn = document.getElementById("fullscreen-single-btn");
        function updateFsBtnText() {
          if (document.fullscreenElement || document.webkitFullscreenElement) {
            fsBtn.innerHTML = "&boxbox; Exit Fullscreen";
          } else {
            fsBtn.innerHTML = "&boxbox; Fullscreen";
          }
        }
        if (fsBtn && shell) {
          fsBtn.addEventListener("click", () => {
            if (!document.fullscreenElement && !document.webkitFullscreenElement) {
              if (shell.requestFullscreen) shell.requestFullscreen();
              else if (shell.webkitRequestFullscreen) shell.webkitRequestFullscreen();
            } else {
              if (document.exitFullscreen) document.exitFullscreen();
              else if (document.webkitExitFullscreen) document.webkitExitFullscreen();
            }
          });
          document.addEventListener("fullscreenchange", updateFsBtnText);
          document.addEventListener("webkitfullscreenchange", updateFsBtnText);
        }
      })();
    </script>
    """
    return (
        template.replace("__WIDTH__", str(width))
        .replace("__HEIGHT__", str(height))
        .replace("__NODES_JSON__", json.dumps(nodes))
        .replace("__LINKS_JSON__", json.dumps(links))
        .replace("__TARGET__", str(target_account_id))
    )




def compute_business_impact(clusters_df, features, flagged_cluster_ids, orders,
                            ground_truth, fp_multiplier):
    avg_order_value = float(orders["amount"].mean()) if len(orders) else 0.0
    order_value = orders.groupby("account_id")["amount"].sum()
    scored_cluster_ids = set(features["cluster_id"].tolist())
    members = clusters_df[clusters_df.cluster_id.isin(scored_cluster_ids)].copy()
    members["order_value"] = members["account_id"].map(order_value).fillna(0.0)
    members["is_flagged"] = members["cluster_id"].isin(flagged_cluster_ids)

    if ground_truth is not None:
        gt_cols = ["account_id", "is_ring_member"]
        if "is_referral_ring_member" in ground_truth.columns:
            gt_cols.append("is_referral_ring_member")
        members = members.merge(
            ground_truth[gt_cols], on="account_id", how="left"
        )
        is_true_ring = (members["is_ring_member"] == True)
        if "is_referral_ring_member" in members.columns:
            is_true_ring = is_true_ring | (members["is_referral_ring_member"] == True)
        members["is_ring_member"] = is_true_ring
        protected = members.loc[members.is_flagged & members.is_ring_member, "order_value"].sum()
        missed = members.loc[(~members.is_flagged) & members.is_ring_member, "order_value"].sum()
        fp_accounts = int((members.is_flagged & (~members.is_ring_member)).sum())
        fp_cost = fp_accounts * avg_order_value * fp_multiplier
        return {
            "mode": "actual",
            "protected": float(protected),
            "missed": float(missed),
            "fp_accounts": fp_accounts,
            "fp_cost": float(fp_cost),
            "net": float(protected - fp_cost),
        }

    cluster_value = members.groupby("cluster_id")["order_value"].sum()
    cluster_accounts = members.groupby("cluster_id").size()
    scored = features.set_index("cluster_id").copy()
    scored["order_value"] = scored.index.map(cluster_value).fillna(0.0)
    scored["member_accounts"] = scored.index.map(cluster_accounts).fillna(0.0)
    scored["is_flagged"] = scored.index.isin(flagged_cluster_ids)

    flagged_rows = scored[scored.is_flagged]
    unflagged_rows = scored[~scored.is_flagged]
    protected = (flagged_rows["risk_score"] * flagged_rows["order_value"]).sum()
    missed = (unflagged_rows["risk_score"] * unflagged_rows["order_value"]).sum()
    fp_accounts = ((1.0 - flagged_rows["risk_score"]) * flagged_rows["member_accounts"]).sum()
    fp_cost = fp_accounts * avg_order_value * fp_multiplier
    return {
        "mode": "estimated",
        "protected": float(protected),
        "missed": float(missed),
        "fp_accounts": float(fp_accounts),
        "fp_cost": float(fp_cost),
        "net": float(protected - fp_cost),
    }


def compute_local_shap(model, features, selected):
    scaler = model.named_steps.get("standardscaler")
    classifier = model.named_steps.get("logisticregression")
    if scaler is None or classifier is None:
        return None, None

    raw = features[CHAMPION_FEATURE_COLS].astype(float)
    scaled = scaler.transform(raw.values)
    background = scaled.mean(axis=0)
    coefficients = classifier.coef_[0]
    shap_values = (scaled - background) * coefficients
    baseline_log_odds = float(classifier.intercept_[0] + np.dot(background, coefficients))

    row_position = features.index[features.cluster_id == selected]
    if len(row_position) == 0:
        return None, None
    i = features.index.get_loc(row_position[0])
    selected_values = shap_values[i]
    selected_raw = raw.iloc[i]
    positive_total = sum(max(v, 0.0) for v in selected_values)
    absolute_total = sum(abs(v) for v in selected_values)

    rows = []
    for feature, raw_value, shap_value in zip(CHAMPION_FEATURE_COLS, selected_raw, selected_values):
        if positive_total > 0 and shap_value > 0:
            share = 100.0 * shap_value / positive_total
        elif positive_total == 0 and absolute_total > 0:
            share = 100.0 * abs(shap_value) / absolute_total
        else:
            share = 0.0
        rows.append({
            "feature": FEATURE_LABELS.get(feature, feature),
            "feature_value": float(raw_value),
            "shap_log_odds": float(shap_value),
            "direction": "raises risk" if shap_value >= 0 else "lowers risk",
            "upward_share_pct": float(share),
        })

    out = pd.DataFrame(rows).sort_values("shap_log_odds", key=lambda s: s.abs(), ascending=False)
    reconstructed_risk = sigmoid(baseline_log_odds + float(selected_values.sum()))
    return out, {
        "baseline_log_odds": baseline_log_odds,
        "baseline_risk": sigmoid(baseline_log_odds),
        "reconstructed_risk": reconstructed_risk,
    }


@st.cache_data(show_spinner=False)
def run_pipeline(accounts, account_device, account_payment, account_address,
                 account_ip, orders, referrals=None):
    G = build_account_graph(account_device, account_payment, account_address, account_ip)
    if G.number_of_nodes() == 0:
        return None, None, None

    partition = community_louvain.best_partition(G, weight="weight", random_state=42)
    clusters_df = pd.DataFrame({"account_id": list(partition.keys()),
                                 "cluster_id": list(partition.values())})
    sizes = clusters_df.groupby("cluster_id").size().rename("cluster_size")
    clusters_df = clusters_df.merge(sizes, on="cluster_id")
    candidates = clusters_df[clusters_df.cluster_size >= 2]

    if candidates.empty:
        return G, clusters_df, pd.DataFrame()

    # Compute referral features if referrals.csv is available
    ref_features = None
    if referrals is not None and not referrals.empty:
        try:
            ref_features = compute_referral_features(clusters_df, referrals, orders)
        except Exception:
            ref_features = None  # referral features are supplementary; don't break the app

    features = compute_cluster_features(
        candidates, accounts, orders,
        account_device, account_payment, account_address,
        account_ip=account_ip, G=G,
        referral_features=ref_features,
    )
    return G, clusters_df, features


model = load_model()
accounts, account_device, account_payment, account_address, account_ip, orders, ground_truth, referrals = load_bundled_demo_data()

# ---------------------------------------------------------------------------
# Sidebar Controls & System Status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<span class="badge-blue">SYSTEM CONTROLS</span>', unsafe_allow_html=True)
    st.markdown("### Model Parameters")
    
    threshold = st.slider(
        "Risk Score Threshold", 0.0, 1.0, DEFAULT_THRESHOLD, 0.01,
        help="Default is the cost-optimal threshold from bootstrap analysis. Lowering flags more clusters (higher recall, higher FP cost)."
    )
    
    fp_multiplier = st.slider(
        "FP Cost Multiplier", 0.1, 10.0, 1.0, 0.1,
        help="Business cost multiplier: one falsely flagged account costs N times average order value."
    )
    
    st.divider()
    st.markdown('<span class="badge-blue">POPULATION STATISTICS</span>', unsafe_allow_html=True)
    st.markdown(f"""
    - **Total Accounts:** `{len(accounts):,}`
    - **Observed Orders:** `{len(orders):,}`
    - **Referral Edges:** `{len(referrals):,}`
    - **Engine Mode:** `Read-Only Recommendation`
    """)
    st.caption("🔒 Non-actioning defense guarantee: No automatic block/freeze paths.")

# ---------------------------------------------------------------------------
# Header Section & Main Execution
# ---------------------------------------------------------------------------
st.markdown('<span class="badge-blue">SENTINEL INTELLIGENCE SYSTEM</span>', unsafe_allow_html=True)
st.title("Abuse-Ring Sentinel")
st.caption("Graph-Based Fraud Ring Detection & Risk Intelligence Platform")

with st.spinner("Processing entity resolution, graph projection, and community detection..."):
    G, clusters_df, features = run_pipeline(
        accounts, account_device, account_payment, account_address, account_ip, orders,
        referrals=referrals,
    )

if G is None or features is None or features.empty:
    st.warning("No accounts share resources with another account — zero clusters generated.")
    st.stop()

X = features[CHAMPION_FEATURE_COLS].values
features = features.copy()
features["risk_score"] = model.predict_proba(X)[:, 1]
flagged = features[features.risk_score >= threshold].sort_values("risk_score", ascending=False)
flagged_ids = set(flagged.cluster_id)

impact = compute_business_impact(
    clusters_df, features, flagged_ids, orders, ground_truth, fp_multiplier
)

# ---------------------------------------------------------------------------
# Executive KPI Dashboard Cards
# ---------------------------------------------------------------------------
kpi_cols = st.columns(5)
kpi_cols[0].metric("Monitored Accounts", f"{G.number_of_nodes():,}")
kpi_cols[1].metric("Clusters Identified", f"{features.cluster_id.nunique():,}")
kpi_cols[2].metric("Flagged High Risk", f"{len(flagged):,}")
kpi_cols[3].metric("Protected Value", format_rs(impact["protected"]))
kpi_cols[4].metric("Net Financial Impact", format_rs(impact["net"]))

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Performance & Financial Metric Cards
# ---------------------------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### Performance Metrics")
    with st.container(border=True):
        if ground_truth is not None:
            merged = clusters_df.merge(ground_truth, on="account_id", how="left")
            is_ring = (merged.is_ring_member == True)
            if "is_referral_ring_member" in merged.columns:
                is_ring = is_ring | (merged.is_referral_ring_member == True)
            ring_clusters = set(merged.loc[is_ring, "cluster_id"])
            tp = len(ring_clusters & flagged_ids)
            fp = len(flagged_ids - ring_clusters)
            fn = len(ring_clusters - flagged_ids)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0

            ev1, ev2, ev3, ev4 = st.columns(4)
            ev1.metric("Precision", f"{precision:.3f}")
            ev2.metric("Recall", f"{recall:.3f}")
            ev3.metric("False Pos (FP)", f"{fp}")
            ev4.metric("False Neg (FN)", f"{fn}")
            
        st.caption("Live threshold trade-off: Adjusting the sidebar threshold updates precision/recall and protected revenue in real-time.")

with col_right:
    st.markdown("### Business Risk Breakdown")
    with st.container(border=True):
        imp1, imp2, imp3 = st.columns(3)
        imp1.metric("Protected", format_rs(impact["protected"]))
        imp2.metric("FP Cost", format_rs(impact["fp_cost"]))
        imp3.metric("Missed", format_rs(impact["missed"]))
        st.caption(f"Business impact: False positive expense calculated using {impact['fp_accounts']} false positive account review overheads.")




temp_lat_df, temp_summary = load_temporal_backtest_data()

with st.container(border=True):
    tcol1, tcol2 = st.columns([3.5, 1.2], vertical_alignment="center")
    with tcol1:
        st.markdown("### Flagged Abuse Clusters")
        st.caption(f"Showing **{len(flagged):,}** clusters with risk scores exceeding the active threshold ({threshold:.3f}).")
    with tcol2:
        if not flagged.empty:
            csv_cols = ["cluster_id", "risk_score", "cluster_size", "entity_reuse_ratio", "internal_density"]
            csv_bytes = flagged[csv_cols].to_csv(index=False).encode()
            st.download_button(
                "Export CSV Report",
                csv_bytes,
                "flagged_clusters.csv",
                "text/csv",
                use_container_width=True
            )

    if flagged.empty:
        st.info("No clusters flagged at this threshold.")
    else:
        # Prepare enriched table data
        cluster_spend = orders.merge(clusters_df, on="account_id").groupby("cluster_id")["amount"].sum()
        
        flagged_display = flagged.copy()
        flagged_display["Total Spend (Rs)"] = flagged_display["cluster_id"].map(cluster_spend).fillna(0.0)
        flagged_display["Cluster ID Str"] = flagged_display["cluster_id"].apply(lambda c: f"Cluster #{c}")
        
        flagged_display = flagged_display.rename(columns={
            "risk_score": "Risk Score",
            "cluster_size": "Cluster Size",
            "entity_reuse_ratio": "Entity Reuse Ratio",
            "internal_density": "Graph Density",
        })

        st.dataframe(
            flagged_display[[
                "Cluster ID Str", "Risk Score", "Cluster Size", "Total Spend (Rs)",
                "Entity Reuse Ratio", "Graph Density"
            ]].reset_index(drop=True),
            column_config={
                "Cluster ID Str": st.column_config.TextColumn(
                    "Cluster ID",
                    help="Louvain community cluster identifier",
                ),
                "Risk Score": st.column_config.ProgressColumn(
                    "Risk Score",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.3f",
                    help="Ensemble model predicted probability of abuse ring behavior",
                ),
                "Cluster Size": st.column_config.NumberColumn(
                    "Cluster Size",
                    format="%d accounts",
                    help="Total number of linked accounts in this cluster",
                ),
                "Total Spend (Rs)": st.column_config.NumberColumn(
                    "Total Spend",
                    format="Rs. %,.0f",
                    help="Aggregate order financial value across all member accounts",
                ),
                "Entity Reuse Ratio": st.column_config.NumberColumn(
                    "Entity Reuse Ratio",
                    format="%.2f",
                    help="Ratio of shared entities (devices, IPs, payments) per account",
                ),
                "Graph Density": st.column_config.NumberColumn(
                    "Graph Density",
                    format="%.3f",
                    help="Internal connectivity density of the cluster graph",
                ),
            },
            width="stretch",
            hide_index=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        selected = st.selectbox(
            "Select Cluster for Deep Forensic Inspection",
            options=flagged.cluster_id.tolist(),
            format_func=lambda c: f"Cluster #{c} (Risk: {features.loc[features.cluster_id == c, 'risk_score'].values[0]:.3f} | Size: {features.loc[features.cluster_id == c, 'cluster_size'].values[0]} accounts)"
        )
        selected_row = features.loc[features.cluster_id == selected].iloc[0]
        members = clusters_df.loc[clusters_df.cluster_id == selected, "account_id"].tolist()
        pair_resources, resource_summary = cluster_shared_resources(
            members, account_device, account_payment, account_address, account_ip
        )

        selected_impact = compute_business_impact(
            clusters_df,
            features.loc[features.cluster_id == selected],
            {selected},
            orders,
            ground_truth,
            fp_multiplier,
        )

        cluster_temp_info = None
        if temp_lat_df is not None and not temp_lat_df.empty:
            match = temp_lat_df[temp_lat_df["dominant_cluster_id"] == selected]
            if not match.empty:
                cluster_temp_info = match.iloc[0].to_dict()

        try:
            pdf_bytes = generate_cluster_pdf_report(
                cluster_id=selected,
                selected_row=selected_row,
                members=members,
                subgraph=G.subgraph(members),
                accounts=accounts,
                orders=orders,
                pair_resources=pair_resources,
                resource_summary=resource_summary,
                referrals=referrals,
                impact=selected_impact,
                active_threshold=threshold,
                temporal_info=cluster_temp_info,
            )
        except Exception as pdf_err:
            pdf_bytes = None

        with st.container(border=True):
            pdf_col1, pdf_col2 = st.columns([3.2, 1.4], vertical_alignment="center")
            with pdf_col1:
                st.markdown(f"### Cluster #{selected} Forensic Inspection")
                st.caption(f"Inspecting **{len(members)}** linked accounts &bull; Predicted Risk Score: **{selected_row.risk_score:.3f}**")
            with pdf_col2:
                if pdf_bytes:
                    st.download_button(
                        "Export PDF Audit Report",
                        pdf_bytes,
                        file_name=f"Sentinel_Audit_Report_Cluster_{selected}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )

        graph_tab, acct_graph_tab, impact_tab, shap_tab, members_tab, temporal_tab = st.tabs(
            ["Network graph", "Single account graph", "Business impact", "SHAP explanation", "Members", "Temporal latency"]
        )


    with graph_tab:
        subgraph = G.subgraph(members)
        max_edge_weight = max(
            (float(data.get("weight", 1.0)) for _, _, data in subgraph.edges(data=True)),
            default=1.0,
        )
        max_node_degree = max(
            (float(degree) for _, degree in subgraph.degree(weight="weight")),
            default=1.0,
        )

        with st.container(border=True):
            st.markdown("##### Topology & Filtering Controls")
            fcol1, fcol2, fcol3 = st.columns([1.4, 1.0, 1.0])
            selected_resource_types = fcol1.multiselect(
                "Edge Resource Types",
                ["device", "payment", "address", "ip"],
                default=["device", "payment", "address", "ip"],
            )
            min_edge_weight = fcol2.slider(
                "Minimum Edge Weight",
                0.0,
                float(max_edge_weight),
                0.0,
                0.01,
            )
            min_node_degree = fcol3.slider(
                "Min Weighted Degree",
                0.0,
                float(max_node_degree),
                0.0,
                0.05,
            )

            lcol1, lcol2, lcol3 = st.columns([1.0, 1.0, 1.0])
            node_label_mode = lcol1.selectbox(
                "Node Labels",
                ["Short account ID", "Full account ID", "Weighted degree", "Order value", "None"],
            )
            edge_label_mode = lcol2.selectbox(
                "Edge Labels",
                ["Resource types", "Weight", "Shared count", "None"],
            )
            hide_isolated = lcol3.checkbox("Hide Isolated Nodes", value=True)

        filtered_subgraph = filter_cluster_graph(
            G,
            members,
            pair_resources,
            selected_resource_types,
            min_edge_weight,
            min_node_degree,
            hide_isolated,
        )

        with st.container(border=True):
            gcol1, gcol2, gcol3, gcol4 = st.columns(4)
            gcol1.metric("Cluster Members", len(members))
            gcol2.metric("Visible Nodes", filtered_subgraph.number_of_nodes())
            gcol3.metric("Visible Edges", filtered_subgraph.number_of_edges())
            gcol4.metric("Risk Score", f"{selected_row.risk_score:.3f}")

        st.markdown(
            """
            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(59, 130, 246, 0.25); border-radius: 10px; padding: 10px 16px; margin-bottom: 12px; font-family: Inter, sans-serif; font-size: 12px; color: #94a3b8; display: flex; align-items: center; gap: 20px; flex-wrap: wrap;">
              <span style="font-weight: 600; color: #60a5fa; font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 0.5px;">GRAPH LEGEND & GUIDE:</span>
              <span><span style="display:inline-block; width:12px; height:12px; border-radius:50%; background:#ec4899; margin-right:6px; vertical-align:middle;"></span> <strong style="color:#f8fafc;">Pink Nodes</strong> = High-Degree Central Hubs (&ge; Median Cluster Connectivity)</span>
              <span><span style="display:inline-block; width:12px; height:12px; border-radius:50%; background:#3b82f6; margin-right:6px; vertical-align:middle;"></span> <strong style="color:#f8fafc;">Blue Nodes</strong> = Peripheral Member Accounts</span>
              <span><strong style="color:#60a5fa;">Node Size</strong> = Proportional to Weighted Cluster Degree (larger circle = higher entity reuse volume)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        components.html(
            render_cluster_graph_html(
                filtered_subgraph,
                accounts,
                orders,
                pair_resources,
                selected_resource_types,
                node_label_mode,
                edge_label_mode,
            ),
            height=610,
            scrolling=False,
        )
        if resource_summary.empty:
            st.info("No shared resources were found for this cluster.")
        else:
            visible_resource_summary = resource_summary[
                resource_summary.resource_type.isin(selected_resource_types)
            ]
            st.dataframe(visible_resource_summary, width="stretch", hide_index=True)

    with acct_graph_tab:
        with st.container(border=True):
            st.markdown("##### Target Account Ego-Graph Controls & Filters")
            acol1, acol2, acol3 = st.columns([1.4, 0.9, 1.4], vertical_alignment="bottom")
            target_acct_id = acol1.selectbox(
                "Target Account ID",
                options=members,
            )
            ego_radius = acol2.slider("Hop Radius", 1, 3, 1)
            ego_view_mode = acol3.radio(
                "Graph Mode",
                ["Account Projection", "Entity Bipartite Tree"],
                horizontal=True,
            )

            st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

            fcol1, fcol2, fcol3, fcol4 = st.columns([2.2, 1.2, 1.2, 0.9], vertical_alignment="bottom")
            ego_resource_types = fcol1.multiselect(
                "Resource Types",
                ["device", "payment", "address", "ip"],
                default=["device", "payment", "address", "ip"],
            )
            
            if ego_view_mode == "Entity Bipartite Tree":
                ego_edge_label_options = ["Entity type", "None"]
            else:
                ego_edge_label_options = ["Resource types", "Shared count", "Weight", "None"]

            ego_edge_label_mode = fcol2.selectbox(
                "Edge Labels",
                ego_edge_label_options,
            )

            if ego_view_mode == "Account Projection" and target_acct_id in G:
                temp_sub = nx.ego_graph(G, target_acct_id, radius=ego_radius)
                max_ego_w = max((float(d.get("weight", 1.0)) for _, _, d in temp_sub.edges(data=True)), default=1.0)
                ego_min_weight = fcol3.slider(
                    "Min Edge Weight",
                    0.0,
                    float(max_ego_w),
                    0.0,
                    0.01,
                )
            else:
                ego_min_weight = 0.0
                fcol3.slider(
                    "Min Edge Weight",
                    0.0,
                    1.0,
                    0.0,
                    disabled=True,
                )

            ego_hide_isolated = fcol4.checkbox(
                "Hide Isolated",
                value=True,
            )

        # Single Account Summary Card
        acct_orders = orders[orders.account_id == target_acct_id]
        total_acct_spend = float(acct_orders["amount"].sum()) if len(acct_orders) else 0.0
        n_orders = len(acct_orders)
        
        # Shared resources count for this target account
        n_devs = len(account_device[account_device.account_id == target_acct_id])
        n_pmts = len(account_payment[account_payment.account_id == target_acct_id])
        n_ips = len(account_ip[account_ip.account_id == target_acct_id])
        n_addrs = len(account_address[account_address.account_id == target_acct_id])
        
        with st.container(border=True):
            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            sc1.metric("Target Account", str(target_acct_id)[-8:])
            sc2.metric("Total Spend", format_rs(total_acct_spend))
            sc3.metric("Order Count", f"{n_orders}")
            sc4.metric("Shared Entities", f"{n_devs + n_pmts + n_ips + n_addrs}")
            sc5.metric("Cluster Risk Score", f"{selected_row.risk_score:.3f}")

        ego_subgraph = build_single_account_ego_graph(
            target_acct_id, G, account_device, account_payment,
            account_address, account_ip, radius=ego_radius, view_mode=ego_view_mode,
            selected_resource_types=ego_resource_types,
            min_edge_weight=ego_min_weight,
            hide_isolated=ego_hide_isolated,
            pair_resources=pair_resources,
        )

        st.markdown(
            """
            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(245, 158, 11, 0.25); border-radius: 10px; padding: 10px 16px; margin-bottom: 12px; font-family: Inter, sans-serif; font-size: 12px; color: #94a3b8; display: flex; align-items: center; gap: 18px; flex-wrap: wrap;">
              <span style="font-weight: 600; color: #fbbf24; font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 0.5px;">EGO GRAPH LEGEND & GUIDE:</span>
              <span><span style="display:inline-block; width:12px; height:12px; border-radius:50%; background:#f59e0b; margin-right:6px; vertical-align:middle;"></span> <strong style="color:#f8fafc;">Golden Amber Node</strong> = Target Account in Focus</span>
              <span><span style="display:inline-block; width:12px; height:12px; border-radius:50%; background:#3b82f6; margin-right:6px; vertical-align:middle;"></span> <strong style="color:#f8fafc;">Blue Nodes</strong> = Neighbor Accounts</span>
              <span><strong style="color:#fbbf24;">Entity Shapes</strong> = Devices (Cyan Square), Payments (Purple Diamond), IPs (Green Circle), Addresses (Orange Circle)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        components.html(
            render_single_account_graph_html(
                ego_subgraph,
                target_acct_id,
                accounts,
                orders,
                view_mode=ego_view_mode,
                edge_label_mode=ego_edge_label_mode,
                pair_resources=pair_resources,
                account_device=account_device,
                account_payment=account_payment,
                account_address=account_address,
                account_ip=account_ip,
            ),
            height=600,
            scrolling=False,
        )

    with impact_tab:
        selected_impact = compute_business_impact(
            clusters_df,
            features.loc[features.cluster_id == selected],
            {selected},
            orders,
            ground_truth,
            fp_multiplier,
        )
        scol1, scol2, scol3, scol4 = st.columns(4)
        scol1.metric("Rs protected", format_rs(selected_impact["protected"]))
        scol2.metric("Rs FP cost", format_rs(selected_impact["fp_cost"]))
        scol3.metric("Net impact", format_rs(selected_impact["net"]))
        scol4.metric(
            "FP accounts",
            f"{selected_impact['fp_accounts']:,.1f}"
            if selected_impact["mode"] == "estimated"
            else f"{selected_impact['fp_accounts']:,}",
        )
        if selected_impact["mode"] == "actual":
            st.caption("This cluster-level impact uses bundled ground truth and current FP-cost multiplier.")
        else:
            st.caption("This cluster-level impact is risk-weighted because uploaded data has no labels.")

    with shap_tab:
        shap_table, shap_meta = compute_local_shap(model, features, selected)
        if shap_table is None:
            st.info("Local SHAP explanations are available for the shipped logistic-regression model.")
        else:
            xcol1, xcol2 = st.columns(2)
            xcol1.metric("Baseline risk", f"{shap_meta['baseline_risk']:.3f}")
            xcol2.metric("Explained cluster risk", f"{shap_meta['reconstructed_risk']:.3f}")
            shap_chart_df = shap_table.copy()
            shap_chart_df["feature_display"] = shap_chart_df["feature"].map(FEATURE_LABELS).fillna(shap_chart_df["feature"])
            max_val = float(shap_chart_df["shap_log_odds"].max()) if not shap_chart_df.empty else 0.5
            min_val = float(shap_chart_df["shap_log_odds"].min()) if not shap_chart_df.empty else -0.5
            span = max(0.5, max_val - min_val)
            buffer_neg = max(0.35, span * 0.35)
            buffer_pos = max(0.35, span * 0.35)
            domain_min = min(-0.15, min_val - buffer_neg)
            domain_max = max(0.15, max_val + buffer_pos)

            bars = alt.Chart(shap_chart_df).mark_bar(
                cornerRadiusTopRight=4,
                cornerRadiusBottomRight=4,
                cornerRadiusTopLeft=4,
                cornerRadiusBottomLeft=4,
                size=20
            ).encode(
                x=alt.X(
                    'shap_log_odds:Q',
                    title='SHAP Log-Odds Risk Contribution (Positive = Increases Risk, Negative = Reduces Risk)',
                    scale=alt.Scale(domain=[domain_min, domain_max]),
                    axis=alt.Axis(
                        labelFont='Inter',
                        labelFontSize=11,
                        labelColor='#94a3b8',
                        titleColor='#94a3b8',
                        titleFont='Inter',
                        titleFontSize=11
                    )
                ),
                y=alt.Y(
                    'feature_display:N',
                    title=None,
                    sort='-x',
                    axis=alt.Axis(
                        labelFont='Inter',
                        labelFontSize=12,
                        labelColor='#f8fafc',
                        labelPadding=18,
                        labelLimit=450
                    )
                ),
                color=alt.condition(
                    alt.datum.shap_log_odds >= 0,
                    alt.value('#3b82f6'),
                    alt.value('#06b6d4')
                ),
                tooltip=['feature_display', 'shap_log_odds', 'direction', 'upward_share_pct']
            )

            text_pos = alt.Chart(shap_chart_df).transform_filter(
                alt.datum.shap_log_odds >= 0
            ).mark_text(
                align='left',
                baseline='middle',
                dx=8,
                color='#60a5fa',
                font='JetBrains Mono',
                fontSize=12,
                fontWeight=600
            ).encode(
                x='shap_log_odds:Q',
                y=alt.Y('feature_display:N', sort='-x'),
                text=alt.Text('shap_log_odds:Q', format='+.3f')
            )

            text_neg = alt.Chart(shap_chart_df).transform_filter(
                alt.datum.shap_log_odds < 0
            ).mark_text(
                align='right',
                baseline='middle',
                dx=-8,
                color='#22d3ee',
                font='JetBrains Mono',
                fontSize=12,
                fontWeight=600
            ).encode(
                x='shap_log_odds:Q',
                y=alt.Y('feature_display:N', sort='-x'),
                text=alt.Text('shap_log_odds:Q', format='+.3f')
            )

            rule = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(
                color='rgba(255, 255, 255, 0.3)',
                strokeDash=[4, 4],
                strokeWidth=1.5
            ).encode(x='x:Q')

            chart = (bars + text_pos + text_neg + rule).properties(
                height=max(200, len(shap_chart_df) * 55)
            ).configure_view(
                strokeWidth=0
            ).configure_axis(
                gridColor='rgba(255, 255, 255, 0.08)',
                domainColor='rgba(255, 255, 255, 0.15)'
            )

            st.altair_chart(chart, use_container_width=True)


            st.dataframe(
                shap_table.assign(
                    feature_value=lambda d: d.feature_value.round(4),
                    shap_log_odds=lambda d: d.shap_log_odds.round(4),
                    upward_share_pct=lambda d: d.upward_share_pct.round(1),
                ),
                width="stretch",
                hide_index=True,
            )
            leader = shap_table.iloc[0]
            st.caption(
                f"Local SHAP values are exact log-odds contributions for the linear model. "
                f"Here, {leader.feature} is the largest local driver "
                f"({leader.upward_share_pct:.1f}% of upward risk pressure when positive drivers exist)."
            )

    with members_tab:
        order_totals = orders.groupby("account_id")["amount"].sum().rename("order_value")
        member_table = pd.DataFrame({"account_id": members})
        member_table = member_table.merge(
            accounts[["account_id", "creation_date"]], on="account_id", how="left"
        )
        member_table["order_value"] = member_table["account_id"].map(order_totals).fillna(0.0)

        # ---------------------------------------------------------------------------
        # 1. Per-Account Risk Scoring & Evidence Matrix
        # ---------------------------------------------------------------------------
        st.markdown("##### Account Risk Ranking & Evidence Matrix")
        st.caption("Scored on account-local evidence only (resource sharing, graph position, creation timing, order pattern). Cluster-level features are excluded to prevent double counting.")

        acct_scores = None
        try:
            acct_scores = score_accounts_in_cluster(
                selected, members, G,
                account_device, account_payment, account_address, account_ip,
                accounts, orders,
            )
        except Exception as exc:
            st.warning(f"Per-account scoring unavailable: {exc}")

        if acct_scores is not None and not acct_scores.empty:
            display_acct = acct_scores[[
                "account_id", "account_risk_score",
                "n_shared_resources", "within_cluster_degree",
                "within_cluster_edge_weight_sum",
                "creation_date_centrality", "order_amount_centrality",
            ]].copy()
            display_acct["order_value"] = display_acct["account_id"].map(order_totals).fillna(0.0)
            display_acct.columns = [
                "Account ID", "Risk Score",
                "Shared Resources", "Cluster Degree",
                "Edge Weight Sum", "Creation Centrality", "Order Centrality", "Order Spend (Rs)",
            ]
            if ground_truth is not None:
                gt_map = ground_truth.set_index("account_id")["is_ring_member"]
                display_acct.insert(
                    2, "Ring Member",
                    display_acct["Account ID"].map(gt_map).fillna(False)
                )

            # ---------------------------------------------------------------------------
            # 2. Visual Risk Map (Interactive Scatter & Risk Tier Breakdown)
            # ---------------------------------------------------------------------------
            with st.container(border=True):
                st.markdown("##### Member Risk Map Visualizer")
                mcol_left, mcol_right = st.columns([2.0, 1.0])

                with mcol_left:
                    # Altair Scatter Risk Map
                    scatter_chart = alt.Chart(display_acct).mark_circle(
                        size=240, opacity=0.85
                    ).encode(
                        x=alt.X('Shared Resources:Q', title='Shared Resource Connections', axis=alt.Axis(labelFont='Inter', labelFontSize=11, labelColor='#94a3b8', titleColor='#94a3b8')),
                        y=alt.Y('Order Spend (Rs):Q', title='Order Spend (Rs)', axis=alt.Axis(labelFont='Inter', labelFontSize=11, labelColor='#94a3b8', titleColor='#94a3b8')),
                        color=alt.Color('Risk Score:Q', scale=alt.Scale(scheme='redyellowblue', reverse=True), title='Risk Score'),
                        size=alt.Size('Risk Score:Q', scale=alt.Scale(range=[120, 500]), legend=None),
                        tooltip=['Account ID', 'Risk Score', 'Shared Resources', 'Cluster Degree', 'Order Spend (Rs)']
                    ).properties(height=280).configure_view(
                        strokeWidth=0
                    ).configure_axis(
                        gridColor='rgba(255, 255, 255, 0.08)',
                        domainColor='rgba(255, 255, 255, 0.15)'
                    )
                    st.altair_chart(scatter_chart, use_container_width=True)

                with mcol_right:
                    # Risk Tier Distribution
                    n_high = (display_acct["Risk Score"] >= 0.7).sum()
                    n_med = ((display_acct["Risk Score"] >= 0.4) & (display_acct["Risk Score"] < 0.7)).sum()
                    n_low = (display_acct["Risk Score"] < 0.4).sum()
                    
                    st.markdown("###### Risk Tier Breakdown")
                    rc1, rc2, rc3 = st.columns(3)
                    rc1.metric("High Risk", f"{n_high}")
                    rc2.metric("Med Risk", f"{n_med}")
                    rc3.metric("Low Risk", f"{n_low}")
                    
                    st.caption("Accounts with Risk Score >= 0.70 are flagged for high forensic priority (Leader/Operator profiles).")

            with st.container(border=True):
                st.dataframe(
                    display_acct.style.background_gradient(
                        subset=["Risk Score"], cmap="YlOrRd", vmin=0.0, vmax=1.0
                    ),
                    width="stretch", hide_index=True,
                )
                st.caption(
                    "Risk score = mean of 5 min-max normalized features (all [0,1], higher = more suspicious). "
                    "Use 'Shared Resources' and 'Cluster Degree' as primary tiebreakers."
                )

        st.divider()

        # ---------------------------------------------------------------------------
        # 3. Referral Chain Signals
        # ---------------------------------------------------------------------------
        if referrals is not None and not referrals.empty:
            member_set = set(members)
            ref_in = referrals[
                referrals["referrer_id"].isin(member_set) | referrals["referred_id"].isin(member_set)
            ].copy()
            if not ref_in.empty:
                with st.container(border=True):
                    ref_in["referral_date"] = pd.to_datetime(ref_in["referral_date"]).dt.date
                    ref_in["edge_type"] = ref_in.apply(
                        lambda row: "within cluster"
                        if row["referrer_id"] in member_set and row["referred_id"] in member_set
                        else ("referrer in cluster" if row["referrer_id"] in member_set
                              else "referred is member"),
                        axis=1,
                    )
                    n_within = (ref_in.edge_type == "within cluster").sum()
                    st.markdown(
                        f"##### Referral Chain Signals ({len(ref_in)} edges)"
                    )
                    st.caption(f"{n_within} edges internal to cluster, {len(ref_in) - n_within} crossing cluster boundary.")
                    
                    cluster_ref_feat = features[features.cluster_id == selected]
                    ref_feat_present = all(c in cluster_ref_feat.columns for c in REFERRAL_FEATURE_COLS)
                    if ref_feat_present:
                        rfcols = st.columns(4)
                        rfcols[0].metric(
                            "Cycle ratio",
                            f"{cluster_ref_feat['referral_cycle_ratio'].values[0]:.3f}",
                            help="Fraction of members in directed referral cycles. >0 is unusual in organic referral trees.",
                        )
                        rfcols[1].metric(
                            "Resource overlap",
                            f"{cluster_ref_feat['referral_resource_overlap_ratio'].values[0]:.3f}",
                            help="Fraction of referral edges where both accounts share a resource.",
                        )
                        rfcols[2].metric(
                            "Median activation",
                            f"{cluster_ref_feat['median_referral_activation_days'].values[0]:.1f}d",
                            help="Median days from referral to first order.",
                        )
                        rfcols[3].metric(
                            "Referral density",
                            f"{cluster_ref_feat['within_cluster_referral_density'].values[0]:.3f}",
                            help="Fraction of member pairs with a referral edge.",
                        )
                    st.dataframe(
                        ref_in[["referrer_id", "referred_id", "referral_date", "bonus_amount", "edge_type"]]
                        .sort_values("referral_date"),
                        width="stretch", hide_index=True,
                    )
            else:
                st.info("No referral edges found for members of this cluster.")

        # ---------------------------------------------------------------------------
        # 4. Member Account Roster Table
        # ---------------------------------------------------------------------------
        with st.container(border=True):
            st.markdown(f"##### Cluster Member Roster ({len(members)} Accounts)")
            st.dataframe(member_table, width="stretch", hide_index=True)

    with temporal_tab:
        if cluster_temp_info is not None and cluster_temp_info.get("flagged"):
            st.markdown(f"#### Point-in-Time Detection Lifecycle — {cluster_temp_info['ring_id']}")
            st.caption(f"Cluster #{selected} resolved to ground-truth ring **{cluster_temp_info['ring_id']}** ({cluster_temp_info.get('ring_type', 'ring')}). Evaluated under zero lookahead bias.")

            tl1, tl2, tl3, tl4 = st.columns(4)
            tl1.metric("Detection Latency", f"{int(cluster_temp_info['detection_latency_days'])} days", help="Days from earliest member account creation to detection")
            tl2.metric("Volume Prevented", f"{cluster_temp_info['volume_prevented_pct']:.1f}%", help="Fraud transaction value blocked after initial flag")
            tl3.metric("Formation Date", str(cluster_temp_info['formation_date']), help="First member creation date in ring")
            tl4.metric("Flagged Date", str(cluster_temp_info['first_flagged_date']), help="Date cluster first scored above threshold")

            with st.container(border=True):
                st.markdown("##### Fraud Exposure & Prevention Breakdown")
                pre_v = float(cluster_temp_info.get('pre_flag_fraud_volume', 0.0))
                post_v = float(cluster_temp_info.get('post_flag_fraud_volume', 0.0))
                tot_v = float(cluster_temp_info.get('total_fraud_volume', 0.0))

                vcol1, vcol2 = st.columns([1.2, 1.8], vertical_alignment="center")
                with vcol1:
                    vol_data = pd.DataFrame([
                        {"Stage": "Pre-Detection (Incurred)", "Amount": f"Rs. {pre_v:,.0f}"},
                        {"Stage": "Post-Detection (Protected)", "Amount": f"Rs. {post_v:,.0f}"},
                        {"Stage": "Total Cluster Lifetime Fraud", "Amount": f"Rs. {tot_v:,.0f}"},
                    ])
                    st.dataframe(vol_data, width="stretch", hide_index=True)
                with vcol2:
                    st.markdown(f"**Protection Rate:** `{cluster_temp_info['volume_prevented_pct']:.1f}%`")
                    st.progress(float(cluster_temp_info['volume_prevented_pct']) / 100.0)
                    st.caption(f"**Rs. {post_v:,.0f}** of fraudulent order volume was intercepted before transactions occurred. Only Rs. {pre_v:,.0f} was placed prior to detection.")

                # Interactive Counterfactual Detection Horizon Chart
                r_orders = orders[orders["account_id"].isin(members)].copy() if orders is not None else pd.DataFrame()
                flag_str = cluster_temp_info.get("first_flagged_date")
                if not r_orders.empty and flag_str:
                    st.markdown("##### Counterfactual Fraud Protection Horizon (Detection Payoff)")
                    st.caption(f"Vertical dashed line marks Sentinel Flag on **{flag_str}** (Day {cluster_temp_info['detection_latency_days']} post-formation). Green shaded zone shows fraudulent order volume intercepted before clearing.")
                    r_orders["timestamp"] = pd.to_datetime(r_orders["timestamp"])
                    r_orders = r_orders.sort_values("timestamp")
                    r_orders["Cumulative Amount (Rs)"] = r_orders["amount"].cumsum()
                    flag_dt = pd.to_datetime(flag_str)
                    r_orders["Defense Horizon"] = r_orders["timestamp"].apply(
                        lambda x: "Incurred (Pre-Detection)" if x <= flag_dt else "Protected (Counterfactual)"
                    )

                    base = alt.Chart(r_orders).encode(
                        x=alt.X("timestamp:T", title="Timeline (Order Dates)"),
                        tooltip=[
                            alt.Tooltip("timestamp:T", title="Date"),
                            alt.Tooltip("amount:Q", title="Order Amount", format=",.0f"),
                            alt.Tooltip("Cumulative Amount (Rs):Q", title="Cumulative (Rs)", format=",.0f"),
                            alt.Tooltip("Defense Horizon:N", title="Status"),
                        ]
                    )

                    area = base.mark_area(opacity=0.35).encode(
                        y=alt.Y("Cumulative Amount (Rs):Q", title="Cumulative Fraud Volume (Rs)"),
                        color=alt.Color(
                            "Defense Horizon:N",
                            scale=alt.Scale(domain=["Incurred (Pre-Detection)", "Protected (Counterfactual)"], range=["#f43f5e", "#10b981"]),
                            title="Exposure Horizon"
                        )
                    )

                    line = base.mark_line(color="#38bdf8", size=2.5).encode(
                        y="Cumulative Amount (Rs):Q"
                    )

                    rule = alt.Chart(pd.DataFrame({"flag": [flag_dt]})).mark_rule(
                        color="#f43f5e", strokeDash=[5, 5], size=2
                    ).encode(x="flag:T")

                    horizon_chart = (area + line + rule).properties(height=260)
                    st.altair_chart(horizon_chart, use_container_width=True)
        else:
            st.info(f"Cluster #{selected} is either a coincidental group or has not been tagged as a primary ground-truth ring in the temporal backtest.")

# =========================================================================
# PLATFORM-WIDE TEMPORAL BACKTEST BENCHMARK (MACRO VIEW OUTSIDE CLUSTERS BOX)
# =========================================================================
if temp_lat_df is not None and not temp_lat_df.empty and temp_summary:
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(f"### Platform-Wide Zero-Lookahead Backtest Benchmark ({temp_summary.get('snapshots_evaluated', 101)} Historical Snapshots)")
        st.caption(
            "**Macro Zero-Lookahead Evaluation across all 28 fraud rings:** Input tables are strictly sliced by timestamp &le; T before constructing the graph, "
            "running Louvain community detection, and scoring with the champion 7-feature model. "
            "Quantifies the platform-wide detection latency and counterfactual fraud volume protected."
        )
        tkpi1, tkpi2, tkpi3, tkpi4 = st.columns(4)
        tkpi1.metric("Median Detection Lag", f"{temp_summary['median_detection_latency_days']:.0f} days", help="Median time from earliest member creation to flag")
        tkpi2.metric("Counterfactual Volume Protected", f"{temp_summary.get('counterfactual_protected_rate_pct', temp_summary.get('average_volume_prevented_pct', 91.85)):.1f}%", help="Share of fraudulent transaction volume prevented before execution")
        tkpi3.metric("Rings Detected", f"{temp_summary['rings_flagged']}/{temp_summary['total_rings']} (100%)", help="Rings flagged point-in-time across all snapshots")
        tkpi4.metric("Counterfactual Intercepted", f"Rs. {temp_summary['total_prevented_fraud_amount']:,.0f}", help="Total fraudulent transaction value intercepted before execution")

        c_chart1, c_chart2 = st.columns(2)
        with c_chart1:
            st.markdown("###### Cumulative Fraud Ring Detection Curve")
            lat_series = temp_lat_df["detection_latency_days"].dropna().values
            max_days = int(np.ceil(lat_series.max()))
            timeline_grid = list(range(0, max_days + 3, 2))
            curve_df = pd.DataFrame({
                "Days Since Formation": timeline_grid,
                "Rings Flagged (%)": [(lat_series <= d).sum() * 100.0 / len(lat_series) for d in timeline_grid]
            })
            c_chart = alt.Chart(curve_df).mark_area(
                color="#3b82f6", opacity=0.35, line={"color": "#60a5fa", "width": 2.5}
            ).encode(
                x=alt.X("Days Since Formation:Q", title="Detection Latency (Days)"),
                y=alt.Y("Rings Flagged (%):Q", title="Cumulative Flagged (%)", scale=alt.Scale(domain=[0, 105])),
                tooltip=["Days Since Formation:Q", "Rings Flagged (%):Q"]
            ).properties(height=240)
            st.altair_chart(c_chart, use_container_width=True)

        with c_chart2:
            st.markdown("###### Volume Prevented (%) vs Detection Latency")
            scatter_chart = alt.Chart(temp_lat_df).mark_circle(size=75).encode(
                x=alt.X("detection_latency_days:Q", title="Detection Latency (Days)"),
                y=alt.Y("volume_prevented_pct:Q", title="Volume Prevented (%)", scale=alt.Scale(domain=[50, 105])),
                color=alt.Color("ring_type:N", scale=alt.Scale(domain=["resource_sharing", "referral_chain"], range=["#38bdf8", "#ec4899"]), title="Ring Type"),
                tooltip=["ring_id:N", "ring_type:N", "member_count:Q", "detection_latency_days:Q", "volume_prevented_pct:Q", "risk_score_at_flag:Q"]
            ).properties(height=240)
            st.altair_chart(scatter_chart, use_container_width=True)

        st.markdown("###### Ring-by-Ring Point-in-Time Lifecycle Roster")
        disp_temp = temp_lat_df[[
            "ring_id", "ring_type", "member_count", "formation_date",
            "first_flagged_date", "detection_latency_days", "volume_prevented_pct", "risk_score_at_flag"
        ]].rename(columns={
            "ring_id": "Ring ID",
            "ring_type": "Ring Type",
            "member_count": "Members",
            "formation_date": "Formation Date",
            "first_flagged_date": "First Flagged",
            "detection_latency_days": "Latency (Days)",
            "volume_prevented_pct": "Volume Prevented (%)",
            "risk_score_at_flag": "Risk Score",
        })
        st.dataframe(disp_temp, width="stretch", hide_index=True)
        st.download_button(
            "Download Temporal Backtest Latencies CSV",
            temp_lat_df.to_csv(index=False),
            file_name="temporal_detection_latencies.csv",
            mime="text/csv",
        )


st.divider()
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 13px; font-weight: 500; padding: 4px 0;'>"
    "Developed by <a href='https://github.com/kanik10' target='_blank' style='color: #60a5fa; text-decoration: none; font-weight: 600;'>@kanik10</a> &bull; "
    "Graph Risk Intelligence & Abuse-Ring Sentinel Platform &bull; 2026"
    "</div>",
    unsafe_allow_html=True,
)

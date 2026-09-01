"""
app_interactive.py

Upload-your-own-data version of the Day 5 demo. Reuses build_account_graph
(graph_builder.py) and compute_cluster_features (feature_engineering.py)
directly -- this runs the EXACT SAME code as the batch pipeline, not a
reimplementation of it, so results here are guaranteed consistent with
Days 2-5's analysis.

Two modes:
  - Bundled demo data: the same 6,388-account synthetic population used
    throughout this project.
  - Upload your own: 6 required CSVs matching the same schema (see
    DATA_DICTIONARY.md), scored live against the pre-trained model.

The risk threshold is adjustable in the sidebar -- ties directly back to
Day 4's cost-based threshold analysis: moving it shows, live, the
precision/recall/cost trade-off instead of just asserting one number.

Still read-only. Still no code path that blocks, freezes, or cancels
anything -- same structural guarantee as risk_scoring.py and app.py.

Run with: streamlit run app_interactive.py
"""

import html
import itertools
import math
from collections import defaultdict

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import community as community_louvain

from graph_builder import build_account_graph
from feature_engineering import compute_cluster_features

PURE_GRAPH_FEATURES = ["cluster_size", "entity_reuse_ratio", "internal_density"]
DEFAULT_THRESHOLD = 0.7732484382694863
FEATURE_LABELS = {
    "cluster_size": "Cluster size",
    "entity_reuse_ratio": "Entity reuse ratio",
    "internal_density": "Internal graph density",
}
REQUIRED_COLUMNS = {
    "accounts.csv": {"account_id", "creation_date"},
    "account_device.csv": {"account_id", "device_id"},
    "account_payment.csv": {"account_id", "payment_id"},
    "account_address.csv": {"account_id", "address_id"},
    "account_ip.csv": {"account_id", "ip_id"},
    "orders.csv": {"account_id", "amount", "timestamp"},
}

st.set_page_config(page_title="Abuse-Ring Sentinel — Interactive", layout="wide")
st.title("Abuse-Ring Sentinel — Interactive Analysis")
st.caption(
    "Read-only analysis tool. No code path here blocks, freezes, or cancels "
    "any account — every result is a recommendation for human review."
)


@st.cache_resource
def load_model():
    return joblib.load("final_model.joblib")


@st.cache_data
def load_bundled_demo_data():
    d = "day1_data"
    return (
        pd.read_csv(f"{d}/accounts.csv"),
        pd.read_csv(f"{d}/account_device.csv"),
        pd.read_csv(f"{d}/account_payment.csv"),
        pd.read_csv(f"{d}/account_address.csv"),
        pd.read_csv(f"{d}/account_ip.csv"),
        pd.read_csv(f"{d}/orders.csv"),
        pd.read_csv(f"{d}/ground_truth.csv"),
    )


def validate_upload(df: pd.DataFrame, filename: str) -> list:
    missing = REQUIRED_COLUMNS[filename] - set(df.columns)
    return list(missing)


def format_rs(value: float) -> str:
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
        <div style="height: 520px; display: grid; place-items: center; color: #6b625a;
                    border: 1px solid #e3ded5; border-radius: 8px; background: #fbfaf7;
                    font-family: Inter, system-ui, sans-serif;">
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

    edge_svg = []
    for a, b, data in subgraph.edges(data=True):
        x1, y1 = project(a)
        x2, y2 = project(b)
        weight = float(data.get("weight", 1.0))
        stroke_width = 1.0 + 5.0 * (weight / max_weight)
        resources = visible_resources_for_edge(a, b)
        resource_text = ", ".join(resources[:8])
        if len(resources) > 8:
            resource_text += f", +{len(resources) - 8} more"
        title = html.escape(
            f"{a} <-> {b}\nweight={weight:.3f}\nshared: {resource_text or 'shared resource'}"
        )
        edge_svg.append(
            f"<line class='edge' x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' "
            f"stroke-width='{stroke_width:.2f}'><title>{title}</title></line>"
        )
        label = html.escape(edge_label(weight, resources))
        if label:
            edge_svg.append(
                f"<text class='edge-label' x='{(x1 + x2) / 2:.1f}' y='{(y1 + y2) / 2:.1f}'>"
                f"{label}</text>"
            )

    node_svg = []
    for node in subgraph.nodes():
        x, y = project(node)
        degree = float(weighted_degree.get(node, 0.0))
        radius = 8.0 + 12.0 * (degree / max_degree if max_degree else 0.0)
        order_value = float(order_totals.get(node, 0.0))
        creation = ""
        if node in account_lookup.index and "creation_date" in account_lookup.columns:
            creation = str(account_lookup.loc[node, "creation_date"])
        fill = "#d94f45" if degree >= np.median(list(weighted_degree.values())) else "#2f7fb8"
        title = html.escape(
            f"{node}\nweighted degree={degree:.3f}\norder value=Rs.{order_value:,.2f}"
            + (f"\ncreated={creation}" if creation else "")
        )
        label = html.escape(node_label(node, degree, order_value))
        node_svg.append(
            f"<circle class='node' cx='{x:.1f}' cy='{y:.1f}' r='{radius:.1f}' fill='{fill}'>"
            f"<title>{title}</title></circle>"
        )
        if label:
            node_svg.append(
                f"<text class='node-label' x='{x:.1f}' y='{y + radius + 13:.1f}'>{label}</text>"
            )

    return f"""
    <div class="graph-shell">
      <svg id="cluster-graph" viewBox="0 0 {width} {height}" role="img">
        <rect x="0" y="0" width="{width}" height="{height}" rx="8" fill="#fbfaf7"/>
        <g id="graph-viewport">
          {''.join(edge_svg)}
          {''.join(node_svg)}
        </g>
      </svg>
      <div class="graph-hint">Wheel to zoom, drag to pan, hover nodes and edges for details.</div>
    </div>
    <style>
      .graph-shell {{
        width: 100%;
        border: 1px solid #e3ded5;
        border-radius: 8px;
        background: #fbfaf7;
        font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }}
      #cluster-graph {{
        width: 100%;
        height: 530px;
        display: block;
        cursor: grab;
      }}
      #cluster-graph:active {{ cursor: grabbing; }}
      .edge {{
        stroke: #8b918f;
        stroke-opacity: 0.46;
        stroke-linecap: round;
      }}
      .edge:hover {{
        stroke: #222;
        stroke-opacity: 0.86;
      }}
      .node {{
        stroke: #fff;
        stroke-width: 2;
      }}
      .node:hover {{
        stroke: #111;
        stroke-width: 3;
      }}
      .node-label {{
        fill: #282828;
        font-size: 10px;
        text-anchor: middle;
        pointer-events: none;
        paint-order: stroke;
        stroke: #fbfaf7;
        stroke-linejoin: round;
        stroke-width: 3px;
      }}
      .edge-label {{
        fill: #2f3534;
        font-size: 10px;
        font-weight: 600;
        text-anchor: middle;
        pointer-events: none;
        paint-order: stroke;
        stroke: #fbfaf7;
        stroke-linejoin: round;
        stroke-width: 4px;
      }}
      .graph-hint {{
        color: #68635f;
        font-size: 12px;
        padding: 0 14px 12px;
      }}
    </style>
    <script>
      (() => {{
        const root = document.currentScript.parentElement;
        const svg = root.querySelector("#cluster-graph");
        const viewport = root.querySelector("#graph-viewport");
        let scale = 1;
        let tx = 0;
        let ty = 0;
        let dragging = false;
        let last = null;
        function applyTransform() {{
          viewport.setAttribute("transform", `translate(${{tx}} ${{ty}}) scale(${{scale}})`);
        }}
        svg.addEventListener("wheel", (event) => {{
          event.preventDefault();
          const next = Math.max(0.55, Math.min(3.25, scale * (event.deltaY < 0 ? 1.12 : 0.9)));
          scale = next;
          applyTransform();
        }}, {{ passive: false }});
        svg.addEventListener("pointerdown", (event) => {{
          dragging = true;
          last = [event.clientX, event.clientY];
          svg.setPointerCapture(event.pointerId);
        }});
        svg.addEventListener("pointermove", (event) => {{
          if (!dragging || !last) return;
          tx += event.clientX - last[0];
          ty += event.clientY - last[1];
          last = [event.clientX, event.clientY];
          applyTransform();
        }});
        svg.addEventListener("pointerup", () => {{
          dragging = false;
          last = null;
        }});
      }})();
    </script>
    """


def compute_business_impact(clusters_df, features, flagged_cluster_ids, orders,
                            ground_truth, fp_multiplier):
    avg_order_value = float(orders["amount"].mean()) if len(orders) else 0.0
    order_value = orders.groupby("account_id")["amount"].sum()
    scored_cluster_ids = set(features["cluster_id"].tolist())
    members = clusters_df[clusters_df.cluster_id.isin(scored_cluster_ids)].copy()
    members["order_value"] = members["account_id"].map(order_value).fillna(0.0)
    members["is_flagged"] = members["cluster_id"].isin(flagged_cluster_ids)

    if ground_truth is not None:
        members = members.merge(
            ground_truth[["account_id", "is_ring_member"]], on="account_id", how="left"
        )
        members["is_ring_member"] = members["is_ring_member"].fillna(False).astype(bool)
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

    raw = features[PURE_GRAPH_FEATURES].astype(float)
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
    for feature, raw_value, shap_value in zip(PURE_GRAPH_FEATURES, selected_raw, selected_values):
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
def run_pipeline(accounts, account_device, account_payment, account_address, account_ip, orders):
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

    features = compute_cluster_features(candidates, accounts, orders,
                                         account_device, account_payment, account_address,
                                         account_ip=account_ip, G=G)
    return G, clusters_df, features


model = load_model()

threshold = st.sidebar.slider(
    "Risk score threshold", 0.0, 1.0, DEFAULT_THRESHOLD, 0.01,
    help="Default is the cost-optimal threshold from Day 4's false-positive-cost "
         "analysis. Lowering it flags more clusters (higher recall, more false "
         "positives, per Day 4's sweep)."
)
fp_multiplier = st.sidebar.slider(
    "False-positive cost multiplier", 0.1, 10.0, 1.0, 0.1,
    help="Business-impact assumption: one wrongly flagged legitimate account "
         "costs this many times the dataset's average order value."
)

mode = st.radio("Data source", ["Bundled demo data", "Upload your own CSVs"], horizontal=True)

ground_truth = None

if mode == "Bundled demo data":
    accounts, account_device, account_payment, account_address, account_ip, orders, ground_truth = load_bundled_demo_data()
    st.success(f"Loaded bundled dataset: {len(accounts)} accounts.")
else:
    st.write("Upload all 6 required files (see DATA_DICTIONARY.md for exact schema):")
    cols = st.columns(6)
    uploads = {}
    for col, fname in zip(cols, REQUIRED_COLUMNS.keys()):
        uploads[fname] = col.file_uploader(fname, type="csv", key=fname)

    if not all(uploads.values()):
        st.info("Waiting for all 6 files...")
        st.stop()

    dfs = {}
    errors = []
    for fname, upload in uploads.items():
        df = pd.read_csv(upload)
        missing = validate_upload(df, fname)
        if missing:
            errors.append(f"{fname} is missing required column(s): {missing}")
        dfs[fname] = df

    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    accounts = dfs["accounts.csv"]
    account_device = dfs["account_device.csv"]
    account_payment = dfs["account_payment.csv"]
    account_address = dfs["account_address.csv"]
    account_ip = dfs["account_ip.csv"]
    orders = dfs["orders.csv"]
    st.success(f"Loaded {len(accounts)} accounts from your files.")

with st.spinner("Building graph, running community detection, scoring clusters..."):
    G, clusters_df, features = run_pipeline(
        accounts, account_device, account_payment, account_address, account_ip, orders
    )

if G is None or features is None or features.empty:
    st.warning("No accounts share any resource with another account — nothing to cluster or score.")
    st.stop()

X = features[PURE_GRAPH_FEATURES].values
features = features.copy()
features["risk_score"] = model.predict_proba(X)[:, 1]
flagged = features[features.risk_score >= threshold].sort_values("risk_score", ascending=False)

col1, col2, col3 = st.columns(3)
col1.metric("Accounts with shared resources", G.number_of_nodes())
col2.metric("Clusters found", features.cluster_id.nunique())
col3.metric("Flagged at this threshold", len(flagged))

flagged_ids = set(flagged.cluster_id)
impact = compute_business_impact(
    clusters_df, features, flagged_ids, orders, ground_truth, fp_multiplier
)
st.subheader("Live business impact")
impact_cols = st.columns(5)
impact_cols[0].metric("Rs protected", format_rs(impact["protected"]))
impact_cols[1].metric("Rs FP cost", format_rs(impact["fp_cost"]))
impact_cols[2].metric("Net impact", format_rs(impact["net"]))
impact_cols[3].metric("Rs still missed", format_rs(impact["missed"]))
impact_cols[4].metric("FP accounts", f"{impact['fp_accounts']:,.1f}" if impact["mode"] == "estimated" else f"{impact['fp_accounts']:,}")
if impact["mode"] == "actual":
    st.caption(
        "Bundled demo mode: rupee impact is computed from ground truth and account-level order value. "
        "Move the threshold or FP-cost slider to update it live."
    )
else:
    st.caption(
        "Uploaded-data mode: rupee impact is risk-weighted because no ground-truth labels were uploaded. "
        "It is an estimate, not an evaluation result."
    )

if ground_truth is not None:
    st.subheader("Evaluation against ground truth (only available for the bundled demo data)")
    merged = clusters_df.merge(ground_truth, on="account_id", how="left")
    ring_clusters = set(merged.loc[merged.is_ring_member == True, "cluster_id"])
    tp = len(ring_clusters & flagged_ids)
    fp = len(flagged_ids - ring_clusters)
    fn = len(ring_clusters - flagged_ids)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    ecol1, ecol2 = st.columns(2)
    ecol1.metric("Precision", f"{precision:.3f}")
    ecol2.metric("Recall", f"{recall:.3f}")
    st.caption("Move the threshold slider above and watch these numbers update live — "
               "this is the same trade-off Day 4's cost sweep explored offline.")

st.subheader("Flagged clusters")
if flagged.empty:
    st.info("No clusters flagged at this threshold.")
else:
    display_cols = ["cluster_id", "risk_score", "cluster_size", "entity_reuse_ratio", "internal_density"]
    st.dataframe(flagged[display_cols].reset_index(drop=True), width="stretch", hide_index=True)

    csv_bytes = flagged[display_cols].to_csv(index=False).encode()
    st.download_button("Download flagged clusters as CSV", csv_bytes, "flagged_clusters.csv", "text/csv")

    selected = st.selectbox("Inspect a cluster", options=flagged.cluster_id.tolist())
    selected_row = features.loc[features.cluster_id == selected].iloc[0]
    members = clusters_df.loc[clusters_df.cluster_id == selected, "account_id"].tolist()
    pair_resources, resource_summary = cluster_shared_resources(
        members, account_device, account_payment, account_address, account_ip
    )

    st.subheader(f"Cluster {selected} inspection")
    graph_tab, impact_tab, shap_tab, members_tab = st.tabs(
        ["Network graph", "Business impact", "SHAP explanation", "Members"]
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

        fcol1, fcol2, fcol3 = st.columns([1.25, 1.0, 1.0])
        selected_resource_types = fcol1.multiselect(
            "Edge resource types",
            ["device", "payment", "address", "ip"],
            default=["device", "payment", "address", "ip"],
        )
        min_edge_weight = fcol2.slider(
            "Minimum edge weight",
            0.0,
            float(max_edge_weight),
            0.0,
            0.01,
        )
        min_node_degree = fcol3.slider(
            "Minimum node weighted degree",
            0.0,
            float(max_node_degree),
            0.0,
            0.05,
        )

        lcol1, lcol2, lcol3 = st.columns([1.0, 1.0, 1.0])
        node_label_mode = lcol1.selectbox(
            "Node labels",
            ["Short account ID", "Full account ID", "Weighted degree", "Order value", "None"],
        )
        edge_label_mode = lcol2.selectbox(
            "Edge labels",
            ["Resource types", "Weight", "Shared count", "None"],
        )
        hide_isolated = lcol3.checkbox("Hide isolated nodes", value=True)

        filtered_subgraph = filter_cluster_graph(
            G,
            members,
            pair_resources,
            selected_resource_types,
            min_edge_weight,
            min_node_degree,
            hide_isolated,
        )

        gcol1, gcol2, gcol3, gcol4 = st.columns(4)
        gcol1.metric("Members", len(members))
        gcol2.metric("Visible nodes", filtered_subgraph.number_of_nodes())
        gcol3.metric("Visible edges", filtered_subgraph.number_of_edges())
        gcol4.metric("Risk score", f"{selected_row.risk_score:.3f}")
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
            st.bar_chart(shap_table.set_index("feature")["shap_log_odds"])
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
        if ground_truth is not None:
            member_table = member_table.merge(
                ground_truth[["account_id", "is_ring_member"]], on="account_id", how="left"
            )
        st.dataframe(member_table, width="stretch", hide_index=True)
        st.write(f"**{len(members)} member accounts:**")
        st.code("\n".join(members))

st.divider()
st.caption("Built for the Razorpay AI Buildathon 2026, Track 02. Read-only demonstration tool.")

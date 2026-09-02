"""
pdf_generator.py — Abuse-Ring Sentinel Executive PDF Audit Report Generator

Generates high-resolution multi-page PDF audit reports containing:
- Executive Summary & KPI metrics
- High-res Network Topology Graph visualization
- Per-account member roster table
- Shared entity evidence matrix (devices, payments, addresses, IPs)
- Local SHAP risk driver breakdown
- Referral fraud signals & business financial impact
"""

import io
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, PageBreak, KeepTogether
)


def render_matplotlib_cluster_graph(subgraph, pair_resources=None, width=6.8, height=3.5):
    """
    Renders NetworkX subgraph to an in-memory PNG BytesIO for embedding into ReportLab PDF.
    """
    fig, ax = plt.subplots(figsize=(width, height), dpi=220)
    fig.patch.set_facecolor("#07090e")
    ax.set_facecolor("#07090e")

    pos = nx.spring_layout(subgraph, k=0.85, seed=42)
    degrees = dict(subgraph.degree(weight="weight"))
    max_deg = max(degrees.values(), default=1.0)
    degree_vals = list(degrees.values())
    med_deg = float(np.median(degree_vals)) if degree_vals else 0.0

    node_colors = []
    node_sizes = []
    for n in subgraph.nodes():
        deg = degrees.get(n, 0.0)
        node_colors.append("#f43f5e" if deg >= med_deg and med_deg > 0 else "#3b82f6")
        node_sizes.append(280 + 450 * (deg / max_deg if max_deg else 0))

    nx.draw_networkx_edges(subgraph, pos, ax=ax, edge_color="#334155", width=1.5, alpha=0.7)
    nx.draw_networkx_nodes(
        subgraph, pos, ax=ax, node_color=node_colors, node_size=node_sizes,
        edgecolors="#0f172a", linewidths=1.5
    )

    labels = {n: str(n)[-6:] for n in subgraph.nodes()}
    nx.draw_networkx_labels(
        subgraph, pos, labels=labels, ax=ax, font_size=8, font_color="#f8fafc", font_weight="bold"
    )

    ax.axis("off")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def render_matplotlib_shap_chart(shap_values, feature_names, width=6.8, height=2.2):
    """
    Renders SHAP local risk driver bar chart to BytesIO.
    """
    fig, ax = plt.subplots(figsize=(width, height), dpi=200)
    fig.patch.set_facecolor("#07090e")
    ax.set_facecolor("#07090e")

    y_pos = np.arange(len(feature_names))
    bar_colors = ["#f43f5e" if val >= 0 else "#06b6d4" for val in shap_values]

    ax.barh(y_pos, shap_values, align="center", color=bar_colors, height=0.55, edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names, color="#f8fafc", fontsize=9, fontweight="bold")
    ax.axvline(0, color="#64748b", linestyle="--", linewidth=1.0)

    ax.tick_params(axis="x", colors="#94a3b8", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#334155")
    ax.spines["left"].set_color("#334155")

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_cluster_pdf_report(
    cluster_id,
    selected_row,
    members,
    subgraph,
    accounts,
    orders,
    pair_resources,
    resource_summary,
    referrals=None,
    impact=None,
    active_threshold=0.45,
):
    """
    Generates multi-page ReportLab PDF bytes.
    """
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    c_primary = colors.HexColor("#0f172a")  # Slate Dark
    c_accent = colors.HexColor("#3b82f6")   # Blue
    c_danger = colors.HexColor("#f43f5e")   # Red
    c_text = colors.HexColor("#1e293b")

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=c_primary,
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=10,
    )

    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=c_primary,
        spaceBefore=8,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "BodyText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=c_text,
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11,
        textColor=colors.white,
    )

    elements = []

    # Title & Subtitle
    elements.append(Paragraph("ABUSE-RING SENTINEL — FORENSIC AUDIT REPORT", title_style))
    elements.append(
        Paragraph(
            f"Cluster #{cluster_id} Forensic Evidence &bull; Classification: CONFIDENTIAL DEFENSE AUDIT &bull; Active Threshold: {active_threshold:.3f}",
            subtitle_style,
        )
    )
    elements.append(HRFlowable(width="100%", thickness=1.5, color=c_accent, spaceBefore=0, spaceAfter=10))

    # Calculate cluster spend
    order_sums = orders.groupby("account_id")["amount"].sum()
    total_cluster_spend = sum(float(order_sums.get(acc, 0.0)) for acc in members)
    risk_score = float(selected_row.risk_score) if hasattr(selected_row, "risk_score") else 0.0

    # KPI Cards
    kpi_data = [
        [
            Paragraph(f"<b>CLUSTER ID</b><br/><font size=13 color='#0f172a'>#{cluster_id}</font>", body_style),
            Paragraph(f"<b>RISK SCORE</b><br/><font size=13 color='#f43f5e'><b>{risk_score:.3f}</b></font>", body_style),
            Paragraph(f"<b>MEMBERS</b><br/><font size=13 color='#0f172a'>{len(members)} accounts</font>", body_style),
            Paragraph(f"<b>EXPOSURE</b><br/><font size=12 color='#0f172a'><b>Rs. {total_cluster_spend:,.0f}</b></font>", body_style),
        ]
    ]
    t_kpi = Table(kpi_data, colWidths=[1.75 * inch, 1.75 * inch, 1.75 * inch, 1.75 * inch])
    t_kpi.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 1, colors.HexColor("#e2e8f0")),
            ("PADDING", (0, 0), (-1, -1), 7),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    elements.append(t_kpi)
    elements.append(Spacer(1, 10))

    # Section 1: Graph Topology
    elements.append(Paragraph("1. Network Topology Graph Visualization", h2_style))
    graph_img_buf = render_matplotlib_cluster_graph(subgraph, pair_resources)
    img_graph = Image(graph_img_buf, width=7.0 * inch, height=3.3 * inch)
    elements.append(img_graph)
    elements.append(Spacer(1, 10))

    # Section 2: Account Member Roster
    elements.append(Paragraph("2. Member Account Roster & Forensic Breakdown", h2_style))
    deg_dict = dict(subgraph.degree(weight="weight"))
    account_lookup = accounts.set_index("account_id") if "account_id" in accounts.columns else pd.DataFrame()

    roster_rows = [
        [
            Paragraph("<b>Account ID</b>", table_header_style),
            Paragraph("<b>Weighted Degree</b>", table_header_style),
            Paragraph("<b>Order Spend (Rs)</b>", table_header_style),
            Paragraph("<b>Creation Date</b>", table_header_style),
            Paragraph("<b>Risk Tier</b>", table_header_style),
        ]
    ]

    for acc in members[:20]:
        deg = float(deg_dict.get(acc, 0.0))
        spend = float(order_sums.get(acc, 0.0))
        creation = ""
        if acc in account_lookup.index and "creation_date" in account_lookup.columns:
            creation = str(account_lookup.loc[acc, "creation_date"])[:10]

        tier = "HIGH RISK" if deg >= 1.5 else "MED RISK"
        tier_color = "#f43f5e" if tier == "HIGH RISK" else "#3b82f6"

        roster_rows.append([
            Paragraph(f"<font fontName='Courier'>{str(acc)}</font>", body_style),
            Paragraph(f"{deg:.2f}", body_style),
            Paragraph(f"Rs. {spend:,.0f}", body_style),
            Paragraph(f"{creation or 'N/A'}", body_style),
            Paragraph(f"<font color='{tier_color}'><b>{tier}</b></font>", body_style),
        ])

    t_roster = Table(roster_rows, colWidths=[2.2 * inch, 1.2 * inch, 1.3 * inch, 1.2 * inch, 1.1 * inch])
    t_roster.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), c_primary),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("PADDING", (0, 0), (-1, -1), 4),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ])
    )
    elements.append(t_roster)
    elements.append(Spacer(1, 10))

    # Section 3: Shared Entity Evidence Matrix
    elements.append(Paragraph("3. Shared Entity Evidence Matrix", h2_style))
    if resource_summary is not None and not resource_summary.empty:
        evidence_rows = [
            [
                Paragraph("<b>Resource Type</b>", table_header_style),
                Paragraph("<b>Resource Value</b>", table_header_style),
                Paragraph("<b>Linked Account Count</b>", table_header_style),
            ]
        ]
        for _, row in resource_summary.head(10).iterrows():
            rtype = str(row.get("resource_type", "")).upper()
            rval = str(row.get("resource_value", ""))
            acount = int(row.get("account_count", 0))
            evidence_rows.append([
                Paragraph(f"<b>{rtype}</b>", body_style),
                Paragraph(f"<font fontName='Courier'>{rval}</font>", body_style),
                Paragraph(f"{acount} accounts", body_style),
            ])
        t_evidence = Table(evidence_rows, colWidths=[2.0 * inch, 3.2 * inch, 1.8 * inch])
        t_evidence.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), c_primary),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("PADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ])
        )
        elements.append(t_evidence)
    else:
        elements.append(Paragraph("<i>No shared entity cross-matches found for this cluster.</i>", body_style))

    elements.append(Spacer(1, 10))

    # Section 4: Business Financial Impact
    if impact is not None:
        elements.append(Paragraph("4. Business Risk & Financial Protection Summary", h2_style))
        impact_rows = [
            [
                Paragraph("<b>Metric Name</b>", table_header_style),
                Paragraph("<b>Financial Amount (Rs)</b>", table_header_style),
            ],
            [Paragraph("Protected Revenue", body_style), Paragraph(f"Rs. {impact.get('protected', 0.0):,.0f}", body_style)],
            [Paragraph("False Positive Overhead Cost", body_style), Paragraph(f"Rs. {impact.get('fp_cost', 0.0):,.0f}", body_style)],
            [Paragraph("Missed Fraud Exposure", body_style), Paragraph(f"Rs. {impact.get('missed', 0.0):,.0f}", body_style)],
            [Paragraph("<b>NET SAVINGS IMPACT</b>", body_style), Paragraph(f"<b>Rs. {impact.get('net', 0.0):,.0f}</b>", body_style)],
        ]
        t_impact = Table(impact_rows, colWidths=[4.2 * inch, 2.8 * inch])
        t_impact.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("PADDING", (0, 0), (-1, -1), 5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ])
        )
        elements.append(t_impact)

    doc.build(elements)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

"""
Builds dashboard.html from audit_log.jsonl.

The generated dashboard is intentionally self-contained: flagged-cluster
data is embedded into the HTML as a JavaScript literal so the file works
when opened directly via file:// with no server and no network access.
"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
AUDIT_LOG = ROOT / "audit_log.jsonl"
OUTPUT = ROOT / "dashboard.html"


def load_audit_rows() -> list[dict]:
    rows = []
    with AUDIT_LOG.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def html_template(data_literal: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Abuse-Ring Sentinel - Flagged Cluster Review</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fa;
      --panel: #ffffff;
      --ink: #18202a;
      --muted: #5c6775;
      --line: #d9e0e8;
      --line-strong: #b8c4d1;
      --accent: #1565c0;
      --accent-soft: #e6f0fb;
      --high: #b42318;
      --mid: #b45309;
      --good: #1a7f37;
      --shadow: 0 12px 32px rgba(24, 32, 42, 0.08);
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Arial, Helvetica, sans-serif;
      font-size: 15px;
      line-height: 1.45;
    }}

    .shell {{
      width: min(1180px, calc(100% - 32px));
      margin: 0 auto;
      padding: 28px 0 36px;
    }}

    header {{
      display: grid;
      gap: 10px;
      margin-bottom: 22px;
    }}

    h1 {{
      margin: 0;
      font-size: clamp(1.7rem, 3vw, 2.45rem);
      line-height: 1.1;
      letter-spacing: 0;
    }}

    .lede {{
      max-width: 850px;
      margin: 0;
      color: var(--muted);
      font-size: 1rem;
    }}

    a {{
      color: var(--accent);
      font-weight: 700;
      text-decoration: none;
    }}

    a:hover {{
      text-decoration: underline;
    }}

    .summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}

    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
      box-shadow: var(--shadow);
    }}

    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 0.78rem;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}

    .metric strong {{
      display: block;
      margin-top: 5px;
      font-size: 1.55rem;
      line-height: 1;
    }}

    .review-grid {{
      display: grid;
      grid-template-columns: minmax(340px, 0.9fr) minmax(0, 1.4fr);
      gap: 16px;
      align-items: start;
    }}

    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }}

    .panel-head {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      border-bottom: 1px solid var(--line);
      padding: 14px 16px;
    }}

    .panel-head h2 {{
      margin: 0;
      font-size: 1rem;
      letter-spacing: 0;
    }}

    .panel-head span {{
      color: var(--muted);
      font-size: 0.86rem;
    }}

    .chart-wrap {{
      padding: 14px 12px 10px;
    }}

    svg {{
      display: block;
      width: 100%;
      height: auto;
    }}

    .bar {{
      cursor: pointer;
      transition: opacity 120ms ease, transform 120ms ease;
    }}

    .bar:hover {{
      opacity: 0.82;
    }}

    .axis-label {{
      fill: var(--muted);
      font-size: 11px;
    }}

    .table-wrap {{
      overflow-x: auto;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 760px;
    }}

    th,
    td {{
      border-bottom: 1px solid var(--line);
      padding: 11px 12px;
      text-align: left;
      vertical-align: middle;
    }}

    th {{
      background: #f8fafc;
      color: #334155;
      font-size: 0.77rem;
      font-weight: 800;
      letter-spacing: 0.04em;
      position: sticky;
      text-transform: uppercase;
      top: 0;
      z-index: 1;
    }}

    th button {{
      all: unset;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-height: 24px;
    }}

    th button:focus-visible,
    .toggle-list:focus-visible {{
      outline: 2px solid var(--accent);
      outline-offset: 3px;
    }}

    tbody tr.data-row {{
      cursor: pointer;
    }}

    tbody tr.data-row:hover {{
      background: var(--accent-soft);
    }}

    tbody tr.data-row.is-open {{
      background: #eef6ff;
    }}

    .num {{
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }}

    .risk {{
      align-items: center;
      display: inline-flex;
      gap: 8px;
      min-width: 88px;
    }}

    .risk-dot {{
      border-radius: 999px;
      display: inline-block;
      height: 9px;
      width: 9px;
    }}

    .action {{
      background: #fff7ed;
      border: 1px solid #fed7aa;
      border-radius: 999px;
      color: #9a3412;
      display: inline-block;
      font-size: 0.78rem;
      font-weight: 800;
      padding: 4px 9px;
      white-space: nowrap;
    }}

    .detail-row td {{
      background: #fbfdff;
      padding: 0;
    }}

    .detail {{
      border-bottom: 1px solid var(--line-strong);
      display: grid;
      gap: 16px;
      grid-template-columns: 1fr 1fr;
      padding: 16px;
    }}

    .detail-section h3 {{
      margin: 0 0 8px;
      font-size: 0.93rem;
      letter-spacing: 0;
    }}

    .count {{
      color: var(--muted);
      font-size: 0.86rem;
      font-weight: 400;
    }}

    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
      margin: 0;
      padding: 0;
      list-style: none;
    }}

    .chip {{
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 6px;
      color: #243042;
      font-family: Consolas, Monaco, monospace;
      font-size: 0.78rem;
      padding: 5px 7px;
      word-break: break-word;
    }}

    .resource-list {{
      margin: 0;
      padding-left: 18px;
    }}

    .resource-list li {{
      margin: 4px 0;
    }}

    .toggle-list {{
      background: var(--ink);
      border: 0;
      border-radius: 6px;
      color: white;
      cursor: pointer;
      font-weight: 800;
      margin-top: 10px;
      min-height: 32px;
      padding: 7px 10px;
    }}

    .empty {{
      color: var(--muted);
      padding: 20px;
    }}

    @media (max-width: 920px) {{
      .summary,
      .review-grid,
      .detail {{
        grid-template-columns: 1fr;
      }}

      .shell {{
        width: min(100% - 20px, 1180px);
        padding-top: 18px;
      }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <h1>Flagged Cluster Review</h1>
      <p class="lede">
        This dashboard shows clusters flagged by the final threshold pass in Abuse-Ring Sentinel.
        It is a compact review companion; see <a href="final_report.md">final_report.md</a> for the full methodology.
      </p>
    </header>

    <section class="summary" aria-label="Flagged cluster summary">
      <div class="metric"><span>Flagged clusters</span><strong id="metricClusters">0</strong></div>
      <div class="metric"><span>Highest risk</span><strong id="metricMaxRisk">0.000</strong></div>
      <div class="metric"><span>Total accounts</span><strong id="metricAccounts">0</strong></div>
      <div class="metric"><span>Shared resources</span><strong id="metricResources">0</strong></div>
    </section>

    <section class="review-grid">
      <section class="panel" aria-labelledby="chartTitle">
        <div class="panel-head">
          <h2 id="chartTitle">Risk Score Distribution</h2>
          <span>sorted high to low</span>
        </div>
        <div class="chart-wrap">
          <svg id="riskChart" viewBox="0 0 620 360" role="img" aria-label="Bar chart of risk score by cluster"></svg>
        </div>
      </section>

      <section class="panel" aria-labelledby="tableTitle">
        <div class="panel-head">
          <h2 id="tableTitle">Cluster Queue</h2>
          <span>click a row for details</span>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th><button type="button" data-sort="cluster_id">Cluster <span></span></button></th>
                <th><button type="button" data-sort="risk_score">Risk <span></span></button></th>
                <th><button type="button" data-sort="cluster_size">Size <span></span></button></th>
                <th><button type="button" data-sort="entity_reuse_ratio">Reuse <span></span></button></th>
                <th><button type="button" data-sort="internal_density">Density <span></span></button></th>
                <th><button type="button" data-sort="recommended_action">Action <span></span></button></th>
              </tr>
            </thead>
            <tbody id="clusterRows"></tbody>
          </table>
        </div>
      </section>
    </section>
  </main>

  <script>
    const AUDIT_LOG = {data_literal};

    const state = {{
      sortKey: "risk_score",
      sortDirection: "desc",
      openClusterId: null,
      expandedLists: {{}}
    }};

    const columns = {{
      cluster_id: row => row.cluster_id,
      risk_score: row => row.risk_score,
      cluster_size: row => row.contributing_features.cluster_size,
      entity_reuse_ratio: row => row.contributing_features.entity_reuse_ratio,
      internal_density: row => row.contributing_features.internal_density,
      recommended_action: row => row.recommended_action
    }};

    const formatters = {{
      cluster_id: value => String(value),
      risk_score: value => value.toFixed(4),
      cluster_size: value => String(Math.round(value)),
      entity_reuse_ratio: value => value.toFixed(4),
      internal_density: value => value.toFixed(4),
      recommended_action: value => value
    }};

    function sortedRows() {{
      return [...AUDIT_LOG].sort((a, b) => {{
        const av = columns[state.sortKey](a);
        const bv = columns[state.sortKey](b);
        let result;
        if (typeof av === "number" && typeof bv === "number") {{
          result = av - bv;
        }} else {{
          result = String(av).localeCompare(String(bv));
        }}
        return state.sortDirection === "asc" ? result : -result;
      }});
    }}

    function riskColor(score) {{
      if (score >= 0.95) return "var(--high)";
      if (score >= 0.70) return "var(--mid)";
      return "var(--good)";
    }}

    function setSummary() {{
      const totalAccounts = AUDIT_LOG.reduce((sum, row) => sum + row.member_account_ids.length, 0);
      const totalResources = AUDIT_LOG.reduce((sum, row) => sum + row.shared_resources.length, 0);
      const maxRisk = Math.max(...AUDIT_LOG.map(row => row.risk_score));

      document.getElementById("metricClusters").textContent = AUDIT_LOG.length;
      document.getElementById("metricMaxRisk").textContent = maxRisk.toFixed(4);
      document.getElementById("metricAccounts").textContent = totalAccounts;
      document.getElementById("metricResources").textContent = totalResources;
    }}

    function drawChart() {{
      const svg = document.getElementById("riskChart");
      svg.replaceChildren();

      const rows = [...AUDIT_LOG].sort((a, b) => b.risk_score - a.risk_score);
      const width = 620;
      const height = 360;
      const margin = {{ top: 18, right: 20, bottom: 58, left: 52 }};
      const plotWidth = width - margin.left - margin.right;
      const plotHeight = height - margin.top - margin.bottom;
      const barGap = 5;
      const barWidth = (plotWidth - barGap * (rows.length - 1)) / rows.length;

      [0, 0.25, 0.5, 0.75, 1].forEach(tick => {{
        const y = margin.top + plotHeight - tick * plotHeight;
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", margin.left);
        line.setAttribute("x2", width - margin.right);
        line.setAttribute("y1", y);
        line.setAttribute("y2", y);
        line.setAttribute("stroke", tick === 0 ? "#94a3b8" : "#e2e8f0");
        svg.appendChild(line);

        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("x", margin.left - 10);
        label.setAttribute("y", y + 4);
        label.setAttribute("text-anchor", "end");
        label.setAttribute("class", "axis-label");
        label.textContent = tick.toFixed(2);
        svg.appendChild(label);
      }});

      rows.forEach((row, index) => {{
        const x = margin.left + index * (barWidth + barGap);
        const barHeight = row.risk_score * plotHeight;
        const y = margin.top + plotHeight - barHeight;

        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rect.setAttribute("class", "bar");
        rect.setAttribute("x", x);
        rect.setAttribute("y", y);
        rect.setAttribute("width", Math.max(4, barWidth));
        rect.setAttribute("height", barHeight);
        rect.setAttribute("rx", 3);
        rect.setAttribute("fill", riskColor(row.risk_score));
        rect.setAttribute("tabindex", "0");
        rect.setAttribute("role", "button");
        rect.setAttribute("aria-label", `Open cluster ${{row.cluster_id}}, risk ${{row.risk_score.toFixed(4)}}`);
        rect.addEventListener("click", () => openCluster(row.cluster_id));
        rect.addEventListener("keydown", event => {{
          if (event.key === "Enter" || event.key === " ") {{
            event.preventDefault();
            openCluster(row.cluster_id);
          }}
        }});
        svg.appendChild(rect);

        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("x", x + barWidth / 2);
        label.setAttribute("y", height - margin.bottom + 18);
        label.setAttribute("text-anchor", "end");
        label.setAttribute("transform", `rotate(-45 ${{x + barWidth / 2}} ${{height - margin.bottom + 18}})`);
        label.setAttribute("class", "axis-label");
        label.textContent = `C${{row.cluster_id}}`;
        svg.appendChild(label);
      }});
    }}

    function updateSortIndicators() {{
      document.querySelectorAll("th button").forEach(button => {{
        const marker = button.querySelector("span");
        marker.textContent = button.dataset.sort === state.sortKey
          ? (state.sortDirection === "asc" ? "↑" : "↓")
          : "";
      }});
    }}

    function renderList(items, type, clusterId, renderer) {{
      const key = `${{clusterId}}:${{type}}`;
      const expanded = Boolean(state.expandedLists[key]);
      const limit = 8;
      const visible = expanded ? items : items.slice(0, limit);
      const hiddenCount = Math.max(0, items.length - limit);
      const container = document.createElement(type === "members" ? "ul" : "ol");
      container.className = type === "members" ? "chips" : "resource-list";

      visible.forEach(item => {{
        const li = document.createElement("li");
        renderer(li, item);
        container.appendChild(li);
      }});

      const wrapper = document.createElement("div");
      wrapper.appendChild(container);

      if (hiddenCount > 0) {{
        const button = document.createElement("button");
        button.type = "button";
        button.className = "toggle-list";
        button.textContent = expanded ? "Show fewer" : `Show all ${{items.length}}`;
        button.addEventListener("click", event => {{
          event.stopPropagation();
          state.expandedLists[key] = !expanded;
          renderTable();
        }});
        wrapper.appendChild(button);
      }}

      return wrapper;
    }}

    function detailRow(row) {{
      const tr = document.createElement("tr");
      tr.className = "detail-row";

      const td = document.createElement("td");
      td.colSpan = 6;

      const detail = document.createElement("div");
      detail.className = "detail";

      const resources = document.createElement("section");
      resources.className = "detail-section";
      resources.innerHTML = `<h3>Shared resources <span class="count">(${{row.shared_resources.length}})</span></h3>`;
      resources.appendChild(renderList(row.shared_resources, "resources", row.cluster_id, (li, item) => {{
        li.textContent = item;
      }}));

      const members = document.createElement("section");
      members.className = "detail-section";
      members.innerHTML = `<h3>Member accounts <span class="count">(${{row.member_account_ids.length}})</span></h3>`;
      members.appendChild(renderList(row.member_account_ids, "members", row.cluster_id, (li, item) => {{
        li.className = "chip";
        li.textContent = item;
      }}));

      detail.append(resources, members);
      td.appendChild(detail);
      tr.appendChild(td);
      return tr;
    }}

    function dataRow(row) {{
      const tr = document.createElement("tr");
      tr.className = "data-row";
      if (state.openClusterId === row.cluster_id) tr.classList.add("is-open");
      tr.addEventListener("click", () => {{
        state.openClusterId = state.openClusterId === row.cluster_id ? null : row.cluster_id;
        renderTable();
      }});

      const cells = [
        ["cluster_id", row.cluster_id],
        ["risk_score", row.risk_score],
        ["cluster_size", row.contributing_features.cluster_size],
        ["entity_reuse_ratio", row.contributing_features.entity_reuse_ratio],
        ["internal_density", row.contributing_features.internal_density],
        ["recommended_action", row.recommended_action]
      ];

      cells.forEach(([key, value]) => {{
        const td = document.createElement("td");
        td.className = key === "recommended_action" ? "" : "num";

        if (key === "risk_score") {{
          const risk = document.createElement("span");
          risk.className = "risk";
          const dot = document.createElement("span");
          dot.className = "risk-dot";
          dot.style.background = riskColor(value);
          const label = document.createElement("span");
          label.textContent = formatters[key](value);
          risk.append(dot, label);
          td.appendChild(risk);
        }} else if (key === "recommended_action") {{
          const action = document.createElement("span");
          action.className = "action";
          action.textContent = value.replaceAll("_", " ");
          td.appendChild(action);
        }} else {{
          td.textContent = formatters[key](value);
        }}

        tr.appendChild(td);
      }});

      return tr;
    }}

    function renderTable() {{
      const tbody = document.getElementById("clusterRows");
      tbody.replaceChildren();
      const rows = sortedRows();

      if (!rows.length) {{
        const tr = document.createElement("tr");
        const td = document.createElement("td");
        td.colSpan = 6;
        td.className = "empty";
        td.textContent = "No flagged clusters found.";
        tr.appendChild(td);
        tbody.appendChild(tr);
        return;
      }}

      rows.forEach(row => {{
        tbody.appendChild(dataRow(row));
        if (state.openClusterId === row.cluster_id) {{
          tbody.appendChild(detailRow(row));
        }}
      }});
      updateSortIndicators();
    }}

    function openCluster(clusterId) {{
      state.openClusterId = state.openClusterId === clusterId ? null : clusterId;
      renderTable();
      const openRow = document.querySelector("tr.is-open");
      if (openRow) openRow.scrollIntoView({{ behavior: "smooth", block: "center" }});
    }}

    document.querySelectorAll("th button").forEach(button => {{
      button.addEventListener("click", () => {{
        const key = button.dataset.sort;
        if (state.sortKey === key) {{
          state.sortDirection = state.sortDirection === "asc" ? "desc" : "asc";
        }} else {{
          state.sortKey = key;
          state.sortDirection = key === "recommended_action" ? "asc" : "desc";
        }}
        renderTable();
      }});
    }});

    setSummary();
    drawChart();
    renderTable();
  </script>
</body>
</html>
"""


def main() -> None:
    rows = load_audit_rows()
    data_literal = json.dumps(rows, indent=4)
    data_literal = data_literal.replace("</", "<\\/")
    OUTPUT.write_text(html_template(data_literal), encoding="utf-8")
    print(f"Wrote {OUTPUT.name} with {len(rows)} embedded flagged clusters.")


if __name__ == "__main__":
    main()

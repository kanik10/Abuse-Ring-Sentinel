# PRD: Abuse-Ring Sentinel

**Track:** Razorpay AI Buildathon 2026 — Track 02, AI Risk Manager  
**Loss Class:** Coordinated Syndicate Abuse (Multi-Account Promo, Referral Farming & Resource Rotation)  
**Window:** Aug 29 – Sep 5, 2026  
**Status:** ✅ Complete & Production-Ready (All Milestones Completed & Validated)

---

## 1. Executive Summary & Overview

**Abuse-Ring Sentinel** is a defense-only, graph-native intelligence engine designed to detect, disentangle, and dismantle organized fraud syndicates colluding to exploit promotional discounts, sign-up incentives, and referral bonuses across e-commerce and digital payment platforms.

Individual syndicate accounts mimic organic customer behavior — placing unremarkable orders at standard prices with realistic intervals. The fraud is invisible in tabular per-row scoring; it is only exposed at the **relational graph infrastructure level**: shared device hardware identifiers, masked virtual card payments, clustered delivery coordinates, and circular referral chain cycles.

Sentinel operates strictly as a **Tier-2 Human-in-the-Loop Decision Support Cockpit**: it reconstructs multi-entity bipartite graphs, runs Louvain community detection, extracts 17 structural and referral features, evaluates calibrated cluster probabilities via a cross-validated Champion Model, and presents complete visual and forensic evidence (D3.js interactive graphs, SHAP explanations, account risk rosters, and C-level PDF audit dossiers). **It never executes automated blocks, cancellations, or account freezes.**

---

## 2. Problem Statement & Loss Mechanism

Coordinated multi-account abuse represents one of the fastest-growing loss categories in fintech and e-commerce:
1. **The Tabular Blindspot**: Fraudsters create dozens of nominally independent accounts, ensuring each account's order size and velocity remain below standard velocity rate limits. Tabular ML models (Random Forest, XGBoost on user tables) classify each account as benign.
2. **Entity Laundering**: Syndicates rotate through disposable emails, burner phone numbers, and proxy subnets, but economic constraints force them to reuse physical devices, payment instruments, delivery addresses, and circular referral codes.
3. **The Lookahead Fallacy in Graph ML**: Post-hoc graph models evaluate static graphs at Day 30 after the funds have already cleared. Production systems require **point-in-time temporal reconstruction** that proves fraud can be intercepted *before* clearing without lookahead leakage.

---

## 3. Product Goals & Objectives

* **G1 — High-Fidelity Syndicate Detection**: Unify multi-entity relationships across Device, IP, Payment, Address, and Referral networks into weighted bipartite projections to identify coordinated clusters.
* **G2 — Statistically Honest Evaluation**: Measure performance using out-of-fold cross-validation, 15-seed pooled sensitivity sweeps, and 10,000-resample non-parametric bootstrap confidence intervals.
* **G3 — Zero-Lookahead Temporal Validation**: Reconstruct historical point-in-time snapshots ($T$) to prove fraud volume is intercepted before execution without future-data leakage.
* **G4 — Forensic Interpretability**: Provide transparent evidence trails per cluster — SHAP log-odds feature attributions, entity reuse ratios, and internal mastermind vs. sleeper account rankings.
* **G5 — Defense-Only Architecture**: Enforce at the type-system level that `RecommendedAction` can only ever be `FLAG_FOR_REVIEW`. No automated punitive code paths exist.

---

## 4. System Architecture & Pipeline Flow

![Abuse-Ring Sentinel End-to-End System Architecture](assets/system_architecture_flow.png)

> **Detailed Architectural & Mathematical Reference**: The comprehensive technical breakdown covering mathematical formulations, formal graph equations, and algorithmic pseudocode across all 5 pipeline layers is documented in [`docs/architectural flow diagram of abuse ring sentinel.pdf`](architectural%20flow%20diagram%20of%20abuse%20ring%20sentinel.pdf) (*"Mathematical & Model Flow for Abuse-Ring Detection & Risk Management"*).

```
                                  [Raw Multi-Entity Tables]
                   (Accounts, Devices, Payments, Addresses, IPs, Orders, Referrals)
                                              │
                                              ▼
                             [Entity Resolution & Bipartite Projection]
                      Weighted Account-to-Account Graph G(V, E) via Inverse Resource Frequency
                                              │
                                              ▼
                                 [Louvain Community Detection]
                             Candidate Syndicate Subgraphs (Size ≥ 2)
                                              │
                                              ▼
                                 [Dual Feature Engineering Engine]
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
           [Graph Topology Features]                       [Referral Dynamic Features]
       (Size, Entity Reuse, Density,                  (Cycles, Activation Latencies,
        PageRank, Degree, Betweenness)                 Cluster Referral Density, Overlap)
                     └────────────────────────┬────────────────────────┘
                                              ▼
                            [Champion Calibrated Classifier]
                  Logistic Regression (StandardScaler) on 7 Champion Features
                                              │
                                              ▼
                                [Pooled Cost-Optimal Threshold]
                         CHOSEN_THRESHOLD = 0.1333 (15-Seed Plateau Midpoint)
                                              │
                                              ▼
                      [Tier-2 Investigator Cockpit & Audit Export]
            Streamlit Interactive Dashboard (D3.js Physics, Single-Account Ego-Graph,
             SHAP Waterfall, Counterfactual Horizon) + Automated ReportLab PDF Dossiers
```

---

## 5. Technical Specifications & Stack

| Layer | Technology | Role / Specification |
| :--- | :--- | :--- |
| **Synthetic Generator** | Python, NumPy, Pandas, Faker | Multi-seed generator with sleeper accounts, bridge anomalies, organic referral DAGs, and circular bonus farming rings |
| **Entity Resolution & Graph** | NetworkX, Python-Louvain | Bipartite projection with co-occurrence weighting; modularity optimization clustering |
| **Feature Extraction** | Custom Feature Pipeline | 17 features spanning graph topology, temporal spans, and directed referral dynamics |
| **Classifier** | Scikit-Learn (`Pipeline`) | Calibrated Logistic Regression + StandardScaler on 7 Champion Features (`CHAMPION_FEATURE_COLS`) |
| **Threshold Optimization** | Cost-Curve Minimization | Pooled 15-seed sweep over 0.1x–100x FP cost multipliers (Plateau Midpoint = `0.1333`) |
| **Validation Engine** | Percentile Bootstrap | 10,000 resamples computing 95% Confidence Intervals for Precision, Recall, and Net Cost |
| **Backtest Engine** | Point-in-Time Temporal Reconstruction | 101 historical snapshots at 7-day intervals evaluating zero-lookahead latency and volume prevention |
| **Investigator UI** | Streamlit, Altair, D3.js | Dark-mode cockpit with ego-graph controls, bipartite tree views, and SHAP explainability |
| **Dossier Generation** | ReportLab | Automated executive PDF reports with matching `#ec4899` pink / `#3b82f6` blue visual tokens |

---

## 6. Validated Benchmark Performance

### Primary Cross-Validated Results (Canonical Population: 70 Candidate Clusters)
* **Monitored Accounts**: 668 accounts with active resource sharing linkages
* **True Positives (TP)**: **26 / 26 flagged clusters** correctly matched to ring membership — 18 clusters contain resource-sharing ring members, 8 contain referral-ring members. (Ground truth has **20 distinct resource-sharing rings + 8 referral rings = 28 named rings**; two resource-ring pairs land in the same Louvain community, so 20 rings collapse to 18 ring-clusters. See §6's temporal backtest below, which evaluates at the ring level and reports 28/28.)
* **False Positives (FP)**: **0** benign clusters falsely flagged
* **False Negatives (FN)**: **0** rings missed
* **Precision**: **`1.000`** (95% Bootstrap CI: `[1.000, 1.000]`)
* **Recall**: **`1.000`** (95% Bootstrap CI: `[1.000, 1.000]`)

### Financial & Business Impact
* **Gross Fraud Volume Protected**: **`Rs. 7,08,255`**
* **Review Friction Cost (at 1.0x AOV)**: **`Rs. 1,777`** (only 2 coincidental bystander accounts caught across all clusters)
* **Net Financial Impact**: **`Rs. 7,06,478`** in verified savings
* **Business Return on Investment**: **`~399x` return** per rupee of false-positive review expense

### Zero-Lookahead Temporal Backtest (101 Historical Snapshots at 7-Day Intervals)
* **Ring Detection Rate**: **`28 / 28` (`100%`)**
* **Median Detection Latency**: **`30.0 days`** from initial member creation
* **Fraud Volume Intercepted Before Execution**: **`90.4%`**
* **Counterfactual Intercepted Value**: **`Rs. 6.71 Lakhs`**

---

## 7. Generative Assumptions & Production Generalization Bounds

To maintain the highest standards of scientific and engineering rigor, the following generative assumptions of the synthetic benchmark are explicitly documented alongside their production transition pathways:

1. **Cluster Scale Disparity**:
   - *Benchmark Property*: Benign coincidental groups in the generator are constrained to small household/pair units ($N = 2 \text{ to } 4$), whereas fraud rings range from $N = 5 \text{ to } 40$.
   - *Production Reality*: Corporate VPNs, university dorms, and public NAT gateways create benign sharing clusters of $N \ge 20$.
   - *Production Mitigation*: Production deployment introduces TF-IDF inverse-entity discounting and soft edge weighting to downweight high-entropy public IP ranges before Louvain community detection.
2. **Orthogonal Entity Allocation**:
   - *Benchmark Property*: Coincidental groups share exactly 1 entity type (e.g. only IP or only address), creating a sharp floor between benign $\text{ERR} \le 0.14$ and fraud $\text{ERR} \ge 0.20$.
   - *Production Reality*: Families frequently share both a physical delivery address and a home Wi-Fi IP without fraudulent intent.
   - *Production Mitigation*: Multi-resource weighting assigns exponential weight to shared payment instruments and hardware device hashes over transient network IPs.
3. **Defense-in-Depth Human Governance**:
   - Because real-world boundary cases exist, Sentinel’s **defense-only architecture** ensures that borderline clusters are placed into an investigator queue with forensic evidence rather than subjected to automated disruption.

---

## 8. Milestone Execution & Deliverables Status

| Day | Milestone | Deliverable | Status |
| :---: | :--- | :--- | :---: |
| **Day 1** | Synthetic Data & Ground Truth | `generate_synthetic_data_v2.py`, multi-entity relational schemas | ✅ Completed |
| **Day 2** | Graph Construction & Clustering | `graph_builder.py`, `community_detection.py` (Louvain partitioning) | ✅ Completed |
| **Day 3** | Feature Engineering & Baseline | `feature_engineering.py`, `referral_features.py`, `classifier.py` | ✅ Completed |
| **Day 4** | Cost-Based Threshold Optimization | `threshold_sweep.py`, cost sensitivity curves across 0.1x–100x FP multipliers | ✅ Completed |
| **Day 5** | Defense-Only Scoring & Cockpit | `risk_scoring.py`, `account_scoring.py`, interactive `app_interactive.py` | ✅ Completed |
| **Day 6** | Statistical Confidence & Backtesting | `bootstrap_threshold_ci.py` (10k resamples), `pooled_threshold_selection.py` (15 seeds), `phase3_temporal_backtest.py` (101 snapshots, built on `temporal_reconstruction.py`) | ✅ Completed |
| **Day 7** | Hardening, Polish & Documentation | Pinned `requirements.txt`, ReportLab `pdf_generator.py`, synchronized PRD and Data Dictionary | ✅ Completed |

---

## 9. Defense-Only Structural Guarantee

Sentinel enforces compliance with the buildathon's defense-only mandate through structural software constraints:
* **The `RecommendedAction` Enum**: Contains exactly one member: `FLAG_FOR_REVIEW`. Adding an automated action like `BLOCK` or `CANCEL` requires an explicit modification to the core enum definition.
* **Investigator Transparency**: Every cluster output carries full feature attributions, shared entity identifiers, and member rosters to ensure human accountability.
* **Audit Trail**: All scoring runs persist verifiable forensic outputs (`temporal_detection_latencies.csv`, `metrics_summary.json`).
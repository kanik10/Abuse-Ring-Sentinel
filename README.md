# Abuse-Ring Sentinel 🛡️🕸️

> **Graph-Native Multi-Entity Syndicate Detection & Tier-2 Human-in-the-Loop Risk Cockpit**  
> *Built for Razorpay AI Buildathon 2026 — Track 02: AI Risk Manager*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit%20Interactive-FF4B4B.svg)](https://streamlit.io/)
[![NetworkX](https://img.shields.io/badge/Graph-NetworkX%20%2B%20Louvain-green.svg)](https://networkx.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Champion%20Calibrated%20Model-orange.svg)](https://scikit-learn.org/)
[![Compliance](https://img.shields.io/badge/Policy-Strictly%20Defense--Only-purple.svg)](PRD.md#11-defense-only-compliance)
[![Audit](https://img.shields.io/badge/Audit-100%25%20Reproducible-success.svg)](final_report.md)

---

## 📌 Executive Summary

Modern organized fraud syndicates no longer operate via crude brute-force attacks. Instead, they deploy **distributed sleeper networks**: dozens of nominally separate accounts that mimic organic consumer habits, place average-ticket orders at standard intervals, and quietly drain margins through sign-up promos and circular referral farming.

**Tabular fraud models are blind to this.** Because each individual account looks completely benign in isolation, row-level classifiers miss coordinated abuse.

**Abuse-Ring Sentinel** solves this at the **relational graph infrastructure level**:
1. **Multi-Entity Bipartite Resolution**: Projects connections across physical devices, payment instruments, delivery addresses, and IP subnets into a unified, weighted account-to-account graph.
2. **Hybrid Network Dynamics**: Combines graph community topology (Louvain modularity) with directed referral chain features (circular cycles, activation latencies, and within-cluster referral density).
3. **Calibrated Champion Scoring**: Evaluates candidate clusters using a cross-validated Logistic Regression pipeline (`StandardScaler` + `LogisticRegression`) tuned to a cross-seed cost-optimal threshold (`0.1333`).
4. **Zero-Lookahead Temporal Reconstruction**: Simulates point-in-time daily snapshots across 101 historical timestamps to prove that **90.4% of fraudulent order volume is intercepted before execution**.
5. **Tier-2 Forensic Cockpit**: Equips risk analysts with an interactive Streamlit UI featuring PyVis physics graphs, single-account ego-graphs (with bipartite tree toggle), SHAP waterfall explanations, and exportable C-level PDF audit dossiers.

---

## 📊 Key Verified Benchmark Metrics

All metrics below are computed directly from the committed canonical dataset (`day1_data/`) and cross-validated out-of-fold:

| Metric Category | Measure | Value | Notes / Validation Source |
| :--- | :--- | :--- | :--- |
| **Detection Accuracy** | **Precision** | **`1.000`** | 95% Bootstrap CI: `[1.000, 1.000]` (10,000 resamples) |
| | **Recall** | **`1.000`** | 95% Bootstrap CI: `[1.000, 1.000]` (26/26 rings detected) |
| | **F1 Score** | **`1.000`** | 0 False Positives, 0 False Negatives across all clusters |
| **Financial Impact** | **Gross Fraud Protected** | **`Rs. 7,08,255`** | Full volume intercepted across 18 resource + 8 referral rings |
| | **Review Friction Cost** | **`Rs. 1,777`** | Only 2 coincidental bystander accounts caught in Cluster #60 crossfire |
| | **Net Financial Return** | **`~399x` ROI** | Rs. 399 saved per rupee of manual investigator review cost |
| **Temporal Latency** | **Volume Prevented** | **`90.4%`** | Evaluated across **101 historical point-in-time snapshots** |
| | **Detection Latency** | **`30.0 days`** | Median days from syndicate formation to first threshold trigger |
| | **Detection Rate** | **`100%` (28/28)** | 100% of rings intercepted before their operational completion |

---

## 🏛️ System Architecture

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
            Streamlit Interactive Dashboard (PyVis Physics, Single-Account Ego-Graph,
             SHAP Waterfall, Counterfactual Horizon) + Automated ReportLab PDF Dossiers
```

---

## 🚀 Key Technical Innovations

### 1. Dual-Topology Feature Engineering
Organized abuse is bifurcated between hardware reuse and incentive manipulation. Sentinel extracts 18 features across two complementary engines:
* **Resource-Sharing Graph Topology**: `cluster_size`, `entity_reuse_ratio` ($\text{ERR}$), `internal_density`, degree centrality, betweenness centrality, PageRank authority, creation-time burstiness, and order amount dispersion.
* **Directed Referral Dynamics**: Directed cycle ratios ($A \to B \to C \to A$), referral-resource overlap ratios, median activation lag in days, and within-cluster referral density.

### 2. Cross-Seed Pooled Threshold Selection (`0.1333`)
Single-dataset threshold tuning suffers from flat-plateau sample variance. Sentinel evaluates candidate thresholds across **15 independent synthetic seeds** (954 clusters, 292 true rings) minimizing total business loss:
$$\text{Cost}(t) = \text{False Positive Review Overhead}(t) + \text{Unintercepted Fraud Loss}(t)$$
* At the legacy threshold of `0.4811`, 3 rings were missed across seeds (dropping recall to 98.9%).
* At the pooled plateau midpoint of **`0.1333`**, **100% of rings are detected with 0 false positive clusters**, reducing total operating cost by 15.4% while maintaining a 50.6% safety buffer above benign background noise.

### 3. Zero-Lookahead Temporal Backtesting
Evaluating graph models on static post-hoc snapshots introduces severe lookahead bias. Sentinel includes a dedicated temporal reconstruction engine ([temporal_reconstruction.py](temporal_reconstruction.py)) that:
* Reconstructs historical graph state $G_t$ considering only entities and transactions observed on or before $t$.
* Evaluates 101 historical daily snapshots from October 2024 to August 2026.
* Measures true detection latency and proves that **90.4% of fraudulent volume is caught before clearance**.

### 4. Strictly Defense-Only Compliance
Sentinel complies with the buildathon's defense-only mandate by design:
* **No Automated Punitive Actions**: The software contains no code paths to block accounts, freeze funds, or cancel orders.
* **Structural Enum Constraint**: The system output schema enforces that `RecommendedAction` is strictly `FLAG_FOR_REVIEW`.
* **Human-in-the-Loop Triage**: All outputs surface forensic evidence for human risk analysts to review.

---

## 🖥️ Interactive Forensic Cockpit (`app_interactive.py`)

The Streamlit dashboard provides a comprehensive workspace for Tier-2 Trust & Safety analysts:

1. **Executive Portfolio Scorecard**: Live counts of monitored accounts, Louvain clusters, precision/recall cards, and net financial ROI.
2. **Interactive PyVis Network Graph**: Physics-simulated cluster visualization highlighting high-degree mastermind hubs in `#ec4899` pink and peripheral sleepers in `#3b82f6` blue.
3. **Single-Account Ego-Graph**:
   * **Projected Ego-Network**: Shows account-to-account co-occurrence edges with hop-radius filters.
   * **Entity Bipartite Tree**: Expands an account's direct links to raw physical devices, payment hashes, and IP subnets.
4. **SHAP Waterfall & Feature Attributions**: Log-odds breakdown explaining exactly which signals drove the risk score.
5. **Internal Cluster Account Roster**: Ranks accounts by risk score, creation timestamp, and total transaction volume.
6. **Counterfactual Temporal Horizon**: Visualizes how early the cluster was detected relative to its order completion timeline.
7. **Executive PDF Export**: Generates a boardroom-ready, multi-page ReportLab PDF audit dossier with embedded KPI tables and color-matched charts.

---

## 📂 Repository Structure

```
Abuse-Ring-Sentinel/
├── day1_data/                              # Canonical multi-entity relational dataset
│   ├── accounts.csv                        # 6,558 accounts (legit, sleepers, ring members)
│   ├── orders.csv                          # 18,966 transactions
│   ├── referrals.csv                       # 2,240 directed referral links
│   ├── resolved_account_device.csv         # 6,591 resolved device linkages
│   ├── resolved_account_payment.csv        # 9,601 resolved payment instrument linkages
│   ├── resolved_account_address.csv        # 9,681 resolved physical address linkages
│   ├── resolved_account_ip.csv             # 6,594 resolved IP subnet linkages
│   ├── ground_truth.csv                    # 26 fraud rings (evaluation only)
│   ├── raw_to_true_resource.csv            # Entity resolution mapping crosswalk
│   └── bridge_log.csv                      # Adversarial bridge edge cases
│
├── entity_resolution.py                    # Entity normalization & mapping
├── graph_builder.py                        # Bipartite network projection & edge weighting
├── community_detection.py                  # Louvain modularity clustering
├── feature_engineering.py                  # Core graph topological feature extraction
├── referral_features.py                    # Directed referral dynamics & cycle detection
├── classifier.py                           # 5-fold cross-validation of candidate models
├── risk_scoring.py                         # Production Champion model fitting & export
├── account_scoring.py                      # Intra-cluster node ranking (mastermind vs sleeper)
├── threshold_config.py                     # Centralized operating threshold configuration (0.1333)
├── threshold_sweep.py                      # 0.1x - 100x FP cost curve sensitivity analysis
├── pooled_threshold_selection.py          # 15-seed pooled cost-curve optimizer
├── bootstrap_threshold_ci.py               # 10,000-resample non-parametric bootstrap CIs
├── temporal_reconstruction.py              # Zero-lookahead 101-snapshot temporal backtest
├── final_threshold_report.py               # Synchronized audit report generator
├── pdf_generator.py                        # ReportLab executive PDF dossier generator
├── app_interactive.py                      # Streamlit interactive forensic cockpit
│
├── final_model.joblib                      # Serialized Champion model pipeline
├── cluster_features.csv                    # 18 engineered features for 70 clusters
├── cluster_predictions.csv                 # Ground truth and out-of-fold predictions
├── temporal_detection_latencies.csv        # 101-snapshot backtest latency audit
├── metrics_summary.json                    # Machine-readable audit metrics
├── final_report.md                         # Detailed markdown evaluation report
├── requirements.txt                        # Pinned dependencies
├── PRD.md                                  # Product Requirements Document
└── DATA_DICTIONARY.md                      # Detailed data dictionary
```

---

## ⚡ Quickstart & How to Run

### 1. Environment Setup
Clone the repository and install the pinned dependencies:
```bash
git clone https://github.com/kanik10/Abuse-Ring-Sentinel.git
cd Abuse-Ring-Sentinel
pip install -r requirements.txt
```

### 2. Launch the Interactive Dashboard
Launch the Streamlit Trust & Safety Cockpit:
```bash
streamlit run app_interactive.py
```
Open your browser at `http://localhost:8501`. Explore clusters, inspect ego-graphs, adjust hop radii, and export executive PDF dossiers.

### 3. Verify Model Evaluation & Metrics
Run the synchronized threshold evaluation against the committed data:
```bash
python final_threshold_report.py
```
*Outputs: 70 clusters, 26 true rings, 26 flagged (100% Precision, 100% Recall), Rs. 708,255 protected, Rs. 1,777 FP review cost (~399x ROI).*

### 4. Run the Zero-Lookahead Temporal Backtest
Execute the 101-snapshot historical reconstruction backtest:
```bash
python temporal_reconstruction.py
```
*Outputs: 101 snapshots evaluated, 100% ring detection rate, 30.0-day median detection latency, 90.4% fraud volume prevented.*

### 5. Run Bootstrap Statistical Confidence Intervals
Compute 10,000 bootstrap resamples for Precision, Recall, and Cost:
```bash
python bootstrap_threshold_ci.py
```

---

## 🔬 Generative Assumptions & Production Transition

To maintain the highest standards of engineering integrity, Sentinel’s synthetic benchmark properties and their production deployment pathways are transparently documented:

1. **Cluster Size Boundary ($N \le 4$)**:
   - *Benchmark Property*: Benign coincidental sharing in the generator is bounded at $N = 2 \text{ to } 4$, while fraud rings range from $N = 5 \text{ to } 40$.
   - *Production Reality*: Corporate VPNs, university dorms, and public Wi-Fi form benign clusters of $N \ge 20$.
   - *Production Pathway*: Production pipelines implement TF-IDF inverse-entity discounting on high-entropy public IP subnets to dampen dense benign hubs before community detection.
2. **Orthogonal Resource Allocation**:
   - *Benchmark Property*: Synthetic coincidental groups share exactly 1 resource type, establishing an artificial floor between benign $\text{ERR} \le 0.14$ and fraud $\text{ERR} \ge 0.20$.
   - *Production Reality*: Roommates and families often share both home Wi-Fi and a physical delivery address without fraud intent.
   - *Production Pathway*: Multi-resource edge weighting gives exponentially higher weight to shared hardware hashes and stored payment instruments over transient IP addresses.
3. **Human-in-the-Loop Governance**:
   - Sentinel is architected as an **analyst triage tool rather than an automated ban hammer**. Borderline clusters are queued for investigation with complete evidence dossiers, preventing customer disruption on edge cases.

---

## 📜 License & Acknowledgments

Developed for the **Razorpay AI Buildathon 2026** (Track 02: AI Risk Manager).  
Built with open-source tools: [Streamlit](https://streamlit.io/), [NetworkX](https://networkx.org/), [Scikit-Learn](https://scikit-learn.org/), [Altair](https://altair-viz.github.io/), and [ReportLab](https://www.reportlab.com/).

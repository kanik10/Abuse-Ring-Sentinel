# Day 4 Final Report — Abuse-Ring Sentinel Operating Point

## Chosen model and threshold
- **Model:** Logistic Regression (pure graph)
- **Threshold:** 0.7732484382694863
- **Why:** cost-optimal threshold was stable across the full 0.1x-100x
  false-positive-cost sweep (Day 4 Phase 2). The model is also
  interpretable, and uses the same pure graph-topology feature set already
  recommended in Day 3 as the more generalizable result.

## Cluster-level confusion matrix (out-of-fold, 65 clusters)
| | Predicted: ring | Predicted: not ring |
|---|---|---|
| **Actual: ring** | TP = 19 | FN = 1 |
| **Actual: not ring** | FP = 0 | TN = 45 |

- Precision: 1.000
- Recall: 0.950

## Account-level cost/benefit at this threshold
- Ring fraud value protected (caught): Rs.460,856.34
- Ring fraud value still missed: Rs.5,202.96
- Legitimate accounts wrongly caught in a flagged cluster: 10 total -- 10 coincidental/benign-lookalike (100%), 0 other (0%)
- Cost of those false positives, at 1x avg order value per account: Rs.8,891.14
- **Net: protects Rs.460,856 of fraud at a cost of roughly
  Rs.8,891 in false-positive review/friction (at a conservative
  1x-avg-order-value cost assumption) -- a ~52x return.**

## Honest limitations of this number
- N=65 clusters (19 flagged at this threshold) -- treat
  precision/recall as directionally reliable, not statistically tight.
- The false-positive cost assumption (1x avg order value per wrongly-
  flagged account) is a modeling choice, not a measured business figure --
  see Day 4 Phase 1 for why it's swept rather than asserted as fact.
- This threshold was tuned on the SAME synthetic dataset it's evaluated on
  (out-of-fold within that one dataset, not a separate holdout population).
  A genuinely held-out second synthetic population, or real data, would be
  needed before trusting this threshold in production.

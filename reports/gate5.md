# Gate 5: Transductive graph, ensembling, final calibration, final submissions

## What was built
- `src/reid/transductive.py`
  1. **Image-pair model** P(same | q, q′) on query–query neighbour pairs (top-10 by cosine). Features: raw/whitened
     cosine, ranks in both directions, same/opposite side, SIFT/RANSAC inliers, inlier ratio and spread, keypoint
     count, norm gap, and whether both images' current top-1 labels agree. LightGBM, out-of-fold by identity.
     **AUC 0.974** vs 0.900 for cosine alone.
  2. **Graph**: edge if P(same) > τₑ. Optional consensus filter (Louvain communities ∧ average-linkage clusters must
     agree), following the AAS paper's "trust only where two clusterings agree".
  3. **Label propagation** (Zhou et al. 2004): $F\leftarrow\alpha S F+(1-\alpha)P^0$, $S=D^{-1/2}WD^{-1/2}$, 30
     iterations, rows renormalised. $P^0$ = each query's probabilities over its candidate identities ∪ {new}. A
     neighbour's identity can enter a query's top 5 even if it was not among that query's own 20 candidates.
- Transductive *features* in the reranker / P(new) model: best cosine to another query, minus own top-1 gallery
  cosine, neighbour votes for each candidate.
- Rerankers for stacking: XGBoost, CatBoost (plus LR and MLP wrappers) in `models.py`, combined by mean logit.
- `scripts/run_transductive.py` (re-tune only this stage from a cache) and `scripts/nested_check.py` (honest estimate).

All of it runs identically on CV query sets (the CV analogue of test.csv) and on the real test set, using
features only, never labels.

## Final ablation (CV MAP@5, test-weighted, 5 encounter-grouped open-set splits)
| # | System | CV MAP@5 | Δ vs previous accepted (paired SE) | Decision |
|---|---|---|---|---|
| 1 | raw cosine + threshold τ (Gate 2 baseline) | 0.8486 ± 0.015 | — | baseline |
| 2 | PCA-whitened cosine + threshold | 0.8615 ± 0.013 | +0.013 (0.002) | accept |
| 3 | LightGBM reranker + P(new), global + viewpoint features | 0.8784 ± 0.010 | +0.017 vs 2 | accept |
| 4 | + SIFT/RANSAC features | 0.8947 ± 0.011 | +0.016 (0.0015) | accept |
| 5 | + viewpoint imputer | 0.8939 | −0.001 (0.0005) | reject |
| 6 | + transductive features | 0.8965 ± 0.011 | +0.0018 (0.0005) | borderline (kept: it feeds the final model) |
| 7 | + stacking LGB+XGB+CatBoost | 0.8982 | +0.0017 (0.0004) | reject (< 0.002, triples train time) |
| 8 | 4 + label propagation (no transductive features) | 0.9047 (nested 0.9026) | +0.0103 vs 4 (0.0006) | accept |
| 9 | 6 + label propagation, consensus filter on | 0.9037 | — | not chosen |
| **10** | **6 + label propagation, no consensus (FINAL)** | **0.9052 ± 0.010 (nested 0.9038)** | **+0.0087 vs 6 (0.0005)** | **accept** |

Per species (final): hyena 0.878, leopard 0.926. Tuned propagation: hyena α=0.6, τₑ=0.5; leopard α=0.9, τₑ=0.5.

**Nested estimate.** Propagation has 3 tuned values per species (α, τₑ, new-scale), chosen from about 390
combinations. Choosing them on 4 splits and scoring on the 5th gives 0.9038 instead of 0.9052. The selection optimism
is about 0.001, and the gain over no propagation stays +0.0073 (SE 0.0006). Quote **0.904** as the honest CV.

## Final calibration and novelty
| | π in CV | mean P(new) on test | Saerens-EM π on test |
|---|---|---|---|
| Hyena | 0.32 | 0.33 | 0.34 |
| Leopard | 0.38 | 0.35 | 0.33 |

The estimates agree within 5 points, so no prior-shift correction is applied. The β offsets are small (+0.2 / +0.15),
so the P(new) model needed no large correction.

## The two final submissions
| File | What | CV | New@rank 1 (h / l) |
|---|---|---|---|
| `submissions/sub_20261119-132154_sub_g5_final.csv` | **best CV**: row 10 | 0.905 (nested 0.904) | 39% / 40% |
| `submissions/sub_20261119-132406_sub_g5_noprop.csv` | **hedge**: row 6, no propagation | 0.8965 | 39% / 41% |

The hedge protects against propagation behaving differently on the real test graph (e.g. a wrong edge flipping a cluster).
Earlier submissions kept for leaderboard comparison: `sub_..._baseline_rawcos.csv` (0.849 CV) and
`sub_..._g4_rerank_sift.csv` (0.894 CV).

## Not done, and why
- **Pseudo-gallery second pass** (plan T5, "first to drop"): it would rebuild every CV and test table with augmented
  galleries (~30 min per run). Propagation already moves a confident test image's identity to its test neighbours,
  which is the same effect. Not attempted.
- **NCA and the viewpoint nuisance projection** (Gate 3 variants): LDA and whitening already cover the linear-metric
  family, and in the ablation removing either costs only 0.002–0.003, so a third linear metric has little room left.
- **HotSpotter LNBNN** (stretch): not attempted.

## What surprised us
- Propagation helps leopards most (+0.010): leopard test images come in large same-animal groups (bursts and
  repeated encounters), so one confident member labels the rest.
- The consensus filter *hurt* slightly (0.9037 vs 0.9052). The pair model is accurate enough (AUC 0.97) that the extra
  agreement requirement mainly discards correct edges.
- The transductive features add only +0.002 once propagation is on: both use the same "is my neighbour a test image" signal.

## Viva questions
1. *Is using test images without labels allowed, and how do you make sure it doesn't inflate CV?* The rules allow
   transforming and matching the provided features. We use only test *features*. Inside CV, the same procedure runs on
   each split's query set, which plays the role of test.csv, so the CV number already includes its effect.
2. *What does α mean in label propagation?* The weight on neighbours vs a node's own prediction. α = 0 keeps the
   original probabilities; α → 1 lets the graph dominate. Leopards (big, clean groups) like α = 0.9, hyenas 0.6.
3. *Why report the nested 0.904 rather than 0.905?* The propagation settings were picked as the best of about 390 tried
   on the same 5 splits. Choosing on 4 splits and testing on the 5th is a fairer estimate of what to expect on new data.

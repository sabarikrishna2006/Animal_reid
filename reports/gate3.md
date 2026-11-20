# Gate 3: Global metric learning and identity aggregation (Stage A)

## What was built
- `src/reid/retrieval.py`:
  - `embed_unsupervised` (L2, standardise, PCA-whitening fitted on train ∪ test features; no labels)
  - `fit_lda` (regularised LDA / learned whitening, fitted on fold gallery labels only)
  - `IdentityIndex` (max / top-k mean / log-sum-exp / prototype aggregation)
- `scripts/run_stageA.py`: threshold-rule evaluation of each variant on the 5 CV splits (τ tuned per species).
- `src/reid/candidates.py`: the (query, identity) feature table builder used by the reranker. The first learned model
  was evaluated in this gate, without SIFT.
- `scripts/compare.py`: paired comparison. Accept if mean Δ > max(0.002, SE of paired per-split Δ).

## Formulas
- PCA-whitening: $\tilde x=\Lambda_d^{-1/2}U_d^\top(x-\mu)$, then L2-normalise. An eigenvalue floor $\epsilon\cdot\bar\lambda$
  is added before the inverse square root, so tiny noisy directions are not blown up.
- Regularised LDA: within-identity scatter $S_w$ (from identities with ≥ 2 gallery images) is shrunk,
  $S_w(\gamma)=(1-\gamma)S_w+\gamma\frac{\mathrm{tr}S_w}{d}I$, then we whiten by it and keep the top 256 between-identity
  directions (computed in a 512-d PCA space).

## Results (threshold rule, paired against raw-cosine max = 0.8486)
| Variant | CV MAP@5 | Δ (SE) | Decision |
|---|---|---|---|
| standardise each dim | 0.8486 | +0.000 (0.001) | reject |
| top-2 mean | 0.8367 | −0.012 (0.002) | reject |
| log-sum-exp τ=0.05 | 0.8279 | −0.021 (0.002) | reject |
| prototype (mean vector) | 0.7902 | −0.058 (0.005) | reject |
| PCA-whiten d=128 | 0.8439 | −0.005 (0.003) | reject |
| PCA-whiten d=256 | 0.8581 | +0.010 (0.002) | accept |
| **PCA-whiten d=512, ε=1** | **0.8615** | **+0.013 (0.002)** | **accept (best single)** |
| LDA γ=0.2 | 0.8566 | +0.008 (0.003) | accept |
| LDA γ=0.5 | 0.8569 | +0.008 (0.003) | accept |
| **LightGBM reranker + P(new) model, global features only** | **0.8784 ± 0.0101** | **+0.030 (0.002)** | **accept** |

Candidate recall@20 (true identity among the top 20 for known queries): raw 0.932 / 0.969 (hyena / leopard),
LDA 0.961 / 0.974.

## What surprised us
- **Max beats every "smoother" aggregation.** Averaging an identity's images (prototype) loses 0.058. One identity mixes
  left and right flanks, whose average looks like neither, so the single best-matching image is the right evidence.
- **Whitening lowers pair AUC (EDA) but raises MAP.** AUC needs one global threshold, while MAP needs the correct
  *ordering within each query*. Whitening removes nuisance directions that rank wrong identities high for one query,
  but rescales the similarities so a single threshold fits less well.
- **LDA helps hyena (+0.024) more than leopard.** Hyenas have more images per identity (median 6), so the
  within-identity scatter is better estimated.
- The reranker's most important features were *relative* ones: z-score of the identity's similarity within the query's
  top 10, rank and gap under LDA/whitening. "How much better than the other candidates" matters more than raw cosine.

## Viva questions
1. *Why keep raw cosine, whitened and LDA all as reranker features instead of picking the best?* They make different
   mistakes (LDA is fitted on few multi-shot identities and can overfit; whitening is unsupervised). The GBDT learns
   which to trust per situation, and it beat every single one by about 0.017.
2. *Why is the LDA fitted on the fold's gallery only?* Fitting it on all train labels would use query labels, the exam
   answers, and inflate CV. At test time it is fitted on the whole training set, which matches the CV protocol.
3. *What does the z-score feature mean?* $(S_c-\bar S_{top10})/\mathrm{sd}(S_{top10})$: how many standard deviations
   identity $c$ stands out from this query's other top candidates. It corrects for queries where everything looks
   similar (blurry, odd pose).

# Leaderboard reality check and CV v2

## What the leaderboard said
| Submission | Our CV (v1) | Public LB (shown with 1 decimal, so ±0.05) |
|---|---|---|
| all `new_individual` (sample) | — | 0.3 |
| raw-cosine baseline | 0.849 | 0.7 |
| Gate 5 final | 0.905 | 0.8 |

1. **All-new = 0.3** → about 30% of public test images are new animals (each image scores 1 only if new).
2. **The baseline loses about 0.15 on LB and the final model about 0.1** → CV v1 was too easy.
3. The model's ranking of methods was right (final > baseline by about 0.1 on both CV and LB). Only the level was wrong.

## Diagnosis
- The model predicted its own test score from its probabilities, $E[AP]=\sum_k P(\ell_k)/k$: **0.90 on test**, the same as
  on CV. Test feature distributions (top-1 cosine, gaps, SIFT inliers) also matched CV closely. Same inputs, worse outcomes:
  a **label shift**, not a feature shift.
- Explanation consistent with all three LB numbers: test has *fewer* new animals than CV v1 assumed (≈ 0.3 vs 0.32–0.38),
  so to produce the same similarity distribution, test's *known* animals must be *harder* to match (lower similarity,
  different encounters). The model saw "low similarity" and said "new", which was confidently wrong. It put `new` first for
  40% of test images while only about 30% are new; each such known animal scores at most ½.
- Why v1 missed it: the KS match on the top-1 distribution has a ridge. "More new + easy known" and "fewer new + hard
  known" give similar top-1 distributions. We picked a point on the wrong end of the ridge.

## CV v2 (LB-calibrated)
`scripts/diagnose_cvlb.py` simulates the *exact* submitted baseline (same τ) and the all-new submission under 288 split
settings per species, then keeps the settings where **π ∈ [0.25, 0.35), baseline MAP < 0.75**, with the best KS:

| | encounter cut | p_move | f_new_id | π | simulated baseline |
|---|---|---|---|---|---|
| hyena | cos > 0.8 | 0.5 | 0.1 | 0.21 | 0.745 |
| leopard | cos > 0.7 | 0.5 | 0.2 | 0.37 | 0.745 |
| overall | | | | **0.30** (LB 0.3) | **0.745** (LB 0.7) |

Only three LB numbers were used, and only to set *how hard* the CV is, not to tune any model parameter.

| Model | CV v1 | **CV v2** | LB |
|---|---|---|---|
| baseline (τ re-tuned) | 0.849 | 0.755 | 0.7 |
| Gate 4 model | 0.895 | 0.832 | — |
| Gate 5 (+ propagation) | 0.905 | 0.860 | 0.8 (v1-tuned version) |

## Where MAP is lost on CV v2 (Gate 4 model, before propagation)
| | Hyena share of loss | Leopard share of loss |
|---|---|---|
| known animal, `new` ranked first | 34% | **51%** |
| known animal not in top-20 candidates | 35% | 25% |
| known animal, wrong identity first | 18% | 13% |
| new animal, not ranked first | 12% | 11% |

The core difficulty is separating a **hard known** animal (different encounter, low similarity) from a **new** one.
Candidate misses are mostly the flank problem: > 90% of them have *no same-side gallery image* and a median of 1 gallery image.

## Viva questions
1. *Your CV said 0.905, the LB said 0.8. What happened?* The CV assumed too many new animals and therefore too-easy
   known ones. Three LB numbers (all-new, baseline, final) pinned the true novelty rate and difficulty. Re-calibrating the split
   reproduces the LB level, and decisions are now made on that CV.
2. *Isn't using the LB to design the CV overfitting the LB?* We used 3 coarse numbers (1 decimal) to set 3 split
   properties. No model or threshold was tuned on the LB. Overfitting the LB means many submissions that tune parameters
   against its noise.
3. *How can a model be well calibrated on CV but overconfident on test?* Calibration only holds under the CV
   distribution. Under label shift (different P(new) and different known difficulty at the same feature values), the
   same probabilities are wrong. That is why the CV must mimic the test's label structure, not just its features.

## Experiments on CV v2 (after recalibration)
All numbers: CV v2 MAP@5 with label propagation, paired vs parent on the same 5 splits.

| Exp | Change | CV v2 | Δ (SE) | Decision |
|---|---|---|---|---|
| v2_g5 | Gate 5 pipeline re-tuned on CV v2 | 0.8603 | — | baseline for v2 |
| **v3_sift** | **SIFT on top-15 ids × 8 imgs (was 10 × 3), 30 candidates (was 20)** | **0.8670** | **+0.0067 (0.0020)** | **accept, current best** |
| v4_profile | + second-order profile similarity, within-identity spread | 0.8666 | −0.0005 (0.0008) | reject |
| v5_ratio09 | Lowe ratio 0.8 → 0.9 (pilot: +9 pts leopard SIFT recall at same precision) | 0.8677 | +0.0007 (0.0022) | reject (noise) |
| v6_xlda | + cross-flank LDA (within-scatter from left−right differences) | 0.8675 | +0.0005 (0.0010) | reject |

Submissions written for LB checking: `sub_20261119-220449_sub_v2_g5.csv` (v2) and
`sub_20261119-230723_sub_v3_sift.csv` (v3, best).

**Lesson.** Once the CV was hard enough, more SIFT *coverage* helped (the true match for a cross-encounter query is
often not among the 3 most cosine-similar images), but more SIFT *sensitivity*, second-order similarity and a
flank-invariant metric did not. The remaining loss is dominated by hard-known vs new decisions with genuinely weak evidence.

## Correction (exact leaderboard scores)
The first scores were read from a 1-decimal display. The exact public scores are:

| Submission | CV v1 | CV v2 | Public LB |
|---|---|---|---|
| all `new_individual` | — | — | **0.31113** |
| baseline | 0.849 | 0.755 | **0.76983** |
| g5_final (tuned on CV v1) | 0.905 | — | **0.84116** |
| v3_sift (tuned on CV v2) | — | 0.867 | **0.83192** |

So the real CV v1 gap was about 0.06–0.08, not 0.1–0.15. CV v2 **over-corrected**: it is harder than the real test (baseline
0.755 vs 0.770), and the model tuned on it scored 0.009 below the v1-tuned model. CV v1 is optimistic by a roughly
constant amount but ranks methods correctly. The diagnosis above (label shift, P(new) sensitivity) is still useful, but
the calibration target should have been: π ≈ 0.31, baseline ≈ 0.77, final ≈ 0.84.

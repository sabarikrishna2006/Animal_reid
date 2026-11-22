# Wildlife Re-ID (spotted hyena + leopard, MAP@5)

Course competition (Problem 3): for each test image, rank up to 5 labels (a training identity of the same species or
`new_individual`). Only the provided MiewID embeddings and SIFT keypoints may be used. **Read `CLAUDE.md` for the rules
before changing anything** (allowed libraries, cv2 restrictions, no images, submissions only via the script).

## Final submission
**`submissions/sub_20261122-171705_sub_team_blend_lnb7.csv`: public LB 0.88073** (best; submitted to Kaggle as `sub_team_blend_lnb7.csv`, produced by
`03_make_submission.py --blend`). Second selected: `sub_team_blend_lnb6.csv` (0.86749).
It is a 50/50 average of the calibrated label probabilities of two independent pipelines: this repo's
`configs/exp_v10_lnbcand_krec.json` (v8 + image-level & strict K=3 LNBNN + `lnb_img` candidate expansion + `lnb_img`-guided RANSAC SIFT + k-reciprocal re-ranking + keypoint query graph) and `dev_pipeline/` (Dev's pipeline with image-level & K=3 LNBNN + keypoint query graph + Zhou propagation). Full write-up (method, validation, results, overfitting check, rules compliance):
**[`reports/final_report.md`](reports/final_report.md)**. Official submission tag log: **[`SUBMISSIONS.md`](SUBMISSIONS.md)**.

| Component | Validation MAP@5 | Public LB |
|---|---|---|
| `exp_v8_mix` -> `+ LNBNN` -> `+ kpgraph` -> `+ lnb_img/K=3 (v9)` -> `+ lnb_img cand/SIFT + k-recip (v10)` | 0.9131 -> 0.9145 -> 0.9176 -> 0.9194 -> **0.9207** (CV v1) | - |
| `dev_pipeline` (`+ LNBNN` -> `+ lnb_img/K=3 + kpgraph + prop`) | 0.8575 -> **0.8715** (Dev's leak-free CV, 8/8 splits) | - |
| **50/50 blend = final (`lnb7`)** | **0.8764** (shared CV with Dev upgraded) | **0.88073** |

## Repository layout
| Path | What |
|---|---|
| `src/reid/` | pipeline package (splits, retrieval, SIFT matching, **lnbnn**, **kpgraph**, candidates, models, calibration, transductive, decision) |
| `scripts/` | `00_audit` -> `04_build_lnbnn` -> `02_run_cv` / `03_make_submission` (also `--blend`), plus diagnostics |
| `configs/` | one JSON per experiment; final: `exp_v8_mix_lnb_kpg.json` (+ `exp_v3_sift_lnb.json` for its extra training design) |
| `dev_pipeline/` | Dev's self-contained notebook (second half of the blend) |
| `submissions/` | every submitted CSV; final = `sub_20261121-160242_sub_team_blend_lnb3.csv` |
| `experiments/` | registry + per-run configs/metrics |
| `reports/` | `final_report.md`, EDA and gate reports |

## Leaderboard history (public, ~30% of test)
| Submission | What it is | CV it was tuned on | Public LB |
|---|---|---|---|
| `sample_submission.csv` | all `new_individual` → measures the new-animal share | — | 0.31113 |
| `sub_20261119-060645_sub_baseline_rawcos.csv` (regenerated as `sub_20261120-205711_...`) | best-cosine identity + threshold for "new" | CV v1: 0.849 | 0.76983 |
| **`sub_20261119-132154_sub_g5_final.csv`** | **full pipeline (below), tuned on CV v1** | **CV v1: 0.905** | **0.84116 (best)** |
| `sub_20261119-230723_sub_v3_sift.csv` | same pipeline + wider SIFT, tuned on CV v2 | CV v2: 0.867 | 0.83192 |
| `sub_20261120-203616_sub_v7_g5wide.csv` | g5_final + wider SIFT (15 ids × 8 imgs, 30 candidates) | CV v1: 0.9108 (nested 0.908) | 0.84640 |
| `sub_20261120-205512_sub_v8_mix.csv` | v7 + second-level models trained on CV v1 + CV v2 tables | CV v1: 0.9133 (nested 0.912) | 0.84856 |
| Dev's `sub_v5.csv` (separate pipeline, see *Team blend*) | centered-cosine + LDA retrieval, deep SIFT, 2-round LightGBM + LambdaRank | Dev's CV: 0.8532 | 0.84102 |
| **`sub_team_blend.csv`** | **50/50 average of v8 and Dev's v5 label probabilities** | — | **0.85813 (best)** |
| `sub_team_blend_prior.csv` | same, P(new) of each model prior-shifted to 31 % before averaging | — | 0.85504 |
| `sub_team_blend_v2b.csv` / `_v2.csv` | blend weights tuned on a shared validation (v2: Sabari model retrained on Dev's splits) | shared CV 0.8675 | 0.85698 / 0.85538 |
| `sub_team_blend_lnb.csv` | **Dev's model + LNBNN features** (see below), 50/50 with original v8 | shared CV 0.8675 | **0.86073** |
| `sub_team_blend_lnb2.csv` | both models + LNBNN (`exp_v8_mix_lnb` 50/50 with Dev's LNBNN model) | v8_lnb CV v1 0.9145 | 0.86164 |
| `sub_team_blend_pseudo.csv` | lnb2 + self-training (confident test images added to the gallery) | sim. +0.0025 | 0.85829 (dropped) |
| `sub_team_blend_lnb3.csv` | lnb2 + keypoint query graph in label propagation (`exp_v8_mix_lnb_kpg`) | CV v1 0.9176 | 0.86515 |
| `sub_team_blend_lnb6.csv` | both pipelines + image-level & strict K=3 LNBNN + Dev keypoint graph + Zhou prop (`exp_v9_mix_lnbimg_kpg`) | CV v1 0.9194 / shared CV 0.8764 | 0.86749 |
| **`sub_team_blend_lnb7.csv`** | **+ `lnb_img` candidate expansion + `lnb_img`-guided RANSAC SIFT + k-reciprocal re-ranking (`exp_v10_lnbcand_krec`) — FINAL** | **CV v1 0.9207** | **0.88073 (best)** |

Read it carefully:
- CV v1 overestimates the LB by about 0.06–0.08 but **ranked the methods correctly** (baseline < final on both).
- CV v2 (`reports/lb_recalibration.md`) was an attempt to make CV match the LB *level*. It was designed from
  leaderboard scores displayed with 1 decimal, which misled the calibration. It made CV harder than the real test
  (it predicted 0.755 for the baseline, real 0.770) and the v3 model tuned on it scored 0.009 *lower* on LB.
  Treat CV v2 conclusions with caution; CV v1 + LB is the better guide so far.
- 0.311 → about 31% of public test images are new animals.

## Pipeline (current best: `configs/exp_v8_mix.json`; previous LB best: `configs/exp_g5_final.json`)
1. **Embeddings** (`src/reid/retrieval.py`): MiewID vectors are *not* unit-norm, so L2-normalise. Three similarities:
   raw cosine, PCA-whitened cosine (fit on train ∪ test, no labels), regularised LDA (fit on the gallery labels of each fold).
2. **Candidates** (`candidates.py`): top-30 identities per query (v7+; was 20); identity score = best image (max beats mean/prototype).
3. **SIFT verification** (`sift_match.py`, `sift_cache.py`): RootSIFT, numpy nearest neighbours, Lowe ratio 0.8,
   mutual matches, then RANSAC partial-affine and homography (OpenCV). Pairs are cached in `data/cache`.
   v7+: SIFT checks the 8 most similar images of each of the top-15 identities (was 3 × 10): +0.0056 CV.
   High precision (≥ 6 inliers → ~98% same animal), low recall; helps leopards much more than hyenas.
4. **Reranker** (`models.py`): LightGBM on ~67 features per (query, identity): similarities, ranks, z-scores, SIFT
   aggregates, flank/side compatibility, query quality. Out-of-fold training grouped by identity.
5. **P(new)** model + softmax temperature → calibrated probabilities (ECE ≈ 0.01). Output = 5 most probable labels
   (MAP@5-optimal; `tests/test_metric.py`).
   v8: these two models are trained on the CV tables of BOTH split designs (v1 easier, v2 harder; `model.extra_cv`),
   so they see the full range of known-animal difficulty; the real test lies between the two. +0.0025 CV.
6. **Transductive** (`transductive.py`): query–query pair model (AUC 0.97), graph over test images, label propagation.
   Worth about +0.009 on CV v1.

## Team blend (current best)
Two independent pipelines (this repo's v8 and Dev's) agree on the top-1 label for only ~89 % of test images, so averaging
their calibrated label probabilities (simple model combination, allowed) beats both on the public LB (0.85813 vs 0.84856 / 0.84102).
Pushing new_individual up with a prior-shift correction hurt (0.85504), so the plain average is kept.
1. Every transductive run of `03_make_submission.py` now also writes `submissions/sub_<id>.probs.pkl`:
   `{test_id: {label: probability}}`, the final propagated probabilities (with the tuned new-scale), renormalised.
   Its argmax equals the first label of the submission written in the same run.
2. Dev's pipeline writes the same format (`test_probs.pkl`, temperature-calibrated on his honest CV).
3. Blend (still only through the submission script, validated like every submission):
```
py scripts/03_make_submission.py --blend submissions/sub_<v8>.probs.pkl path/to/test_probs.pkl --w 0.5
```
Next: tune `--w` and the new_individual offset on a validation set that both pipelines are run on (Dev's 8 splits).

## LNBNN keypoint matching (HotSpotter-style, current best ingredient)
`retrieval.lnbnn = true` adds per-(query, identity) LNBNN scores: every RootSIFT descriptor of the query is matched against
ALL training descriptors of its species (faiss-cpu, k=50); for each identity among the first K=10 valid neighbours the
score gains `d_background - d_identity` (background = 11th neighbour). Images outside the fold gallery and the query itself
are masked at scoring time (CV stays honest). Features: `lnb`, `lnb_rank`, `lnb_rel`, `lnb_log`, `q_lnb_max`, `q_lnb_gap`,
`q_lnb_log_max`. Effect: Dev's model 0.8532 -> 0.8583 (shared CV, 8/8 splits); v8 0.9131 -> 0.9145 (CV v1); public LB
0.85813 -> 0.86073 -> 0.86164. Classical NN keypoint matching of the provided descriptors (allowed; Crall et al. 2013).

### Keypoint query graph (`transductive.kp_graph = true`)
Label propagation originally linked each query only to its 10 MiewID-cosine neighbours. `src/reid/kpgraph.py` matches
every query image's RootSIFT descriptors against the other query images of the same fold (faiss-cpu, LNBNN-style votes)
and adds each image's 5 strongest keypoint neighbours as extra candidate edges, plus 3 pair features (`kp_qq`,
`kp_qq_sym`, `kp_qq_rank`). The existing pair model (OOF AUC 0.974 -> 0.981) decides which edges to keep. On leopards
~25 % of the new links are the same animal (links that MiewID misses). CV v1 0.9145 -> 0.9176 (5/5 splits);
public LB of the team blend 0.86164 -> 0.86515. Transductive and label-free (test features only).

### Reproduce the final submission (`sub_team_blend_lnb3.csv`)
```
py scripts/00_audit.py
py scripts/04_build_lnbnn.py                                     # or copy Dev's lnbnn/ cache into data/cache/lnbnn
py scripts/03_make_submission.py --config configs/exp_v8_mix_lnb_kpg.json  # also builds the extra_cv tables; -> sub_<id>.probs.pkl
# run dev_pipeline/wildlife_reid_solution.ipynb (from dev_pipeline/) -> dev_pipeline/test_probs.pkl
py scripts/03_make_submission.py --blend dev_pipeline/test_probs.pkl submissions/sub_<id>.probs.pkl --w 0.5
```

## Setup
```
py -m pip install -r requirements.txt          # Python >= 3.10 (Windows: use `py`, not `python`)
```
Put the Kaggle files here (`data/` is git-ignored):
```
data/reid_student_data_main/train.csv, test.csv, miewid_features.npy, miewid_ids.csv, sample_submission.csv
data/sift_features_train.npz        # the original .npz file (an extracted folder of .npy files also works)
data/sift_features_test.npz
```

## Run
```
py scripts/check_compliance.py                                   # rule check (also runs inside every script)
py scripts/00_audit.py                                           # integrity checks + packs SIFT into data/cache (~2 min)
py scripts/02_run_cv.py --config configs/exp_v3_sift.json        # CV v2 tables (needed by v8's mixed training)
py scripts/02_run_cv.py --config configs/exp_v8_mix.json         # CV + experiment log (first run builds SIFT cache: 1-2 h)
py scripts/03_make_submission.py --config configs/exp_v8_mix.json     # writes submissions/sub_<id>.csv (validated)
py scripts/compare.py PARENT_EXP CHILD_EXP                       # paired comparison of two CV runs
py -m unittest discover tests
```
Every run writes `experiments/<exp_id>/` and a row in `experiments/registry.csv`. Configs are JSON and override
`configs/base.json`. The CV split is defined by the `cv` block (v1 = base.json, v2 = `configs/cv2.json`).

## Repository map
| Path | What |
|---|---|
| `src/reid/` | the package: io, splits, metrics, retrieval, sift_match, sift_cache, candidates, models, calibration, pipeline, transductive, decision, tracking |
| `scripts/` | CLI entry points, plus diagnostics (`diagnose_cvlb.py`, `error_breakdown.py`, `nested_check.py`, `tune_models.py`, `gate4_analysis.py`) |
| `configs/` | one JSON per experiment |
| `reports/` | `eda.md`, `gate1.md`…`gate5.md`, `lb_recalibration.md` (each with viva questions) |
| `experiments/` | registry + per-run metrics/configs/logs |
| `eda/` | exploratory scripts behind `reports/eda.md` |
| `PLAN.md`, `Wildlife Re-ID Architecture Plan.md` | the original design documents |

## Where to go next (open problems)
1. **Submit v7 and v8** and check the gains (+0.005 / +0.008 on CV) appear on the LB (expected about 0.845–0.85).
2. **Hard known vs new** is still the biggest loss: known animals seen in a different encounter (low cosine, no SIFT
   match) get `new_individual` ranked first. `scripts/error_breakdown.py SLUG` shows the breakdown.
3. CV v1 is about 0.065 optimistic but ranks methods correctly. CV v2 is too hard. Mixing both for *training* helped;
   a better single CV design between them is still open.
4. Tried with no gain: second-order profile similarity, looser SIFT ratio (0.9), cross-flank LDA, viewpoint imputer,
   stacking LGB+XGB+CatBoost, LightGBM hyper-parameters, mean/prototype aggregation, consensus filter on the graph.

Compute notes: SIFT RANSAC is CPU-only (OpenCV); GPU libraries (torch, cupy, faiss-gpu) are not allowed. On Windows,
long background runs can be throttled; raising process priority helps.

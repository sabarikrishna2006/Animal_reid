# Wildlife Re-ID (spotted hyena + leopard, MAP@5)

Course competition (Problem 3): for each test image, rank up to 5 labels (a training identity of the same species or
`new_individual`). Only the provided MiewID embeddings and SIFT keypoints may be used. **Read `CLAUDE.md` for the rules
before changing anything** (allowed libraries, cv2 restrictions, no images, submissions only via the script).

## Leaderboard so far (public, ~30% of test)
| Submission | What it is | CV it was tuned on | Public LB |
|---|---|---|---|
| `sample_submission.csv` | all `new_individual` → measures the new-animal share | — | 0.31113 |
| `sub_20261119-060645_sub_baseline_rawcos.csv` (regenerated as `sub_20261120-205711_...`) | best-cosine identity + threshold for "new" | CV v1: 0.849 | 0.76983 |
| **`sub_20261119-132154_sub_g5_final.csv`** | **full pipeline (below), tuned on CV v1** | **CV v1: 0.905** | **0.84116 (best)** |
| `sub_20261119-230723_sub_v3_sift.csv` | same pipeline + wider SIFT, tuned on CV v2 | CV v2: 0.867 | 0.83192 |
| `sub_20261120-203616_sub_v7_g5wide.csv` | g5_final + wider SIFT (15 ids × 8 imgs, 30 candidates) | CV v1: 0.9108 (nested 0.908) | *not yet submitted* |
| **`sub_20261120-205512_sub_v8_mix.csv`** | **v7 + second-level models trained on CV v1 + CV v2 tables** | **CV v1: 0.9133 (nested 0.912)** | ***not yet submitted*** |

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

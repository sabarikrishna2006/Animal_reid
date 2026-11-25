# Wildlife Re-ID (spotted hyena + leopard, MAP@5)
(Problem 3): for each test image, rank up to 5 labels (a training identity of the same species or
`new_individual`). Only the provided MiewID embeddings and SIFT keypoints may be used. **Read `CLAUDE.md` for the rules
before changing anything** (allowed libraries, cv2 restrictions, no images, submissions only via the script).





## Repository layout
| Path | What |
|---|---|
| `src/reid/` | pipeline package (splits, retrieval, SIFT matching, **lnbnn**, **kpgraph**, candidates, models, calibration, transductive, decision) |
| `scripts/` | `00_audit` -> `04_build_lnbnn` -> `02_run_cv` / `03_make_submission` (also `--blend`), plus diagnostics |
| `configs/` | one JSON per experiment; |
| `dev_pipeline/` |notebook 
| `submissions/` | 
| `experiments/` | registry + per-run configs/metrics |
| `reports/` | `final_report.md`, EDA and gate reports |







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

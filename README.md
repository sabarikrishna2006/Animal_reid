# Wildlife Re-ID (spotted hyena + leopard, MAP@5) — Group 53 (Team NeuerXjensen)
(Problem 3): for each test image, rank up to 5 labels (a training identity of the same species or
`new_individual`). Only the provided MiewID embeddings and SIFT keypoints may be used. **Read `PLAN.md` for the rules
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







## Final Submission & Team Blend (Rank 1 — `0.88690`)
**`sub_team_blend_lnb39_swproto_g2q_mlp.csv` (`sub-v31-2026-11-27-blend-lnb39`): Public LB `0.88690` (Rank 1)** (second selected submission: `sub_team_blend_lnb38_v28_g2q_safe.csv`, `sub-v30-2026-11-27-blend-lnb38`, Public LB `0.88605`; third: `sub_team_blend_lnb36_sw2152_v25.csv`, `sub-v29-2026-11-26-blend-lnb36`, Public LB `0.88585`).
Official submission tag log: **[`SUBMISSIONS.md`](SUBMISSIONS.md)**. Full report: **[`reports/final_report.md`](reports/final_report.md)**.

### Individual Contributions (Group 53 — Team NeuerXjensen)
| Team Member | GitHub Account | Key Contributions |
|---|---|---|
| **Sabari Krishna R** | `sabarikrishna2006` | Repository setup, compliance & leakage unit test suite (`tests/`), Exploratory Data Analysis (`eda/`, `reports/eda.md`), CV-vs-LB recalibration & error diagnostics (`scripts/diagnose_cvlb.py`, `error_breakdown.py`, `mix_cv.py`, `reports/gate1..5.md`), and the core `src/reid/` pipeline (encounter-disjoint open-set CV v1/v2, retrieval, SIFT RANSAC, monotone LightGBM reranker, softmax calibration, and transductive propagation powering `sub-v01..v04` and the 5-model `src/reid/` ensemble `v17..v26` in the final blend). |
| **Devyaansh** | `devyaansh-9` | Independent `dev_pipeline/` (2-round neighbor-stacked LightGBM, `wildlife_reid_solution.ipynb`), HotSpotter LNBNN (`src/reid/lnbnn.py`), keypoint query graph (`kpgraph.py`), `lnb_img` candidate/SIFT expansion & k-reciprocal re-ranking (`v9`/`v10`, `sub-v05..v11` up to **`0.88073`**); dual-pool `dev_bg_sw2152` + `exp_v25` + `make_team_blend.py` (`sub-v18..v24`, `sub-v29` **`0.88585`**); `Sw2152` centroid prototypes/DBA (`sw_proto`), non-tree Neural Net (`MLPClassifier`), `sw_qq` learned propagation (`sub-v31` Rank 1 **`0.88690`**), Private-LB anti-overfitting verification, and the final report notebook (`Team_NeuerXjensen_Final_Report.ipynb`). |
| **Jazl Anwar MV** | `jazpeak` | Read and researched animal re-ID papers by **Prof. Saket Anand** and related metric-learning literature, running experiments that guided team feature design; Stage-2 full-frame camera-trap SIFT filtering, gallery hubness (`r_g_diff`), missing-candidate veto fix & learned query-query graph propagation (`lprop`); gallery-only linear metric learning (`PCA(512)-LDA`, `Side-LDA`, and full 2152-dim `Sw` whitening `sw2152`, reaching Rank 1 **`0.88424`** in `sub-v26`); `v27` research-paper experiments (identity prototypes, gallery aggregation, and cross-flank alignment, `sub-v27`); `v28` gallery-to-query competitor features (`G2Q`) and safe `new_individual` gating (`sub-v30` Rank 1 **`0.88605`**, integrated into the final **`0.88690`** blend); blend ablations (`sub-v25`, `sub-v28`, `sub-v30a`), `dev_pipeline` helper modules (`core.py`, `model2.py`, `exp_blend_check.py`), and report documentation. |
| **Lokesh Akurathi** | `Lokesh-akurathi` | Dual-pipeline `bg_clean` upgrades (`nprobe=64` LNBNN with `lnb_uimg`/`lnb_rimg` background-keypoint masking, Stage-1 corner camera-trap SIFT filter, and `conn_new_scale` isolated-node propagation fix driving `sub-v12..v17` up to **`0.88101`**); standalone Kaggle notebooks (`wildlife_reid_v4` `sub-v20`, `v25_v26_notebook.ipynb`), `v25` HNSW rescue + `v26` dual-model pipeline (`src/reid/dual_model.py`, `scripts/eval_blend.py`, `tests/test_v25_and_dual.py`), and experimental `alt_approach_v4.py` / `wildlife_reid_alt_approach.md`. |

Our final solution blends the temperature-calibrated label probability distributions of two independent, strictly zero-leakage pipelines:
- **Pipeline A — Single-Round Monotone Ensemble (`50%`, `src/reid/`)**: Equal 5-model monotone LightGBM ensemble of `exp_v17_bgclean_conn`, `exp_v19_bgclean_v2`, `exp_v24_fullbg_sw2152`, `exp_v25_fullbg_sw2152_lda512`, and `exp_v26_fullbg_sw2152_proto`.
- **Pipeline B — Two-Round Stacked Dual-Pool Ensemble (`50%`, `dev_pipeline/`)**: Equal 50/50 dual-pool blend of `dev_bg_swproto_g2q` (`LGB + MLP`, `sw_qq` `lprop`) and `dev_v10_swproto_g2q` (`LGB + MLP`, `sw_qq` `lprop`).

### Key Zero-Leakage Discoveries
1. **Two-Stage Camera-Trap SIFT Background Filtering (`bg_clean` & SIFT Killers)**:
   - **`nprobe=64` LNBNN Background-Keypoint Masking (`lnb_uimg`, `lnb_rimg`)**: Masks static camera-trap background keypoints that match across multiple unrelated gallery identities before scoring.
   - **Stage-1 Corner & Stage-2 Full-Frame Camera-Trap SIFT Killers**: Zeroes out false-positive RANSAC SIFT geometric matches caused by static camera-trap overlays, timestamps, and rigid background foliage (`abs_rot < 0.50° & spread_min < 0.25` and full-frame identity-agnostic static keypoint filters).
2. **Missing-Candidate Veto Fix & Gallery-Only Hubness (`r_g_diff`)**:
   - Eliminates false veto penalties when an identity is absent from a secondary candidate pool and penalizes gallery hubs using strictly gallery-fitted neighborhood statistics (`r_g_diff`).
3. **Strictly Nested Learned BG-Filtered Query–Query Graph Propagation (`lprop`, OOF AUC `0.9777`) & `conn_new_scale`**:
   - Replaces hand-tuned query–query similarity heuristics with a strictly nested out-of-fold pairwise classifier over MiewID + background-filtered RootSIFT keypoint graph edges, paired with degree-aware `new_individual` scaling (`conn_new_scale`) so isolated test nodes are not over-dampened during label propagation.
4. **Gallery-Only Linear Metric Learning (`PCA(512)-LDA`, `Side-LDA`, `sw2152`)**:
   - Fits `PCA(512)-LDA`, viewpoint-conditioned `Side-LDA`, and full 2152-dim within-class covariance whitening (`Sw^{-1/2}`) strictly on fold-gallery identity labels with zero test-label leakage.

### Blending & Reproduction
1. Every transductive run of `03_make_submission.py` writes `submissions/sub_<id>.probs.pkl`:
   `{test_id: {label: probability}}`, the final propagated probabilities (with the tuned new-scale), renormalised.
2. The `dev_pipeline/` modules write the same calibrated probability dictionary format.
3. Blend (validated through the submission script):
```
py scripts/03_make_submission.py --blend submissions/sub_<pipeline_a>.probs.pkl path/to/pipeline_b_probs.pkl --w 0.5
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

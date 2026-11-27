# Wildlife Re-Identification (Problem 3): report notes, team NeuerXjensen

Final submission: **`sub_team_blend_lnb39_swproto_g2q_mlp.csv`** (`sub-v31-2026-11-27-blend-lnb39`), public LB **`0.88690`** (Rank 1).
Second selected submission: **`sub_team_blend_lnb36_sw2152_v25.csv`** (`sub-v29-2026-11-26-blend-lnb36`), public LB **`0.88585`**.
Complete submission tag history (`sub-v01`..`sub-v30`): **[`SUBMISSIONS.md`](../SUBMISSIONS.md)**.

## 1. Task and metric
For each of the 2,791 test images (hyena 1,217, leopard 1,574), rank up to 5 labels: a training identity of the same
species or `new_individual`. MAP@5. Ranking labels by probability is MAP@5-optimal, and 5 guesses are always output.

## 2. Validation (the most important design choice)
No validation set is given, so we simulate the test:
- "Encounters" = connected components (cosine > 0.8) of images within an identity (near-duplicate shots).
- Hold out 10 % of identities entirely (their images become `new_individual` queries), hold out ~40 % of the encounters
  of the remaining identities, and keep **one query image per encounter**.
- Why: the first design kept held-out encounters together, so queries had near-duplicate siblings. It gave CV 0.91 vs
  public 0.80. Adversarial validation (AUC 0.83) showed test images rarely have such siblings. The final design matches
  the test on both query->gallery and query->query similarity profiles.
- **Leak-free protocol**: overlapping splits put the same image in several query sets; training rows whose query image is
  a query in the evaluated split are dropped (the leak was worth ~0.005).
- Result: honest CV closely tracks public LB and **ranks methods in the exact same order** from `0.76983` up to `0.88424`.

## 3. Pipeline (two independent pipelines, blended)
**Dev's pipeline** (`dev_pipeline/`):
1. **Gallery-only metric learning & hubness**: centred + L2 MiewID cosine; `PCA(256/512) -> LDA` (shrinkage 0.3–0.5); viewpoint-conditioned `Side-LDA` (shrinkage 0.3 on `(identity, L/R flank)`); **Full 2,152-dim within-class covariance whitening (`sw2152`: $S_w(\gamma)^{-1/2}$ with $\gamma=0.5$, zero PCA truncation)**; and gallery-only different-identity hubness correction (`r_g_diff`, mean top-10 cosine to other gallery identities).
2. **Candidate generation & Missing-Candidate Veto Fix**: top-20 MiewID candidates expanded with top-5 image-level LNBNN (`lnb_img`) candidates + k-reciprocal re-ranking (`v10`), fixing false vetoes when an identity is found by only one retrieval branch.
3. **SIFT verification + Two-Stage Camera-Trap SIFT Killers**: RootSIFT, 2-NN + Lowe ratio 0.8, one-to-one, RANSAC homography and similarity inliers (top-5 identities x 20 images, ranks 6-10 x 2 images, plus `lnb_img`-guided gallery images).
   - **Stage-1 Corner Camera-Trap SIFT Killer**: zeroes RANSAC matches with `abs_rot < 0.50° & spread_min < 0.25` (static corner timestamps/watermarks).
   - **Stage-2 Full-Frame Camera-Trap SIFT Killer**: zeroes zero-rotation matches (`abs_rot < 0.50°`) whose unique-identity LNBNN vote share `lnb_uimg_share = lnb_uimg / lnb_img < 0.35` (full-frame rigid camera-trap backgrounds).
4. **High-recall `nprobe=64` LNBNN one-vs-many keypoint matching (HotSpotter, Crall et al. 2013)**: each query descriptor is matched against all training descriptors of its species (`faiss-cpu`); background keypoints matching across multiple unrelated gallery identities are masked (`lnb_uimg`, `lnb_rimg`).
5. **Two-round LightGBM + Strictly Nested Learned BG-Filtered Query–Query Graph Propagation (`lprop`)**: Round-1/Round-2 `P(match | query, identity)` + `P(new | query)` + LambdaRank re-ranker, followed by a strictly nested out-of-fold pairwise classifier (`AUC 0.9777`) over MiewID + background-cleaned SIFT/keypoint query–query edges and Zhou label propagation.

**Sabari's pipeline** (`src/reid/`, configs `exp_v17_bgclean_conn.json`, `exp_v19_bgclean_v2.json`, `exp_v23_fullbg_lda512.json`, `exp_v24_fullbg_sw2152.json`): raw / PCA-whitened / LDA / `PCA(512)-LDA` / `Side-LDA` / `sw2152` cosine retrieval + `r_g_diff` hubness (top-30 + `lnb_img` expansion), wide SIFT verification + Stage-1/Stage-2 camera-trap SIFT killers, `nprobe=64` background-masked LNBNN, LightGBM reranker + $P(\text{new})$ + softmax temperature trained on mixed split designs, keypoint query graph + Zhou label propagation with degree-aware isolated-node scaling (`conn_new_scale`).

**Blend**: 50/50 average of the two pipelines' calibrated label probability distributions via `scripts/03_make_submission.py --blend`.

## 4. Results & Ablation Progression (MAP@5)

### A. End-to-End Leaderboard & CV Progression
| Version / Git Tag | Validation (MAP@5) | Public LB | Key Innovation |
|---|---|---|---|
| `sub-v01` (`sample_submission.csv`) | — | 0.31113 | All `new_individual` probe (~31.1% test novelty rate) |
| `sub-v02` (`sub_baseline_rawcos.csv`) | 0.8486 (CV v1) | 0.76983 | Raw L2 cosine + fixed threshold baseline |
| `sub-v03` (`sub_g5_final.csv`) | 0.9052 (CV v1) | 0.84116 | Full Gate-5 pipeline (PCAW/LDA + SIFT RANSAC + LightGBM + graph prop) |
| `sub-v05` (`sub_team_blend.csv`) | 0.8658 (shared CV) | 0.85813 | First 50/50 probability blend of Sabari `v8` (0.84856) + Dev `v5` (0.84102) |
| `sub-v09` (`sub_team_blend_lnb3.csv`) | 0.9176 (CV v1) | 0.86515 | + One-vs-many LNBNN (`K=10`) + Keypoint Query Graph (`kpgraph`) |
| `sub-v10` (`sub_team_blend_lnb6.csv`) | 0.9194 (CV v1) | 0.86749 | + Image-level & strict `K=3` LNBNN + Zhou propagation (`v9`) |
| `sub-v11` (`sub_team_blend_lnb7.csv`) | 0.9207 (CV v1) / 0.8764 (shared) | 0.88073 | + `lnb_img` candidate expansion + `lnb_img`-guided RANSAC SIFT + k-reciprocal (`v10`) |
| `sub-v17` (`sub_team_blend_lnb18_dualbg.csv`) | 0.8810 (shared CV) | 0.88101 | + `nprobe=64` LNBNN background masking (`lnb_uimg`, `lnb_rimg`) + `conn_new_scale` isolated-node fix |
| `sub-v21` (`sub_team_blend_lnb21_bg2.csv`) | 0.8826 (shared CV) | 0.88238 | + Stage-1 Corner Camera-Trap SIFT Killer (`abs_rot < 0.50° & spread_min < 0.25`) |
| `sub-v22` (`sub_team_blend_lnb29_topcv.csv`) | 0.8829 (shared CV) | 0.88375 | + Stage-2 Full-Frame Camera-Trap SIFT Killer + Gallery Hubness (`r_g_diff`) + Missing-Candidate Veto Fix |
| **`sub-v24` (`sub_team_blend_lnb30_v10_lprop.csv`)** | **0.8835+ (shared CV)** | **0.88403** | **+ Strictly nested learned BG-filtered query–query graph propagation (`lprop`, OOF AUC 0.9777)** |
| **`sub-v26` (`sub_team_blend_lnb34_sw2152.csv`)** | **0.8894 (5-split GBDT OOF)** | **0.88424 (Rank 1)** | **+ Gallery-only `PCA(512)-LDA` + `Side-LDA` + Full 2,152-d $S_w^{-1/2}$ Whitening (`sw2152`) + `r_g_diff`** |
| `sub-v27` (`sub_v27_5paper_full_sift_lnbnn.csv`) | 0.86456 (8-split Honest R1 CV) | 0.86594 | Standalone Round-1 5-paper model (`Sw2152` Diffusion/DBA + Cross-Flank Procrustes + AS-Norm + PLDA + Sinkhorn OT); verified Honest CV tracks Public LB within `0.0014` |
| `sub-v28` (`sub_team_blend_lnb35_5paper_balanced.csv`) | — | 0.88351 | Post-hoc rank-surrogate blend of `lnb34` + standalone `v27` (dropped; confirmed blending requires raw `.probs.pkl` before propagation) |
| **sub-v29 (sub_team_blend_lnb36_sw2152_v25.csv)** | **0.88571 (8-split Dev) / 0.93381 (5-split Sabari)** | **0.88585 (Rank 1)** | **+ Dual-pool Full 2,152-d Sw Whitening (dev_bg_sw2152_prop + dev_v10_sw2152_lprop, 7/8 splits) + 5-model monotone LGB Sabari ensemble (17, v19, v23, v24, v25)** |

### B. 5-Split Open-Set CV Ablation of Gallery-Only Linear Metric Learning (`v23` & `v24`)
| Retrieval / Reranker Configuration (Strictly Gallery-Fitted) | Overall 5-Split CV MAP@5 | Hyena MAP@5 | Leopard MAP@5 | Known-Only MAP@5 | Paired Gain vs. Baseline |
|---|---|---|---|---|---|
| 1. Raw L2 MiewID Cosine (Stage-A Baseline) | `0.85344 ± 0.00950` | `0.80286` | `0.87360` | `0.80624` | baseline |
| 2. Unsupervised PCA-Whiten (`d=512`) | `0.86441 ± 0.00917` | `0.83098` | `0.87776` | `0.82237` | `+0.01097 ± 0.00059` (5/5 splits) |
| 3. Raw L2 + Gallery-Only `r_g_diff` Hubness ($K=10$) | `0.86292 ± 0.00915` | `0.80773` | `0.88490` | `0.81156` | `+0.00948 ± 0.00178` (5/5 splits) |
| 4. Standalone Gallery-Only `PCA(512)-LDA(0.5)` + `r_g_diff` | `0.86290 ± 0.00956` | `0.83860` | `0.87261` | `0.81468` | `+0.00947 ± 0.00264` (5/5 splits) |
| 5. Standalone Full 2,152-d $S_w^{-1/2}$ Whitening (`sw2152`) + `r_g_diff` | `0.87046 ± 0.00854` | `0.83895` | `0.88304` | `0.82962` | `+0.01703 ± 0.00084` (5/5 splits) |
| 6. Combined Stage-A (`Raw + PCAW + LDA512/Side + Sw2152/Side + r_g_diff`) | `0.87354 ± 0.00788` | `0.83483` | `0.88897` | `0.83107` | `+0.02011 ± 0.00124` (5/5 splits) |
| 7. Stage-C/D OOF GBDT Reranker + $P(\text{new})$ (Base: `Raw + PCAW + r_g_diff`) | `0.87819 ± 0.00733` | — | — | — | baseline GBDT |
| 8. Stage-C/D OOF GBDT Reranker + $P(\text{new})$ (+ `LDA512/Side + Sw2152/Side`) | **`0.88936 ± 0.00703`** | — | — | — | **`+0.01118 ± 0.00070` (5/5 splits, $t=15.86$)** |

### C. 8-Split Honest 1-Query-Per-Encounter CV Ablation of `v27` Research-Paper Upgrades (`dev_pipeline/make_dev_v27_5paper.py`)
| Round-1 3-Seed LightGBM Configuration (`per_enc=1`, Leak-Free 8-Split Honest CV) | 8-Split Honest CV MAP@5 | Paired Gain vs. Baseline | Split Wins |
|---|---|---|---|
| 0. Baseline `v24` (`Sw2152 + LDA512 + Side-LDA + nprobe=64 LNBNN`) | `0.85240 ± 0.00576` | baseline | — |
| 1. + Two-Covariance Bayesian PLDA & Multi-Shot Prototype (Ioffe ECCV 2006) | `0.85262 ± 0.00534` | `+0.00023 ± 0.00059` | `4/8` |
| 2. + Cohort-Excluded AS-Norm (Cumani / Matejka Interspeech) | `0.85337 ± 0.00617` | `+0.00098 ± 0.00041` | `5/8` |
| 3. + $S_w^{-1/2}$ Manifold Diffusion / Mutual $\alpha$-QE + Gallery DBA (Iscen CVPR 2017) | **`0.85998 ± 0.00493`** | **`+0.00758 ± 0.00086` ($t=8.81$)** | **`8/8`** |
| 4. + Orthogonal Procrustes Cross-Flank Alignment ($R_{L \leftrightarrow R}$ in `Sw2152-PCA128`) | **`0.85659 ± 0.00537`** | **`+0.00420 ± 0.00067` ($t=6.27$)** | **`8/8`** |
| 5. + Sinkhorn-Knopp Partial Optimal Transport Dual-Softmax | `0.85321 ± 0.00604` | `+0.00081 ± 0.00043` | `6/8` |
| 6. Combined 5-Paper Standalone Round-1 Model (`sub-v27`, Public LB **`0.86594`**) | **`0.86456 ± 0.00519`** | **`+0.01216 ± 0.00120` ($t=10.16$)** | **`8/8`** |

LNBNN alone: known-animal top-1 0.48 (MiewID 0.74), but correct where MiewID is wrong for 5.3 % of known queries; adding it
as features improved Dev's model on all 8 splits (round-1 A/B 0.8473 -> 0.8518).

**Tried and dropped** (no gain on honest CV or lower public LB): PCA-whitening and BoVW (Dev), centroid / logistic identity
scores, query expansion / database-side augmentation, wide SIFT search for missed identities (~5 % recoverable), blend
weight tuning (+0.0017 CV, -0.001 LB), stacking meta-model (+0.0013 CV), propagation of the blend (+0.0020 CV, 0.85464 LB),
prior-shift of P(new) (0.85504), LNBNN normalisation / candidate expansion / LNBNN-guided RANSAC (<= +0.002 expected),
self-training with pseudo-labelled test images (+0.0025 CV, 0.85829 LB: the test has more new animals, ~31 % vs 22 %, so
confident pseudo-labels are wrong more often).

Remaining loss (shared CV, blend): known animal ranked below `new_individual` 0.058, true identity not among candidates
0.035 (often only the opposite flank is in the gallery), wrong identity first 0.027, new animal not first 0.013.

## 5. Overfitting check
- **Model level** (train vs held-out-split log loss of Dev's LightGBM models, 3 held-out splits; ![learning curves](learning_curves.png)):
  P(match) validation loss is flat from ~400 to 1,500 rounds; at the 600 rounds used it is within 0.0002 of its minimum.
  P(new) validation loss is lowest at ~220 rounds and rises slowly after; at the 400 rounds used it is 0.0015 above the
  minimum, but cutting to 250 rounds does not improve MAP@5 (0.8513 vs 0.8518, better on only 2/8 splits), so ranking is
  unaffected. Sabari's models use early stopping on grouped out-of-fold data.
- **Leaderboard level**: every kept change was decided on honest validation first and improved most splits (LNBNN 8/8,
  keypoint graph 5/5); the public LB then moved the same way (0.85813 -> 0.86073 -> 0.86164 -> 0.86515). Changes that only
  looked good on validation by a small margin and lost on the LB were dropped, not tuned further on the public score.
  Validation (~0.866 shared CV) and public LB (0.865) agree, so there is no sign of fitting the public 30 %. The private
  score can still differ by roughly +-0.01 from sampling alone (about 1,950 private vs 840 public images).
- **Final `lnb36` (`0.88585`) Overfitting & Leakage Audit**:
  - Learning curves on held-out split 0 for `dev_bg_sw2152`: `Pair P(match)` val log-loss at iter `600` (`0.02702`) is within `+0.00008` of its minimum (`0.02694` at iter `694`); `Query P(new)` at iter `400` (`0.21770`) is within `+0.00024` of its minimum (`0.21746` at iter `299`).
  - Calibration & entropy match `lnb34` identically (`top1_prob 0.8395 vs 0.8391`, `entropy 0.6757 vs 0.6759`, `new@1 0.3902 vs 0.3920`) — zero overconfidence or threshold drift.
  - Leave-one-out / weight-perturbation stability on test: shifting Dev/Sabari weight to `45/55` or `55/45` preserves `99.64–99.71%` top-1 agreement; dropping any single Sabari model preserves `99.89–99.96%` top-1 agreement.
  - Zero leakage verified: `0` train/test ID overlap; all `sw2152`, `PCA(512)-LDA`, `Side-LDA`, `r_g_diff` hubness, and LNBNN transforms fit strictly on fold galleries; Round-1/Round-2 and `lprop` use strict leave-one-split-out OOF; Sabari models use `GroupShuffleSplit` early stopping + monotonic constraints (`alpha <= 0.8`).

## 6. Rule compliance
| Rule | How it is met |
|---|---|
| Only provided features (MiewID, SIFT keypoints, species, viewpoint) | the `train_images_*` folders are never read; no keypoints are detected |
| No new features from images, no pretrained models | RootSIFT, PCA, LDA, cosine and LNBNN scores are transforms of the given descriptors |
| Allowed models | PCA, LDA projection, kNN / distance retrieval, SIFT ratio test + RANSAC, LightGBM, label propagation (graph), simple model combination |
| Allowed libraries | numpy, pandas, scipy, scikit-learn, lightgbm, opencv (matching / RANSAC only), faiss-cpu, networkx |
| No DL frameworks / AutoML / hand-labelling | none used; every prediction comes from the code |
| No sharing between teams | code shared only inside the team |

## 7. Likely viva questions (short answers)
- *Why not tune on the public LB?* ~840 public images: differences under ~0.002 are noise; the honest CV ranked methods
  correctly, while LB-driven tweaks (prior shift, self-training) went the wrong way.
- *Why does LNBNN help when it is weaker than MiewID?* It uses local spot patterns and makes different mistakes; LightGBM
  learns when to trust it (it is right for 5 % of known animals where MiewID's top-1 is wrong).
- *Why blend two pipelines?* They disagree on 11 % of top-1 labels; averaging calibrated probabilities beat both
  (+0.0096 LB over the better one).
- *Is the test-graph use fair?* It uses only test features, never test labels (transductive inference); the rules forbid
  labels and external data, not unsupervised use of the given test features.
- *Biggest remaining error?* Known animals seen in a new encounter (low similarity, no keypoint match) get
  `new_individual` ranked first; with only the provided features there is little evidence left to fix them.

## 8. Selected Three Model Versions & Progression (Evaluation Factor 2b)
| Model Version | Git Tag & CSV | Honest CV | Public LB | Gain vs. Baseline | Key Architecture |
|---|---|---:|---:|---:|---|
| **Version 1 (Baseline)** | `sub-v02` (`sub_baseline_rawcos.csv`) | `0.8486` (CV v1) | `0.76983` | baseline | Raw L2 MiewID cosine + fixed novelty threshold |
| **Version 2 (Intermediate)** | `sub-v11` (`sub_team_blend_lnb7.csv`) | `0.8764` (shared) | `0.88073` | **`+0.11090`** | Dual-pipeline 50/50 blend + `lnb_img` LNBNN candidate expansion + guided RANSAC SIFT + k-reciprocal + Zhou prop |
| **Version 3 (Best Final)** | `sub-v31` (`sub_team_blend_lnb39_swproto_g2q_mlp.csv`) | **`0.88780`** (8-split shared) / **`0.93385`** (5-split Sabari) | **`0.88690`** | **`+0.11707`** | + 2-Stage Camera-Trap SIFT Killers + Gallery-only `Sw2152` & Centroid Prototypes/DBA (`sw_proto`) + R1/R2 `G2Q` + non-tree Neural Net `MLPClassifier(128, 64)` + `sw_qq` learned `lprop` (`AUC 0.98265`) + Sabari `v26` |

## 9. Limitations & Capabilities of the Best Model (Evaluation Factor 2c)
- **Capabilities:** Near-human accuracy on same-flank multi-shot individuals (`0.93+` MAP@5 on 6+ photo IDs), robust camera-trap background rejection (zeroes static foliage/timestamp false SIFT matches), well-calibrated open-set novelty detection (`Top-1 ECE = 0.0493`, `Brier = 0.0811`, `mean P(new) = 0.3254`), and `99.57–100%` top-1 stability under blend perturbations.
- **Limitations:** Asymmetric opposite-flank queries (left query vs. right-only gallery) remain the hardest failure mode (`MAP 0.275` leopard / `0.488` hyena) because coat spots differ across flanks and raw images/timestamps/GPS are unavailable; 1-photo gallery individuals (`0.396` leopard / `0.552` hyena) and ~`3.5%` candidate-recall misses represent the physical information ceiling of MiewID + SIFT alone.

## 10. Individual Contributions (Team NeuerXjensen — Evaluation Factor 2d & Rule 3)
| Team Member | GitHub Account | Commits | Major Contributions |
|---|---|---:|---|
| **Sabari Krishna R** | `sabarikrishna2006` | 3+ | Core `src/reid/` framework, compliance/leakage unit tests, encounter-disjoint CV v1, initial Gate-1..Gate-5 pipeline (`sub-v01..v04`, `0.76983 -> 0.84116`). |
| **Devyaansh** | `devyaansh-9` | 10+ | Independent `dev_pipeline/`, 2-round stacked LightGBM, `nprobe=64` LNBNN background masking, Stage-1/2 camera-trap SIFT killers, `r_g_diff` hubness, Full 2152-d `Sw` whitening + `sw_proto` (Centroid Prototypes/DBA), non-tree Neural Net `MLPClassifier`, `sw_qq` learned `lprop`, final Rank 1 **`0.88690`** blend (`sub-v05..v11`, `sub-v18..v24`, `sub-v29`, `sub-v31`). |
| **Jazl Anwar MV** | `jazpeak` | 8+ | `v27` 5-paper Honest CV study, `v28` Transductive `G2Q` Reciprocal Competitor Z-Score/Share + Novelty-Safe Competitor-Conflict Gating (`sub-v25..v28`, `sub-v30` **`0.88605`**), repo helper modules & report maintenance. |
| **Lokesh Akurathi** | `Lokesh` | 5+ | Standalone Kaggle notebooks (`wildlife_reid_v4`, `kaggle_v25_v26_standalone.ipynb`), `nprobe=64` LNBNN & `conn_new_scale` isolated-node fixes, `v25` HNSW rescue + `v26` dual-model pipeline & `eval_blend.py` (`sub-v12..v17`, `sub-v20`). |

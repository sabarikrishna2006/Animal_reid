# CLAUDE.md — Wildlife Re-Identification (spotted hyena + leopard, MAP@5)

You are the execution engine for a course Kaggle competition. The architecture is already decided
(see "Wildlife Re-ID Architecture Plan"). Your job: implement it gate by gate, measure every change on a
leak-free open-set CV, and produce verified submissions. Do not redesign the pipeline without evidence
from CV; propose changes in the gate report instead.

The student (Aman) must be able to defend every decision in a viva. So for every gate you also write a
short teaching report (see "Reporting") explaining what was done, why, and what the numbers say.

---------------------------------------------------------------------------------------------------
## 0. Hard rules (academic misconduct if broken) — read before writing any code

1. **Inputs.** Only these files: `train.csv`, `test.csv`, `miewid_features.npy`, `miewid_ids.csv`,
   `sift_features_train.npz`, `sift_features_test.npz`, `sample_submission.csv`. No external data.
   Never search for, identify or download the original images/labels or source dataset.
2. **No new image features.** We have no images; never try to recover or render them.
3. **Allowed third-party libraries only:** `numpy, pandas, scipy, scikit-learn (sklearn), statsmodels,
   imbalanced-learn (imblearn), xgboost, lightgbm, catboost, networkx, opencv-python (cv2), faiss-cpu
   (faiss), matplotlib, seaborn`. Python standard library is fine (`json, pathlib, argparse,
   concurrent.futures, multiprocessing, unittest, hashlib, time, logging, ...`).
   NOT allowed without TA approval: `joblib` (as a direct import), `pyyaml`, `pytest`, `pyarrow`,
   `tqdm`, `optuna`, `torch`, `tensorflow`, `keras`, `jax`, any AutoML (auto-sklearn, TPOT, H2O, FLAML,
   AutoGluon), any pretrained model. Configs are JSON; tests use `unittest`; caches are `.npy/.npz/.csv/.pkl`.
4. **OpenCV is for matching + geometric verification of the PROVIDED keypoints only.** Allowed cv2 API:
   `BFMatcher, BFMatcher_create, FlannBasedMatcher, DMatch, NORM_L2, findHomography,
   estimateAffinePartial2D, estimateAffine2D, findFundamentalMat, perspectiveTransform, transform,
   RANSAC, USAC_MAGSAC, USAC_DEFAULT, LMEDS, setNumThreads`. Forbidden: `SIFT_create`, any detector,
   `imread/imwrite`, any image-processing function, `cv2.dnn`.
5. **Models allowed:** linear/ridge/lasso/Bayesian regression, logistic/softmax, GLMs, LDA/QDA, naive
   Bayes, k-NN/distance retrieval, KDE, kernel methods, GPs, SVMs, metric learning with LINEAR
   projections (e.g. sklearn NCA), GMM/EM, k-means, hierarchical/graph clustering, PCA and other linear
   reductions, trees, RF, bagging, stacking, XGBoost/LightGBM/CatBoost/HistGradientBoosting,
   `sklearn.neural_network.MLPClassifier` (plain fully connected, trained from scratch), classical
   keypoint matching (NN, ratio test, RANSAC). Nothing else (no conv/recurrent/attention layers).
6. **No hand-labelling and no LLM-produced predictions.** You write code; predictions come ONLY from
   `scripts/03_make_submission.py`. Never edit a submission CSV by hand or by an ad-hoc snippet.
   Never "fix" individual test rows.
7. `scripts/check_compliance.py` must pass before every CV run and every submission (wire it into both).

If anything you want to do is not clearly allowed, stop and write the question into
`reports/questions_for_TA.md` instead of doing it.

---------------------------------------------------------------------------------------------------
## 1. Problem facts

- Train 7,163 images (`id, species, viewpoint, identity`), test 2,791 (`id, species, viewpoint`).
  Ids: `tr_*`, `te_*`. Species: `hyena`, `leopard`. Viewpoints: left, right, front, back, frontleft,
  frontright, backleft, backright, up, down; a few NaN.
- `miewid_features.npy`: 9,954 × 2,152 float32, row order = `miewid_ids.csv`. MiewID was trained with
  sub-center ArcFace → expect unit-norm vectors, compare with cosine.
- SIFT npz: `z[f"{id}_xy"]` (n×2 float32, pixels, longer side ≤ 512), `z[f"{id}_desc"]` (n×128 uint8),
  n ≤ 500; ~20% < 500, ~5% < 50, a few n = 0.
- Output per test id: 1–5 space-separated labels, best first, no duplicates, each a training identity
  OF THE SAME SPECIES or `new_individual`. Metric MAP@5 (1/k at the first correct rank k).
- The same new individual can appear several times in test. Some identities have 1–2 training images.
- Public LB = random 30% of test; private = other 70%.

---------------------------------------------------------------------------------------------------
## 2. Repository layout (create exactly this)

```
wildlife-reid/
├── CLAUDE.md
├── configs/base.json, configs/exp_*.json        # JSON only; exp files override base
├── data/raw/                                    # the 7 files, never modified
├── data/cache/                                  # aligned.npz, sift_kp/, sift_pairs.npz, splits/*.json
├── src/reid/  io.py audit.py splits.py metrics.py global_feats.py retrieval.py kreciprocal.py
│              viewpoint.py sift_match.py sift_cache.py candidates.py models.py calibration.py
│              transductive.py decision.py tracking.py
├── scripts/   00_audit.py 01_build_sift_cache.py 02_run_cv.py 03_make_submission.py check_compliance.py
├── experiments/registry.csv + experiments/<exp_id>/{config.json, metrics.json, oof.npz, notes.md}
├── reports/   eda.md gate1.md ... gate5.md figures/ questions_for_TA.md
├── submissions/sub_<exp_id>.csv
└── tests/     test_metric.py test_splits.py test_leakage.py test_compliance.py test_submission.py
```

Module boundaries:
- `io.py` is the ONLY module that reads `data/raw`. It returns aligned arrays (assert every train/test id
  is present exactly once in `miewid_ids.csv` and in the right npz) and a lazy SIFT accessor.
- `splits.py` is the ONLY module that decides gallery vs query membership.
- Every fit function receives a `FoldData` dataclass: gallery ids/labels/features, query ids/features,
  and NO query labels. Query labels live only in a separate `FoldTruth` object used by `metrics.py`.
- `decision.py` is the ONLY writer of submission files and always runs the validator.

CLI:
```
python scripts/check_compliance.py
python scripts/00_audit.py
python scripts/01_build_sift_cache.py --k 60 --workers 8
python scripts/02_run_cv.py --config configs/exp_XXX.json        # writes experiments/<exp_id>/
python scripts/03_make_submission.py --config configs/exp_XXX.json
python -m unittest discover tests
```

---------------------------------------------------------------------------------------------------
## 3. Global conventions

- Everything is computed PER SPECIES. A hyena query never sees a leopard gallery item. Assert it.
- Fixed seeds (`base.json: seed=2026`); CV uses seeds 0..4 for the 5 splits.
- Cache anything that takes > 1 minute, keyed by a hash of its config section; never silently reuse a
  cache built with different parameters.
- NaN is a legal feature value (0-keypoint images). GBDTs take NaN; for LR/MLP impute + add a flag.
- Log wall time of every stage in `metrics.json`.

---------------------------------------------------------------------------------------------------
## 4. Metric (implement first, test first)

```python
def map_at_5(truth: dict[str, str], preds: dict[str, list[str]]) -> float:
    total = 0.0
    for qid, y in truth.items():
        labels = preds[qid][:5]
        assert len(labels) == len(set(labels)) and 1 <= len(labels) <= 5
        for k, lab in enumerate(labels, start=1):
            if lab == y:
                total += 1.0 / k
                break
    return total / len(truth)
```
`tests/test_metric.py` must reproduce the official worked example: A→1, B→1/2, C→1/3, D→0,
mean = 0.4583 (assert abs(x - 0.458333) < 1e-4).

Also report per split: MAP@5 per species, MAP@5 on known-only queries, top-1 accuracy,
`new_individual` recall at rank 1, ROC-AUC of P(new), candidate recall@N (true identity in top-N).

---------------------------------------------------------------------------------------------------
## 5. Leakage rules (enforced by tests/test_leakage.py)

| Fitted thing | May use | Must never use |
|---|---|---|
| LDA / learned whitening / NCA / rerankers / pair models / calibrators / P(new) / temperature / β | fold gallery labels, or out-of-fold candidate tables from OTHER splits | query labels of the evaluated split |
| PCA whitening, centering, QE/DBA, k-reciprocal, test graph | gallery ∪ query FEATURES of the fold (train ∪ test at inference) | any labels |
| viewpoint imputer | all images with a recorded viewpoint | identity labels |
| SIFT pair cache | shared across folds (shortlist is label-free raw cosine) | — |
| hyperparameters | mean CV over 5 splits | public LB beyond the allowed probes |

Required tests:
1. gallery ∩ query = ∅ and held-out "new" identities have zero gallery images, for every split.
2. Permuting query labels leaves every feature table bit-for-bit identical (features don't read labels).
3. Every query appears in exactly one evaluation split per seed; species never mixed.
4. Second-level models evaluated on split r were trained only on tables from splits ≠ r.

---------------------------------------------------------------------------------------------------
## 6. Gates (do not start a gate before the previous exit check passes)

### Gate 1 — EDA & data-integrity audit
Build `src/reid/io.py`, `audit.py`, `scripts/00_audit.py`. Produce `reports/eda.md` + figures with:
1. Integrity asserts: shapes, dtypes, NaN/inf, duplicate ids, all ids aligned, every id has `_xy` and
   `_desc`, xy within [0, 512], desc uint8 ×128. Save `data/cache/aligned.npz`.
2. Embedding norms (expect ≈1); quick LR species-from-embedding accuracy (sanity only).
3. Per species: #images, #identities, shot histogram (1, 2, 3–5, 6–10, 11–20, >20).
4. Viewpoint distribution train vs test, NaN counts, identities with both left-side and right-side images
   (side(v): L = {left, frontleft, backleft}, R = {right, frontright, backright}, N = others).
5. Keypoint-count histograms per species × viewpoint; list 0-kp and <50-kp ids.
6. Same-ID vs diff-ID cosine distributions per species; viewpoint-pair compatibility matrix
   (mean same-ID cosine + #pairs for every viewpoint pair) saved to `data/cache/vp_compat.csv`.
7. Near-duplicates: train-train cosine > 0.97 (flag different-identity ones as possible label noise),
   test-train near-duplicates.
8. Novelty estimate: per species, test top-1 cosine to train vs train leave-one-out top-1 (images with ≥1
   sibling). Fit a 2-component GaussianMixture on test top-1 → π̂ (weight of the low-mean component).
9. Share of test images whose nearest pool neighbour is another test image.
10. SIFT pilot: on 3,000 shortlisted pairs per species, ROC-AUC of RANSAC inliers vs cosine vs LR(both).
**Exit:** all asserts pass; π̂ per species written to `configs/base.json`; `reports/gate1.md` written.

### Gate 2 — Open-set CV + baseline submission
1. `splits.py` (per species, seeds 0..4):
   a. Within each identity, agglomerative clustering (average linkage, cosine distance threshold
      `dup_thr`, start 0.10) → encounter clusters. Splits move whole clusters.
   b. Hold out whole identities (stratified by shot bin) until their images ≈ π̂ × |Q|. Label them
      `new_individual`; none of their images stay in gallery.
   c. From remaining identities with ≥2 clusters, move clusters to query at rate giving |Q|/|G| ≈ 0.39;
      keep ≥1 cluster in gallery. Single-image identities stay in gallery.
   d. Compute KS distance between CV-query top-1 cosine and test top-1 cosine; tune `dup_thr` and move
      rate to minimise it. Log KS in every experiment.
   Save splits to `data/cache/splits/` (json of ids). `tests/test_splits.py` covers a–c.
2. `metrics.py` + `tests/test_metric.py`.
3. Baseline: raw cosine (faiss IndexFlatIP), identity score = max image cosine, top-5 identities;
   insert `new_individual` after identities with score ≥ τ (τ tuned per species on CV); if fewer than 5
   labels, fill with next identities. Always output 5 labels.
4. `decision.py` validator: 2,791 rows, ids exactly = test ids, 1–5 labels, no duplicates, every label is
   `new_individual` or a training identity of the row's species.
5. LB probes allowed in this gate: (i) the unchanged `sample_submission.csv` (public score = public new
   fraction; compare with π̂), (ii) the baseline.
**Exit:** CV mean ± std over 5 splits reported (std < 0.01); KS logged; LB within ≈ ±0.03 of CV
(if not, fix CV before anything else); `reports/gate2.md`.

### Gate 3 — Global metric learning + re-ranking (Stage A)
Each item is a separate experiment against the Gate 2 baseline; keep only if mean ΔMAP@5 > max(0.002, 1 SE
of the paired per-split differences).
1. Per-species centering + PCA whitening (fit on train ∪ test features, unsupervised), d ∈ {128, 256,
   512, 1024}, eigenvalue floor ε ∈ {0, 0.1, 1} × mean eigenvalue; re-L2.
2. Regularised LDA / learned whitening fit on gallery labels: S_w from identities with ≥2 images,
   S_w(γ) = (1−γ)S_w + γ·tr(S_w)/d·I, γ ∈ [0.05, 0.5]; project to d′ between-class directions.
   Variant: remove per-viewpoint mean subspace (nuisance projection). Variant: sklearn NCA on PCA-256.
3. Identity aggregation: max, top-2 mean, prototype cosine, soft-max (τ ∈ {0.02, 0.05, 0.1}),
   best compatible-viewpoint cosine.
4. α-QE / DBA on the train ∪ test pool (k ∈ {2,3,5}, α ∈ {1,3}).
5. k-reciprocal re-ranking (start k1=20, k2=6, λ=0.3; also try k1 ∈ {5, 10}).
6. Viewpoint imputer: LogisticRegression on PCA-128 embeddings → soft viewpoint probs + missing flag.
7. Choose N (candidates per query) so candidate recall@N > 97% on known queries (start 20).
**Exit:** best Stage-A config chosen on CV; feature table builder `candidates.py` emits all A-features;
`reports/gate3.md` with an ablation table.

### Gate 4 — SIFT verification + learned reranker + P(new) (Stages B, C, D)
1. `sift_match.py`: RootSIFT (L1-normalise, sqrt, float32); BFMatcher NORM_L2 knnMatch k=2 both ways;
   Lowe ratio r=0.8 (tune 0.75–0.85) AND mutual; if ≥ 8 matches: `estimateAffinePartial2D` RANSAC
   (thr ≈ 0.02 × max side) + `findHomography` RANSAC. Pair features: kp counts, ratio matches, mutual
   matches, affine inliers, H inliers, inlier ratios (÷ mutual, ÷ min kp), mean inlier RootSIFT distance,
   median reprojection error, log scale, |rotation|, inlier convex-hull area ratio in both images, std of
   inlier xy. 0-kp → NaN + flag. Unit-test on synthetic data: a point set + a known similarity transform
   + noise must give high inliers; a random set must give low inliers.
2. `sift_cache.py` / `01_build_sift_cache.py`: for every image in train ∪ test, top-60 same-species
   neighbours by RAW cosine over the pool; match each unordered pair once with ProcessPoolExecutor
   (cv2.setNumThreads(1) per worker); store in `data/cache/sift_pairs.npz`; lazily add missing pairs.
3. Entry check (must pass before continuing): Gate 1 item 10 repeated on the real cache, per species.
4. Identity-level SIFT aggregates per candidate: max / top-2-sum inliers, #gallery images with ≥10 inliers,
   best inlier ratio, spread of best pair; isotonic-calibrated P(same) (WildFusion-style fused baseline:
   average of calibrated cosine and calibrated inliers — log it as its own experiment).
5. Reranker (`models.py`): rows = (query, candidate identity) for top-N, label = is true identity.
   Feature groups: global, SIFT, viewpoint (compat value, has-compatible-side image, count), gallery stats
   (#images, #viewpoints of c), query context (top-1 cosine, kp count, vp missing, species).
   Models: LightGBM binary (monotone +1 on cosines/inliers) as primary; then LightGBM lambdarank,
   CatBoost, XGBoost, L2 LogisticRegression (standardised), MLPClassifier(early_stopping=True).
   Training: leave-one-split-out over the 5 splits. No resampling.
6. Stage D (`calibration.py`): known posterior = softmax(z/T) over candidates, T by OOF NLL;
   P(new) model (LightGBM + LR check) on query-level features; P(c) = (1−π)·P(c|known);
   prior-shift correction with π_test (EM on test predictions, compare with π̂ and the LB probe);
   one β offset on logit π per species tuned on CV.
7. `decision.py`: sort {candidates ∪ new} by probability, output top 5. (Proof: E[score] = Σ P(ℓ_k)/k with
   decreasing weights → sorted order is optimal; always output 5.)
**Exit:** gain over Gate 3 per species reported; reliability diagram + ECE + NLL for top-1 and P(new);
feature-importance and per-group ablation in `reports/gate4.md`; one LB submission.

### Gate 5 — Transductive graph, ensembling, final calibration
1. Image-pair model (same features as pairs, trained on CV image pairs, leave-one-split-out).
2. Test graph per species: edges for mutual top-10 neighbours with P(same) > τ_e. Clusters by
   (a) agglomerative average linkage on 1−P(same) and (b) networkx louvain_communities; keep only
   groups both agree on.
3. Graph features into reranker and P(new) (cluster size, best gallery prob in cluster, vote share for c,
   is-nearest-neighbour-a-test-image).
4. Label propagation P* = (1−α)(I−αS)⁻¹P⁰ with S = D^-½ W D^-½, rows renormalised; α tuned on CV
   (α=0 = off). The SAME procedure runs on CV query sets.
5. Optional pseudo-gallery second pass (P(c) > 0.95 added to gallery) — keep only if clearly better.
6. Stacking of reranker models (LR on OOF logits vs rank-average); keep if > 1 SE better than best single.
7. Final: re-fit second-level models on all 5 splits' tables, compute test features against the full
   training gallery, run decision + validator.
**Exit:** final ablation table (each component on/off), two submissions selected:
(1) best CV, (2) best CV without label propagation/pseudo-gallery. `reports/gate5.md`.

---------------------------------------------------------------------------------------------------
## 7. Experiment tracking (mandatory)

- Every CV run gets `exp_id = <yyyymmdd-hhmm>_<short-slug>` and a folder with `config.json` (full merged
  config), `metrics.json` (per split + mean/std, per species, timings, KS), `oof.npz`, `notes.md`
  (one paragraph: hypothesis, change, result, decision).
- Append one row to `experiments/registry.csv`:
  `exp_id, date, parent_exp, change, cv_mean, cv_std, map_hyena, map_leopard, known_map, new_auc,
   ks, accepted, lb_public(optional)`.
- Acceptance rule: accept iff mean ΔMAP@5 vs parent > max(0.002, SE of paired per-split deltas).
  Otherwise mark rejected and revert. Never accept on LB alone.
- LB budget: Gate 2 two probes, then ≤ 1 per gate; record every LB score in the registry.

---------------------------------------------------------------------------------------------------
## 8. Compliance checker (scripts/check_compliance.py) — implement exactly this logic

```python
import ast, sys, pathlib
ALLOWED_THIRD_PARTY = {"numpy","pandas","scipy","sklearn","statsmodels","imblearn","xgboost",
                       "lightgbm","catboost","networkx","cv2","faiss","matplotlib","seaborn"}
LOCAL = {"reid", "src", "scripts", "tests"}
CV2_ALLOWED = {"BFMatcher","BFMatcher_create","FlannBasedMatcher","DMatch","NORM_L2",
               "findHomography","estimateAffinePartial2D","estimateAffine2D","findFundamentalMat",
               "perspectiveTransform","transform","RANSAC","USAC_MAGSAC","USAC_DEFAULT","LMEDS",
               "setNumThreads","__version__"}
STDLIB = set(sys.stdlib_module_names)
errors = []
for path in pathlib.Path(".").rglob("*.py"):
    if any(p in path.parts for p in (".venv", "venv", "site-packages")):
        continue
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods = [node.module.split(".")[0]]
        for m in mods:
            if m not in ALLOWED_THIRD_PARTY | STDLIB | LOCAL:
                errors.append(f"{path}:{node.lineno} forbidden import '{m}'")
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == "cv2" and node.attr not in CV2_ALLOWED):
            errors.append(f"{path}:{node.lineno} cv2.{node.attr} not allowed")
        if isinstance(node, ast.ImportFrom) and node.module == "cv2":
            errors.append(f"{path}:{node.lineno} use 'import cv2' + allowlisted attributes only")
if errors:
    print("\n".join(errors)); sys.exit(1)
print("compliance OK")
```
Requires Python ≥ 3.10 (`sys.stdlib_module_names`). `joblib`, `torch`, `yaml`, `pytest` etc. are not
stdlib and not allowlisted, so they fail automatically. Also add to
`tests/test_compliance.py`: no file under `submissions/` is newer than its experiment's `metrics.json`
without a matching registry row (i.e. every submission traces to a pipeline run).

---------------------------------------------------------------------------------------------------
## 9. Default hyperparameters (starting points, all tuned on CV)

| Parameter | Start | Range |
|---|---|---|
| dup_thr (near-duplicate clustering) | 0.10 | 0.05–0.20 |
| N candidate identities | 20 | 10–40 |
| SIFT shortlist K | 60 | 30–100 |
| Lowe ratio | 0.80 | 0.75–0.85 |
| RANSAC threshold | 0.02 × max side | 0.01–0.04 |
| PCA-whiten d | 256 | 128–1024 |
| LDA shrinkage γ | 0.2 | 0.05–0.5 |
| k-reciprocal (k1, k2, λ) | (20, 6, 0.3) | k1 5–20 |
| LightGBM | lr 0.03, leaves 31, min_child 20, feature_frac 0.8, 2000 rounds + early stop | — |
| test-graph edge τ_e | 0.9 | 0.7–0.97 |
| propagation α | 0.5 | 0–0.9 |

---------------------------------------------------------------------------------------------------
## 10. Reporting (teaching mode)

After each gate write `reports/gateN.md` with:
1. What was built (modules, functions) — one line each.
2. Why (link each choice to the failure mode it addresses: species leakage, flank asymmetry, long tail,
   open set, repeated new individuals, burst near-duplicates, empty SIFT, gallery-size shift).
3. Results table (CV mean ± std, per species, known-only, new AUC, KS, LB if any).
4. What surprised you / what failed and why.
5. Three viva-style questions the student should be able to answer about this gate, with answers.
Keep math in LaTeX (`$...$`). Keep it short and concrete.

---------------------------------------------------------------------------------------------------
## 11. When to stop and ask the human

- Any rule ambiguity (write to `reports/questions_for_TA.md`).
- CV and LB disagree by > 0.03 after Gate 2.
- π̂ estimates (GMM, LB probe, EM) disagree by > 5 points.
- A stage would take > 3 hours of compute.
- You want to change the architecture (explain the evidence first).

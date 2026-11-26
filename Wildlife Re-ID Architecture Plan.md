# Wildlife Re-ID Architecture Plan

## Nov 19, 2026 ·

The plan is a calibrated **retrieve → verify → rerank → decide** system: MiewID cosine retrieval shortlists candidates, SIFT + RANSAC verifies them, a gradient-boosted reranker scores each (query, identity) pair, and a separate model estimates P(new\_individual). Labels are then output in descending probability, which is provably the MAP@5-optimal ordering. A test-to-test graph exploits the fact that new individuals repeat in test.csv. Every change is accepted only if it improves a leak-free open-set cross-validation built to mimic the test split.

## 1. What the papers teach us

Saket Anand sir's group has used the same recipe in every paper: a strong global descriptor ranks the gallery, then something cheaper and more local (flank side, SIFT geometry, pairwise constraints) fixes the top of the ranking. We can't train CNNs, but MiewID already gives us the global part, so our job is everything *after* the embedding.

| Paper | Key finding | What we borrow |
| --- | --- | --- |
| [Shukla, Anderson, Cheema, …, Anand, Farrell — Hybrid Tiger Re-ID, ICCVW 2019](https://doi.org/10.1109/ICCVW.2019.00039) | Rank by global cosine, push same-flank images first, then re-rank only the top 20 with SIFT (ratio test + homography inliers). Re-ranking more than 20 *hurt*, because background clutter creates false SIFT matches. Each step improved mAP in the ablation. | Shortlist-then-verify; viewpoint/flank priority; the shortlist size K is a tuned hyperparameter, not "as big as possible". |
| [Shukla, Cheema, Anand et al. — Macaque MFID, 2018](https://arxiv.org/abs/1811.00743) and [PFID, 2019](https://arxiv.org/abs/1907.02642) | Cross-entropy alone overfits with few images per identity; adding a pairwise same/different loss improves open-set and verification most. Evaluation uses identity-disjoint 80/20 splits with closed-set, open-set and verification protocols. | Pairwise supervision (a "same individual?" pair classifier) instead of a per-identity softmax; identity-disjoint open-set CV. |
| [Cheema & Anand — Detection and Recognition of Patterned Species, ECML-PKDD 2017](https://arxiv.org/abs/2005.02905) | Frozen CNN features + PCA (99% energy) + L2-regularised logistic regression beat HotSpotter on stripes, but **SIFT-based HotSpotter won on jaguar spots** at rank-1. | Spotted coats (leopards, hyenas) are exactly where local SIFT evidence adds the most; PCA + regularised linear models are the baseline toolkit. |
| Shukla & Anand — Metric-learning based segmentation of patterned species, ICIP 2016 (your project file) | A low-rank Mahalanobis metric (sparse eigenvalues) generalised better than full-rank LMNN/ITML on tiny training sets. | Prefer low-rank, shrinkage-regularised linear metrics (whitened PCA, regularised LDA) over full-rank ones. |
| [Sani, Khurana, Anand — Ambiguity-Aware Sampling, AAAI 2026](https://arxiv.org/abs/2511.06658) | Open-set protocol holds out 20% of identities; disagreement between two different clusterings (density-based vs nearest-neighbour-based) marks ambiguous regions; similarity built on reciprocal k-NN. Reports MiewID zero-shot results too. | Identity hold-out CV; test-graph clustering with two methods and trust only where they agree; reciprocal-neighbour features. |
| [WildFusion — Cermak et al., ECCV-W 2024](https://arxiv.org/abs/2408.12934) | Global cosine and local-match counts live on different scales, so each is first calibrated (isotonic or logistic) into P(same), then averaged. | Calibrate-then-fuse is our first fusion baseline before the learned reranker. |
| [MiewID — Otarashvili et al., 2024](https://arxiv.org/abs/2412.05602) and [model card](https://huggingface.co/james-burgess/miewid) | EfficientNetV2 backbone trained with sub-center ArcFace; output is a unit-norm 2,152-d vector meant to be compared with cosine. | Our features are angular by construction: keep cosine geometry, verify unit norms in EDA. |
| AnimalCLEF 2025 winners: [1st place](https://ceur-ws.org/Vol-4038/paper_251.pdf), [2nd place](https://ceur-ws.org/Vol-4038/paper_253.pdf) | Same open-set task with `new_individual`. In the 1st-place ablation, swapping in MiewID added +0.043 and **adding local matching added +0.212**, the biggest single gain. 2nd place used per-species thresholds, k-reciprocal re-ranking and an XGBoost gate on neighbour-distance features. | Confirms the architecture and that per-species calibration matters. |
| [HotSpotter — Crall et al., WACV 2013](https://arxiv.org/abs/2508.17605) | One-vs-many SIFT scoring (LNBNN) beats one-vs-one matching; already applied to leopards. | Stretch goal: identity-level SIFT score. |
| [Zhong et al. — k-reciprocal re-ranking, CVPR 2017](https://arxiv.org/abs/1701.08398); [Arandjelović & Zisserman — RootSIFT, CVPR 2012](https://projet.liris.cnrs.fr/imagine/pub/proceedings/CVPR2012/data/papers/368_P3B-01.pdf) | Jaccard distance over reciprocal neighbour sets fixes rankings without labels; RootSIFT (Hellinger kernel) beats raw SIFT for matching. | Stage A and Stage B building blocks. |

What we **cannot** copy: the CNN fine-tuning, KL-divergence losses, JPEG augmentation and detectors in these papers. Rules ban deep learning and computing new image features, so each idea is re-expressed with allowed tools (pair classifiers, linear projections, GBDTs, graph methods).

## 2. Problem formulation and failure modes

For a test image x of species s(x), we must output an ordered list of up to 5 distinct labels from the candidate set below, where 𝓘ₛ is the set of training identities of that species:

```latex
\mathcal{L}_{s(x)} = \mathcal{I}_{s(x)} \cup \{\texttt{new\_individual}\}, \qquad \mathrm{MAP@5} = \frac{1}{N}\sum_{i=1}^{N} \sum_{k=1}^{\min(5,n_i)} \frac{\mathbb{1}[\hat{y}_{i,k} = y_i]}{k}
```

The task is therefore **open-set retrieval**, not classification: we rank identities, and "new" competes as one more label. Each failure mode below drives one design decision.

1. **Species leakage.** A hyena query must never receive a leopard label. Every index, projection, model and threshold is built per species, so the project is really two independent problems that share code.
2. **Viewpoint and flank asymmetry.** A spotted animal's left and right flanks carry *different* spot patterns, but one identity label covers both sides. If a query shows the left flank and the gallery only holds right-flank images of that animal, SIFT cannot match (no shared spots) and global similarity drops, so the query looks "new". This is the most likely source of false `new_individual` predictions. Fix: learn a viewpoint-pair compatibility matrix from training same-identity pairs (don't hand-code that `frontleft` ≈ `left`), add per-identity "side coverage" features, and let the P(new) model learn that low similarity with *no compatible-side gallery image* is weaker evidence of novelty. Missing viewpoints are imputed with a logistic-regression classifier on PCA-reduced MiewID vectors (pose is encoded in the embedding), kept as soft probabilities plus a missing flag.
3. **Long tail (1–2 shots).** A softmax classifier needs one weight vector per identity estimated from 1–2 samples. With separable data the maximum-likelihood weights diverge, the classifier memorises, and the closed-world normalisation (probabilities sum to 1 over known identities) leaves no room for "new". We use non-parametric retrieval instead: adding an identity costs nothing, and a single class-agnostic "same individual?" pair model learns from all pairs. Positive pairs come from multi-shot identities, negatives from everyone, so 1-shot identities still contribute. A side effect to correct: max-similarity aggregation favours identities with many gallery images (more chances to score high), so gallery-size features go into the reranker.
4. **Open set with unknown novelty rate.** A fixed cosine threshold is fragile because the "right" threshold moves with species, viewpoint, keypoint count and gallery size. Instead we learn P(new | x) from query-level evidence and correct it for the difference between the CV novelty rate and the test novelty rate (Section 8).
5. **Repeated new individuals in test.** The same unseen animal can appear several times in test.csv. A query that is very close to *another test image* but far from every training image is strong evidence of "new". Known animals also repeat, and their test images should agree on a label. We exploit both with a test-to-test graph (Section 9).
6. **Near-duplicate bursts.** Camera traps fire bursts of near-identical frames. If a random within-identity split puts one burst frame in the gallery and its twin in the query set, CV looks far better than the test score will be. CV moves whole near-duplicate clusters together (Section 4).
7. **Degenerate SIFT.** About 5% of images have fewer than 50 keypoints and a few have none. Pair features must be NaN-aware (GBDTs route missing values natively) and the system must fall back to global evidence, never crash or score a zero-keypoint pair as "different".
8. **Gallery-size shift.** At test time the gallery is all 7,163 training images; in CV it is smaller because some identities and images are held out. Raw counts ("number of neighbours above 0.6") shift with gallery size, so features are expressed as ranks, gaps and ratios where possible.

## 3. At a glance

&#91;embedded content: pipeline · 3 inputs, 4 stages, 1 test graph\]

Read top to bottom: MiewID ranks, SIFT verifies a cached shortlist, the reranker weighs all evidence per (query, identity), and Stage D turns scores into probabilities that alone decide the order of the five labels. The open-set CV on the left is the only judge of whether any box earns its place.

## 4. Leak-free open-set cross-validation

CV is the only judge, so it must reproduce four properties of the real split: the fraction of `new_individual` queries, the query-to-gallery size ratio (2,791 / 7,163 ≈ 0.39), the per-identity shot distribution, and the distribution of each query's top-1 similarity to the gallery.

**Step 1 — Estimate the test novelty rate π̂ per species** (two independent estimates, then compare):

1. *Mixture model.* For every test image compute its top-1 cosine to the training gallery. Novel images form a low-similarity mode and known images a high one. Fit a 2-component Gaussian mixture (EM) on these values; the weight of the low component estimates π̂. Sanity-check by overlaying the leave-one-out top-1 similarity of training images that have at least one same-identity sibling (a "known" reference distribution).
2. *One leaderboard probe.* Submit `sample_submission.csv` unchanged (everyone = `new_individual`). Its public score is exactly the fraction of new individuals in the public 30% of test (≈ 837 images, so standard error ≈ 1.7 points). It uses only provided files, but spend at most this one probe on it.

**Step 2 — Build R = 5 repeated splits per species** (different seeds):

1. *Near-duplicate grouping.* Inside each identity, run agglomerative clustering on cosine distance (average linkage; threshold tuned in EDA, start at 0.10). A cluster ≈ one encounter/burst. Splits move whole clusters, never single frames.
2. *Held-out "new" identities.* Sample whole identities, stratified by shot-count bin (1, 2, 3–5, 6–10, >10), until their images make up π̂ of the query set. All their images become queries labelled `new_individual`; none stay in the gallery.
3. *Known queries.* From the remaining identities with ≥ 2 clusters, move clusters to the query set with a rate chosen so |Q| / |G| ≈ 0.39, always keeping ≥ 1 cluster in the gallery. Single-image identities stay in the gallery as distractors.
4. *Match the test.* Compute the Kolmogorov–Smirnov distance between the CV queries' top-1 similarity distribution and the test's. Tune the cluster threshold and move rate to minimise it, and log the KS value with every experiment. This is our adversarial-validation check that CV is as hard as test.

**Step 3 — Score.** Per split: MAP@5 overall and per species, MAP@5 on known queries only, top-1 accuracy, recall of `new_individual` at rank 1, and ROC-AUC of P(new). Report mean ± standard deviation over the 5 splits. A change is accepted only if mean MAP@5 rises by more than max(0.002, one standard error of the paired difference).

**Leakage rules** (enforced by unit tests in `tests/test_leakage.py`):

| What is fitted | Allowed data inside a fold | Why |
| --- | --- | --- |
| Anything using identity labels: LDA / learned whitening, rerankers, pair classifiers, calibrators, P(new) model, temperatures, β offsets | Gallery labels of that fold only, or out-of-fold candidate tables from *other* splits | Query labels are the exam answers. |
| Unsupervised transforms: centering, PCA whitening, k-reciprocal re-ranking, query expansion, test-graph clustering | Gallery ∪ query features of that fold (never labels) | Mirrors test time, where we may use train ∪ test features without labels. |
| Viewpoint imputer | All images with a recorded viewpoint | Viewpoint is a given input in test too, not the target. |
| SIFT pair cache | Shared across folds | Pairs are chosen by raw cosine without labels, so the cache contains no label information. |
| Hyperparameters | Chosen on mean CV across splits | The public leaderboard is a 30% sample and is used only as a sanity check. |

The second-level models (Stage C reranker, P(new) model) are trained with **leave-one-split-out**: features for split r are computed from split r's gallery, and the model evaluated on split r is trained on candidate tables from the other four splits. For the final test submission, they are retrained on all five splits' tables, and test features are computed against the full training gallery.

## 5. Stage A — Global MiewID retrieval

Stage A turns each query into a short list of about 20 candidate identities with a vector of global-similarity features. MiewID was trained with an angular (ArcFace) loss, so every transform below must keep cosine geometry meaningful, and each one is an experiment that stays only if CV improves.

**A1. Baseline (Gate 2).** L2-normalise, take cosine to every same-species gallery image (exact search with `faiss.IndexFlatIP`), score each identity by its best image, and insert `new_individual` after the identities whose score is ≥ τ, with τ tuned on CV. This sets the floor every later stage must beat.

**A2. Centering + PCA whitening (unsupervised).** Fit per species on train ∪ test features. Whitening divides each principal direction by its standard deviation, so that a few high-variance "nuisance" directions (lighting, pose, background) stop dominating the dot product:

```latex
\tilde{x} = \Lambda_d^{-1/2} U_d^{\top}(x - \mu), \qquad \hat{x} = \frac{\tilde{x}}{\lVert \tilde{x} \rVert_2}
```

Here U\_d and Λ\_d are the top-d eigenvectors and eigenvalues of the covariance. Search d ∈ {128, 256, 512, 1024} and an eigenvalue floor (add ε · mean eigenvalue before the inverse square root) to avoid amplifying noise directions.

**A3. Supervised linear metric learning (fit on gallery labels only).** This is "learned whitening", equivalent to a regularised LDA projection. Estimate the within-identity scatter S\_w from identities with ≥ 2 images, shrink it toward a scaled identity (the long tail makes S\_w noisy), whiten by it, then keep the top-d′ between-identity directions:

```latex
S_w(\gamma) = (1-\gamma)\,S_w + \gamma\,\frac{\operatorname{tr}(S_w)}{d}\,I, \qquad P = V_{d'}^{\top}\, S_w(\gamma)^{-1/2}
```

V\_d′ are the leading eigenvectors of the between-identity scatter after whitening. Tune γ ∈ \[0.05, 0.5\] and d′. This is the "low-rank, regularised metric" lesson from Shukla & Anand 2016. Variants to try: (a) a viewpoint *nuisance projection* that removes the subspace spanned by per-viewpoint mean vectors; (b) scikit-learn `NeighborhoodComponentsAnalysis` on a 256-d PCA input (allowed: metric learning with a linear projection). Risk: MiewID may already have seen these species during pre-training, so a supervised projection can overfit. Keep both raw and projected cosines as reranker features rather than replacing one with the other.

**A4. Identity aggregation.** For identity c with gallery images G\_c, compute several scores, because "best image" and "average image" fail differently:

```latex
S^{\max}_c = \max_{g \in G_c} s(x,g), \quad S^{\mathrm{proto}}_c = \cos\!\Big(x, \tfrac{1}{|G_c|}\textstyle\sum_{g} g\Big), \quad S^{\mathrm{soft}}_c = \tau \log \textstyle\sum_{g \in G_c} e^{s(x,g)/\tau}
```

Plus the mean of the top-2 image similarities, and the best similarity among images of a *compatible* viewpoint only.

**A5. Query expansion / database-side augmentation (unsupervised).** Replace each vector by a similarity-weighted average of itself and its top-k neighbours in the train ∪ test pool (α-QE: weights sᵃ, k ∈ {2, 3, 5}, α ∈ {1, 3}), then renormalise. Because test neighbours are included, repeated test sightings of the same animal pull toward each other.

**A6. k-reciprocal re-ranking (Zhong et al. 2017).** Two images are trusted neighbours only if each is in the other's top-k1 list. The Jaccard distance between those reciprocal sets is blended with the original distance:

```latex
d^{*}(x, g) = (1-\lambda)\, d_J(x, g) + \lambda\, d(x, g)
```

Start from k1 = 20, k2 = 6, λ = 0.3, but expect smaller k1 to win: k-reciprocal assumes several same-identity neighbours, and our 1–2-shot identities cannot have them. So d\* is added as a feature, not used as the only ranking.

**Output of Stage A:** for each query, the top N = 20 identities (tune 10–40; check that the true identity is inside the list for > 97% of known queries, the "candidate recall"), with all A-features attached.

## 6. Stage B — Local SIFT + RANSAC verification

SIFT answers a different question from MiewID: "do the *same spots* appear in the *same geometric arrangement*?" A match surviving RANSAC is hard to get by chance, which is why it is the strongest evidence for spotted coats and the biggest single gain reported by AnimalCLEF winners.

**Why not all pairs.** Test × train is 2,791 × 7,163 ≈ 20 million pairs, and CV also needs train × train. At roughly 2–5 ms per 500 × 500 match plus RANSAC, that is 11–28 CPU-hours for test alone. Worse, almost all of those pairs are different animals, and the tiger paper showed extra candidates mostly add background false matches. So we verify a shortlist.

**B1. Label-free shortlist and cache.** For every image in train ∪ test, take its top K = 60 same-species neighbours by *raw* MiewID cosine over the whole pool, and match each unordered pair once. Because the shortlist uses no labels, one cache (`data/cache/sift_pairs.npz`, keyed by sorted id pair) serves every CV fold and the final test run. This is about 9,954 × 60 / 2 ≈ 300k pairs: around 1 hour on 8 processes. Missing pairs are computed lazily and appended.

**B2. Descriptor preparation (RootSIFT).** SIFT descriptors are gradient histograms, and Euclidean distance on histograms is dominated by large bins. RootSIFT L1-normalises and takes the square root, so Euclidean distance becomes the Hellinger distance:

```latex
\bar{d} = \frac{d}{\lVert d \rVert_1 + \varepsilon}, \qquad d' = \sqrt{\bar{d}}, \qquad \lVert d'_a - d'_b \rVert_2^2 = 2 - 2\sum_i \sqrt{\bar{d}_{a,i}\, \bar{d}_{b,i}}
```

**B3. Matching.** `cv2.BFMatcher(cv2.NORM_L2)` with `knnMatch(k=2)` in both directions; keep a match only if it passes Lowe's ratio test (best distance < r × second-best, r ∈ \[0.75, 0.85\], start 0.8) **and** is mutual. Images with 0 keypoints get all-NaN features plus a flag; with fewer than 8 matches, skip RANSAC and keep only counts.

**B4. Geometric verification.** Animal bodies bend, so a rigid model is only approximate on a flank patch:

1. `cv2.estimateAffinePartial2D` (4 degrees of freedom: rotation, uniform scale, translation) with RANSAC, reprojection threshold ≈ 2% of the longer side (≈ 10 px at 512). Primary model: it needs only 2 points and cannot express a mirror flip, which rejects left-vs-right flank "matches".
2. `cv2.findHomography` (RANSAC or USAC\_MAGSAC) as a looser secondary model.

**B5. Pair features** (all saved to the cache):

| Group | Features |
| --- | --- |
| Counts | keypoints in each image, ratio-test matches, mutual matches |
| Inliers | affine inliers, homography inliers, inliers / mutual matches, inliers / min(keypoints) |
| Quality | mean RootSIFT distance of inliers, median reprojection error, symmetric transfer error |
| Transform plausibility | log scale, absolute rotation angle (same-view pairs should be near 0) |
| Spatial spread | convex-hull area of inliers / area of all keypoints, in both images; std of inlier x and y (20 inliers on one spot cluster are weaker than 20 spread across the flank) |

**B6. Identity-level aggregation** for candidate c: maximum affine inliers over G\_c, sum of the top-2, number of gallery images with ≥ 10 inliers, best normalised inlier ratio. Then a WildFusion-style isotonic calibration maps max-inliers to P(same), which is used in the first fused baseline.

**B7. Entry check before building on SIFT.** On \~3,000 shortlisted pairs per species, compare ROC-AUC of inlier count vs cosine for same/different identity, and of a logistic regression on both. If the combination beats cosine alone, SIFT is complementary. Expect the gain to be larger for leopards' rosettes than for hyenas; measure, don't assume.

**Stretch: HotSpotter-style LNBNN.** Index all gallery RootSIFT descriptors per species in `faiss.IndexHNSWFlat`; for each query descriptor, add to each identity the gap between the (k+1)-th neighbour distance and that identity's nearest descriptor distance. This scores identities one-vs-many. It must be rebuilt per CV fold, so attempt it only if Gate 4 leaves time.

Parallelism uses the standard library (`concurrent.futures.ProcessPoolExecutor`) with `cv2.setNumThreads(1)` per worker. OpenCV is used only for matching and geometric verification, as the rules require: no `SIFT_create`, no image I/O.

## 7. Stage C — Pairwise candidate reranker and ensembling

The reranker learns *how much to trust each kind of evidence in each situation* (e.g. "high cosine but zero inliers on a same-side pair is suspicious; low cosine on an opposite-side pair is normal"). A hand-tuned weighted sum cannot express those interactions; a GBDT can.

**Unit of learning.** One row per (query q, candidate identity c) for the top-N candidates from Stage A. Label y = 1 if c is q's true identity, else 0. New-individual queries and retrieval misses contribute only negatives, which still teach the model what a convincing wrong candidate looks like. About 1 positive per 20 rows is a mild imbalance: no resampling (SMOTE would destroy calibration).

**Features** (each group is ablated on CV):

| Group | Features |
| --- | --- |
| Global | raw / whitened / LDA cosine (max, top-2 mean, prototype, soft-max), k-reciprocal distance, rank of c, gap to rank-1, gap to rank-(r+1), z-score of S\_c within the query's top-N |
| Local SIFT | identity-level max and top-2-sum inliers, hit count, best inlier ratio, spatial spread of the best pair, calibrated P(same) |
| Viewpoint | query viewpoint (soft), compatibility-matrix value for the best pair, whether c has any compatible-side gallery image, how many |
| Gallery stats | number of gallery images of c, number of distinct viewpoints of c |
| Query context | query top-1 cosine, query keypoint count, viewpoint-missing flag, species |
| Transductive | rank of q in the neighbour lists of c's images (reciprocity), votes: share of q's mutual test neighbours whose top-1 is c, best cosine between c and any member of q's test cluster |

**Models** (all trained from scratch on these tables, all allowed):

1. LightGBM binary classifier: the primary model. Use monotone constraints (+1 on inliers and cosines) so the model can't learn noisy reversals.
2. LightGBM `lambdarank` with one group per query: optimises ranking directly; its scores are not probabilities, so it is used only inside the stack.
3. CatBoost and XGBoost binary classifiers: different tree-building biases, useful for diversity.
4. L2 logistic regression on standardised features: a stable, well-calibrated anchor.
5. scikit-learn `MLPClassifier` (one or two hidden layers, early stopping): fully connected and trained from scratch, so it is allowed.

**Stacking.** Collect out-of-fold logits from leave-one-split-out training, then fit a logistic-regression stacker on them (or a simple average of rank-normalised scores if the stacker doesn't beat it). Keep the stack only if it beats the best single model by more than one standard error.

**A second pair model** (image-to-image, not image-to-identity) with the same global + SIFT + viewpoint pair features is trained on CV image pairs. It supplies the edge weights of the test graph in Section 9.

## 8. Stage D — Calibrated probabilities and the MAP@5-optimal decision

The key result: once we have calibrated probabilities over {candidate identities} ∪ {new}, the best possible submission is simply the five labels with the highest probability, in descending order. No hand-set "new threshold" is needed; the threshold is implied by the probabilities.

**D1. Known-identity posterior.** Turn the reranker scores z\_c of the candidate set C\_x into a distribution with a temperature T fitted by minimising negative log-likelihood on out-of-fold data (a conditional-logit GLM):

```latex
P(y = c \mid x, \text{known}) = \frac{\exp(z_c / T)}{\sum_{c' \in C_x} \exp(z_{c'} / T)}
```

**D2. Novelty probability.** A separate query-level classifier (LightGBM, with an L2 logistic regression as a check) predicts π(x) = P(new | x) from: top-1 known posterior, its gap to the 2nd, top-1 raw cosine, best SIFT inliers, keypoint count, whether any compatible-side gallery image exists, and the test-graph features (size of x's test cluster, best gallery score of any cluster member, whether x's nearest neighbour is a test image). Labels come from the CV hold-out identities. The two heads combine as:

```latex
P(\texttt{new} \mid x) = \pi(x), \qquad P(c \mid x) = \big(1 - \pi(x)\big)\, P(y = c \mid x, \text{known})
```

Variant to compare: put "new" into the conditional logit as one more pseudo-candidate with its own feature vector, so a single model learns both.

**D3. Prior-shift correction (Saerens et al., 2002).** π(x) is learned at the CV novelty rate π\_cv. If the test rate π\_test differs, Bayes' rule rescales the odds:

```latex
\pi'(x) = \frac{\rho_1\, \pi(x)}{\rho_1\, \pi(x) + \rho_0\, \big(1 - \pi(x)\big)}, \qquad \rho_1 = \frac{\pi_{\text{test}}}{\pi_{\text{cv}}}, \quad \rho_0 = \frac{1 - \pi_{\text{test}}}{1 - \pi_{\text{cv}}}
```

π\_test can itself be estimated by EM on the test predictions: start from π̂ (Section 4), compute π′ for all test images, set π\_test to their mean, and repeat until it stops changing. Compare the EM value with the mixture-model and leaderboard-probe estimates; if they disagree by more than \~5 points, investigate before trusting any.

**D4. Why sorting by probability maximises MAP@5.** For a ranked list ℓ₁, …, ℓ₅ of distinct labels, the expected score of one query is:

```latex
\mathbb{E}[\text{score}] = \sum_{k=1}^{5} \frac{P(\ell_k \mid x)}{k}
```

Proof by exchange: suppose positions i < j hold labels with P(ℓ\_i) < P(ℓ\_j). Swapping them changes the expectation by the amount below, which is strictly positive because 1/i > 1/j:

```latex
\Delta = \big(P(\ell_j) - P(\ell_i)\big)\Big(\frac{1}{i} - \frac{1}{j}\Big) > 0
```

So any list that is not sorted can be improved; the sorted list is optimal (this is the rearrangement inequality). Adding a label never lowers the expectation because probabilities are non-negative, so **always output 5 labels**. `new_individual` simply takes the position its probability earns.

**Worked example.** P(A) = 0.45, P(new) = 0.30, P(B) = 0.15, P(C) = 0.06, P(D) = 0.04.

| Ordering | Expected score |
| --- | --- |
| A, new, B, C, D (sorted, optimal) | 0.45 + 0.15 + 0.05 + 0.015 + 0.008 = **0.673** |
| A, B, C, D, new ("new only as a fallback") | 0.45 + 0.075 + 0.02 + 0.01 + 0.06 = 0.615 |
| new, A, B, C, D ("top score below threshold") | 0.30 + 0.225 + 0.05 + 0.015 + 0.008 = 0.598 |

**D5. Calibration safety net.** Because only the *ordering* between P(new) and the P(c) values matters, one scalar offset β added to logit π′ (tuned per species on CV MAP@5) absorbs residual miscalibration. If the tuned β is far from 0, the probability model is miscalibrated and should be fixed upstream. Report reliability diagrams, expected calibration error and NLL for top-1 and for P(new), per species.

## 9. Transductive module — exploiting repeats inside test.csv

Test images of the same animal should receive the same answer, whether that animal is known or new. We link test images that are confidently the same individual and let their evidence flow along those links. It uses test *features* only, never labels, which the rules allow ("transform, combine, reduce and match" the provided features).

**T1. Edges.** Per species, connect test images i and j when they are mutual top-10 neighbours **and** the image-pair model (Section 7) gives P(same) > τ\_e. Edge weight w\_ij = P(same).

**T2. Clusters, with a consensus check.** Following the AAS paper's idea that two different clusterings disagree exactly where the structure is ambiguous, build clusters two ways: (a) average-linkage agglomerative clustering on 1 − P(same), and (b) `networkx` Louvain communities on the weighted graph. Keep only the groups both methods agree on; disagreement regions get no propagation. This guards against chaining two different animals together.

**T3. Label propagation of posteriors.** Stack each test image's distribution over the shared label set (candidate identities plus `new`) as rows of P⁰. With the normalised affinity S = D^(−1/2) W D^(−1/2) (Zhou et al. 2004):

```latex
P^{*} = (1 - \alpha)\,(I - \alpha S)^{-1} P^{0}, \qquad 0 \le \alpha < 1
```

α = 0 switches the module off; tune α on CV. Renormalise each row afterwards. Two images in one cluster that are both mostly "new" are consistent (the same new animal), so "new" can be a single shared label.

**T4. Graph features.** Cluster size, best gallery probability within the cluster, share of the cluster whose top-1 is c, and "is my nearest neighbour a test image?" feed the reranker and the P(new) model (Sections 7 and 8).

**T5. Optional second pass (pseudo-gallery).** Test images with P(c) > 0.95 are added to identity c's gallery and Stage A–D are re-run once. This helps another test image of the same animal seen from a viewpoint the original gallery lacked. It must be simulated identically inside CV, and it is the first thing to drop if it doesn't clearly help.

CV mirrors all of this naturally: CV queries also contain held-out identities with several images and known identities with several query images. Error propagation is the main risk, so every transductive piece is ablated on its own, and one of the two final submissions is chosen *without* it.

## 10. Execution gates

We work through five gates in order and do not start a gate until the previous one's exit check passes. The full instructions are in the companion `PLAN.md` file.

| Gate | Build | Exit check |
| --- | --- | --- |
| 1. EDA & integrity audit | `scripts/00_audit.py`, figures, `reports/eda.md`, π̂ per species | All integrity asserts pass; π̂ from the mixture model; viewpoint compatibility matrix and keypoint stats saved |
| 2. Open-set CV & baseline | `splits.py`, `metrics.py` (unit-tested on the official worked example = 0.458), raw-cosine baseline, first submission | 5 splits built; KS distance to test logged; CV MAP@5 std across splits < 0.01; leaderboard within roughly ±0.03 of CV |
| 3. Metric learning & re-ranking | Whitening, regularised LDA, QE, k-reciprocal, viewpoint features, identity aggregation | Each variant logged; only changes beating the baseline by > 1 SE kept |
| 4. SIFT verification & reranker | SIFT cache, pair features, LightGBM reranker, P(new) model, probability-sorted decision rule | SIFT entry check (B7) passed; gain over Gate 3 reported per species; calibration plots saved |
| 5. Ensemble, transductive, calibration | Stacking, test graph + label propagation, prior-shift EM, β offset | Final CV table with ablations; 2 final submissions picked |

**Gate 1 EDA checklist** (every item ends as a number or a figure in `reports/eda.md`):

1. Integrity: shapes, dtypes, NaN/inf, duplicate ids; every train/test id appears exactly once in `miewid_ids.csv` and has both `_xy` and `_desc` arrays; keypoint coordinates within \[0, 512\]; descriptors `uint8` with 128 columns.
2. Embeddings: norms (expected ≈ 1), and a quick check that a logistic regression predicts species almost perfectly (sanity, not a model).
3. Per species: images, identities, shot histogram (1, 2, 3–5, 6–10, 11–20, >20).
4. Viewpoints: train vs test distribution, missing counts, how many identities have both a left-side and a right-side image.
5. Keypoints: count histogram per species and viewpoint, list of 0-keypoint and <50-keypoint images.
6. Same-identity vs different-identity cosine distributions per species, and the viewpoint-pair compatibility matrix (mean same-identity cosine and pair count per viewpoint pair).
7. Near-duplicates: train-train pairs with cosine > 0.97 (different identities here suggest label noise), test-train near-duplicates.
8. Novelty: test top-1 cosine to train overlaid on train leave-one-out top-1; 2-component mixture → π̂.
9. Repeats: share of test images whose nearest neighbour in the pool is another test image rather than a train image.

**Experiment discipline.** Every run writes `experiments/<exp_id>/config.json`, `metrics.json` and its out-of-fold predictions, and appends one row to `experiments/registry.csv` (id, date, parent id, one-line change, CV mean, CV std, per-species MAP@5, KS distance, accepted yes/no). Seeds are fixed. A change is accepted only on CV, never on the leaderboard alone.

**Leaderboard budget.** One probe with the all-new sample submission; one submission per gate after that. At the end pick two: the best CV, and the best CV *without* transductive propagation, as a hedge against error propagation on the private 70%.

## 11. Repository blueprint

One package, thin CLI scripts, expensive things cached once. Configs are JSON and tests use the standard-library `unittest`, because PyYAML and pytest are not on the allowed library list.

```text
wildlife-reid/
├── PLAN.md                       # execution rules and gate checklist (companion file)
├── configs/
│   ├── base.json                 # paths, seeds, species list, K, N, CV settings
│   └── exp_*.json                # one file per experiment, overrides base
├── data/
│   ├── raw/                      # the 7 competition files, read-only
│   └── cache/                    # aligned.npz, sift_kp/*.npz, sift_pairs.npz, splits/*.json
├── src/reid/
│   ├── io.py                     # load + align ids, lazy SIFT loader
│   ├── audit.py                  # Gate 1 checks
│   ├── splits.py                 # open-set CV with near-duplicate grouping
│   ├── metrics.py                # MAP@5 + diagnostics
│   ├── global_feats.py           # whitening, regularised LDA, QE
│   ├── retrieval.py              # faiss search, identity aggregation
│   ├── kreciprocal.py            # Zhong et al. re-ranking
│   ├── viewpoint.py              # imputer + compatibility matrix
│   ├── sift_match.py             # RootSIFT, ratio test, RANSAC, pair features
│   ├── sift_cache.py             # shortlist + multiprocess cache builder
│   ├── candidates.py             # (query, identity) feature tables
│   ├── models.py                 # LightGBM / CatBoost / XGBoost / LR / MLP + stacker
│   ├── calibration.py            # temperature, P(new), prior shift EM, beta
│   ├── transductive.py           # test graph, consensus clusters, propagation
│   ├── decision.py               # probability sort, submission writer + validator
│   └── tracking.py               # experiment registry
├── scripts/
│   ├── 00_audit.py
│   ├── 01_build_sift_cache.py
│   ├── 02_run_cv.py --config configs/exp_X.json
│   ├── 03_make_submission.py --config configs/exp_X.json
│   └── check_compliance.py       # import/API allowlist scan
├── experiments/                  # registry.csv + one folder per run
├── reports/                      # eda.md, figures, final ablation table
├── submissions/                  # sub_<exp_id>.csv
└── tests/                        # test_metric, test_splits, test_leakage, test_compliance, test_submission
```

Module boundaries: `io` is the only module that touches raw files; `splits` is the only one that decides gallery/query membership; every model `fit` receives a `FoldData` object that physically does not contain query labels, so leaking them is a type error rather than a discipline problem.

## 12. Risks and mitigations

| Risk | Symptom | Mitigation |
| --- | --- | --- |
| CV easier than test (burst leakage, wrong novelty rate) | Leaderboard well below CV | Near-duplicate grouping, KS check against test, novelty rate from two estimates |
| Supervised projection overfits (MiewID may already know these species) | Whitened/LDA cosine worse than raw on CV | Keep raw cosine as a feature; shrinkage; accept only on CV |
| SIFT false matches on background or opposite flanks | Reranker over-trusts inliers on cross-side pairs | Mutual + ratio test, partial-affine RANSAC (no mirror), spatial-spread features, viewpoint compatibility features |
| Error propagation in the test graph | One wrong edge flips a whole cluster | Consensus clusters only, high edge threshold, α tuned on CV, hedge submission without it |
| Novelty-rate mismatch between CV and test | Too many / too few `new_individual` at rank 1 | Prior-shift correction, EM estimate, β offset per species |
| Rule violation slips in (banned import, `cv2` image functions, manual edits to predictions) | Academic misconduct | `check_compliance.py` in every run; submissions only written by `decision.py` |
| Compute blow-up in SIFT stage | Cache build takes many hours | K = 60 shortlist, multiprocessing, cached once, lazy fill |

## 13. Questions you should be able to answer

If you can answer these in your own words, you understand the design well enough to defend it.

1. **Why not train a softmax classifier over all identities?** One weight vector per identity from 1–2 images overfits, and the closed-world normalisation leaves no probability for "new". Retrieval plus a class-agnostic pair model learns from all pairs and handles new identities for free.
2. **Why does a random image-level train/validation split give misleading scores?** It never creates unseen identities, and burst near-duplicates land on both sides, so similarity looks far higher than on test.
3. **Why verify only the top-K candidates with SIFT?** Cost (all pairs ≈ 20M) and precision: extra candidates mostly add false matches, as the tiger paper found when re-ranking more than 20 hurt.
4. **What does RANSAC add over counting ratio-test matches?** It demands that matches agree on one geometric transform. Random spot-to-spot matches rarely do, so inliers are much stronger evidence than raw matches.
5. **Why partial affine instead of a full homography?** Fewer parameters (4) means it is stable with few matches and can't represent a reflection, so left-vs-right flank pairs don't fake a match.
6. **Why is sorting by probability optimal for MAP@5?** Expected score is Σ P(ℓ\_k)/k with decreasing weights; any out-of-order pair can be swapped for a strict gain (exchange argument / rearrangement inequality).
7. **Why not a fixed cosine threshold for `new_individual`?** The right threshold depends on species, viewpoint, keypoints and gallery size, and a threshold only decides rank 1. Calibrated P(new) places "new" at the rank it deserves.
8. **How can we use the test set without labels and stay within the rules?** Unsupervised operations on provided features (PCA, k-reciprocal, graph clustering, propagation) never use test labels; the same steps are simulated inside CV on query features.
9. **What is prior shift, and why does it matter here?** P(new) learned at the CV novelty rate is biased if test has a different rate; Bayes' rule rescales the odds by the ratio of rates.
10. **Why keep raw MiewID cosine even after learning a projection?** The projection is fitted on few multi-shot identities and may overfit; the reranker can choose which similarity to trust in which situation.
11. **How do left/right flanks of one identity get matched at all?** Only through global similarity and the reranker's learned tolerance for low similarity on opposite-side pairs; SIFT legitimately cannot match them.
12. **How does this plan follow Saket Anand sir's work?** Global ranking, flank-aware reordering and top-K SIFT re-ranking (tiger paper); pairwise supervision and open-set protocols (MFID/PFID); low-rank regularised metrics (ICIP 2016); identity hold-out evaluation and clustering-disagreement (AAS).

## 14. Sources

- Project files: ML Project Statements (Problem 3); Hybrid Tiger Re-ID (ICCVW 2019); MFID (arXiv 1811.00743); PFID (arXiv 1907.02642); Patterned species detection & recognition (arXiv 2005.02905); Metric-learning segmentation of patterned species (ICIP 2016); Active Learning with Ambiguity-Aware Sampling (arXiv 2511.06658).
- [WildFusion: calibrated similarity fusion (arXiv 2408.12934)](https://arxiv.org/pdf/2408.12934)
- [MiewID: Multispecies Animal Re-ID (arXiv 2412.05602)](https://arxiv.org/pdf/2412.05602) and [MiewID model card](https://huggingface.co/james-burgess/miewid)
- [AnimalCLEF 2025, 1st-place notebook (CEUR Vol-4038 paper 251)](https://ceur-ws.org/Vol-4038/paper_251.pdf)
- [AnimalCLEF 2025, 2nd-place notebook (CEUR Vol-4038 paper 253)](https://ceur-ws.org/Vol-4038/paper_253.pdf)
- [DS@GT at AnimalCLEF 2026: graph construction with global + local evidence (CEUR Vol-4283 paper 366)](https://ceur-ws.org/Vol-4283/paper366.pdf)
- [HotSpotter (arXiv 2508.17605)](https://arxiv.org/pdf/2508.17605)
- [k-reciprocal re-ranking, CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.html)
- [Three things everyone should know to improve object retrieval (RootSIFT), CVPR 2012](https://projet.liris.cnrs.fr/imagine/pub/proceedings/CVPR2012/data/papers/368_P3B-01.pdf)
- From memory, not looked up this session: Saerens, Latinne & Decaestecker (2002) on prior-shift EM; Zhou et al. (2004) on label propagation; Radenović et al. on learned whitening.

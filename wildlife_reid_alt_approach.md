

## Pipeline Flow

```
Inputs: MiewID (2,152-d) + RootSIFT (xy, desc) + Species / Viewpoint
   │
   ▼  [Per-Species Partition: Hyena vs. Leopard]
┌────────────────────────────────────────────────────────────────┐
│ 1. Multi-Space Global Retrieval + Orthogonal Discriminators    │
│    • Raw L2, Soft Power-Whitened PCA (512-d, pw=0.35), LDA-256 │
│    • k-Reciprocal Jaccard Reranking (l2 & lda, k1=20, k2=6)    │
│    • Gallery-only Dual RBF Kernel Ridge (γ=1.5, λ=0.5)         │
│    • Gallery-only Balanced Cosine-Ridge + Per-Flank Sub-Protos │
└────────────────────────────────────────────────────────────────┘
   │
   ▼
┌────────────────────────────────────────────────────────────────┐
│ 2. HotSpotter RootSIFT LNBNN (FAISS HNSW) + Candidate Pool     │
│    • Fold-masked identity LNBNN (K=10, K=3) & image-level max  │
│    • Candidate pool: Top-30 combined rank + Top-5 lnb_img      │
└────────────────────────────────────────────────────────────────┘
   │
   ▼
┌────────────────────────────────────────────────────────────────┐
│ 3. Wide + LNBNN-Guided SIFT RANSAC Verification                │
│    • Top-15 IDs × 8 gallery imgs + top-2 lnb_img gallery imgs  │
│    • Bidirectional Lowe ratio (0.80) + 4-DoF Partial-Affine    │
│      (10px & strict 5px inliers) + MAGSAC Homography           │
└────────────────────────────────────────────────────────────────┘
   │
   ▼
┌────────────────────────────────────────────────────────────────┐
│ 4. Two-Round Monotone LightGBM + Mixed-CV Training             │
│    • Trained jointly (species_is_leo) on CV v1 + CV v2 splits  │
│    • GroupKFold(5) by true query ID + monotone +1 constraints  │
│    • Round 2 stacks MiewID + SIFT kpgraph neighbour posteriors │
└────────────────────────────────────────────────────────────────┘
   │
   ▼
┌────────────────────────────────────────────────────────────────┐
│ 5. Calibration, Zhou Label Propagation & Dual Export           │
│    • Per-species softmax temp T_s + logit offset β_s (no EM)   │
│    • CV-gated Zhou et al. (2004) diffusion on MiewID + kpgraph │
│    • Exports submissions.csv + v4_test_probs.pkl (for blends)  │
└────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Components

| Stage | Key Design Choices | Role in Score Gain |
| --- | --- | --- |
| **Open-Set CV & Mixed Training** | Unbiased CDF-deconvolved $\hat{\pi}_s$ per species; two 5-split burst-clustered CV sets (`v1: 0.10` encounter-aware, `v2: 0.25` harder) concatenated for training; evaluated strictly on `v1` with `GroupKFold(5)` by true query identity. | Prevents burst leakage and avoids overfitting LightGBM to a single split difficulty. |
| **Global Retrieval + v3 Orthogonal Models** | Combines `l2`, unsupervised soft power-whitened `pcaw` (`d=512, pw=0.35`), gallery-only shrinkage `lda` (`d=256, γ=0.30`), `$k$-reciprocal` Jaccard (`l2` & `lda`), plus gallery-only **Dual RBF Kernel Ridge**, **Balanced Cosine-Ridge**, and **Per-Flank Sub-Prototypes**. | Adds global discriminative separation (multi-modal flanks & 391-shot hub control) orthogonal to nearest-neighbour cosines. |
| **HotSpotter LNBNN (`faiss-cpu`)** | One-vs-many RootSIFT descriptor search (`IndexHNSWFlat`, `k=50`) masked strictly to each fold's gallery; computes identity `K=10`, strict `K=3`, and image-level `lnb_img`. Adds top-5 `lnb_img` IDs to the top-30 candidate pool. | Recovers spot-pattern matches across the entire gallery that MiewID misses. |
| **Wide + LNBNN-Guided SIFT RANSAC** | BLAS bidirectional Lowe's ratio (`0.80`) + 4-DoF `estimateAffinePartial2D` (`10px` & strict `5px` inliers, mirror-rejecting) + `USAC_MAGSAC` homography across top-15 IDs × 8 images + top-2 `lnb_img` images. | Supplies high-precision geometric features (`sift_inl_max`, `sift_inl_strict_max`, `sift_n6`, spread, scale/rotation). |
| **Two-Round Monotone LightGBM** | Round 1 predicts $P(\text{match} \mid q, c)$ (monotone `+1` on similarities, KRR, LNBNN, SIFT) and $P(\text{new} \mid q)$. Round 2 stacks query-neighbour posteriors (`nb_p_max`, `nb_p_w`, `nb_known_conf`, `nb_agree`) over a combined **MiewID + unsupervised SIFT `kpgraph`**. | Leverages repeated sightings in the query pool while preventing noisy tree reversals. |
| **Calibration & Propagation** | Per-species conditional-logit temperature $T_s$ + CV-tuned novelty logit shift $\beta_s$ (**no prior-shift EM**), followed by CV-gated Zhou et al. (2004) label propagation on MiewID + `kpgraph` and MAP@5-optimal probability sorting. | Preserves raw probability calibration and outputs both `submissions.csv` and `v4_test_probs.pkl` for ensembling. |

---

## jump from `0.835 → 0.873`
1. **Added global RootSIFT LNBNN + `lnb_img` candidate expansion:** Removed v3's MiewID-only recall bottleneck.
2. **Expanded SIFT verification (2 → 8 + 2 LNBNN-guided gallery images) + strict `5px` inliers:** Caught matching poses within multi-shot identities.
3. **Dropped explicit prior-shift EM:** Kept raw calibrated `P(new)` probabilities aligned with MAP@5.
4. **Added mixed CV v1+v2 training + Round 2 `kpgraph` stacking** Improved generalization and linked repeated unseen/known test sightings via both MiewID and SIFT keypoints.

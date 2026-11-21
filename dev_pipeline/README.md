# Dev's pipeline (second half of the final team blend)

`wildlife_reid_solution.ipynb` is a self-contained notebook (executed outputs included).
Run it from this folder; it finds the competition files under the repository's `data/` folder (or on Kaggle) by itself.

1. **Validation design**: encounter-aware open-set splits (cosine > 0.8 components within an identity; 10 % of identities
   held out as new, ~40 % of encounters as known queries, one query image per encounter), 16 splits, leak-free training
   (rows whose query image is a query in the evaluated split are dropped).
2. **Retrieval**: centred + L2 MiewID cosine and a PCA(256) -> LDA (shrinkage 0.3) projection, fitted on the gallery only;
   identity score = max over the identity's images; top-20 candidates.
3. **SIFT verification** of the provided keypoints: RootSIFT, 2-NN + Lowe ratio 0.8, one-to-one, RANSAC homography and
   similarity inliers (top-5 identities x 20 images, ranks 6-10 x 2 images).
4. **LNBNN** one-vs-many keypoint matching (HotSpotter-style, faiss-cpu), the same features as `src/reid/lnbnn.py`.
5. Flank (left/right) features and query-query kNN graph features (transductive, no labels).
6. **Two-round LightGBM** (P(match), P(new); round 2 adds neighbours' round-1 predictions) + LambdaRank re-ranker,
   temperature-calibrated label probabilities (T fitted by NLL on the honest CV).

Honest CV MAP@5: 0.8532 without LNBNN, **0.8583 with LNBNN**. Outputs: `submission.csv` (this model alone) and
`test_probs.pkl` ({test_id: {label: probability}}), the input of the final blend:
```
py scripts/03_make_submission.py --blend dev_pipeline/test_probs.pkl submissions/sub_<v8_mix_lnb_kpg>.probs.pkl --w 0.5
```
Section 6b (self-training with pseudo-labelled test images) is kept for reference but switched off
(`SELF_TRAIN = False`): it scored lower on the public leaderboard (0.85829 vs 0.86164 for the same blend without it).

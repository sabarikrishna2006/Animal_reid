# CLAUDE.md — Wildlife Re-Identification (Hyena & Leopard, MAP@5)

Execution guide for a course Kaggle competition. Follow the architecture in `PLAN.md` gate by gate. Never redesign the pipeline without CV evidence, and keep `reports/gateN.md` concise and viva-ready after every gate.

---

## 1. Hard Compliance Rules (Zero Tolerance)
1. **Data:** Use ONLY `train.csv`, `test.csv`, `miewid_features.npy`, `miewid_ids.csv`, `sift_features_train.npz`, `sift_features_test.npz`, `sample_submission.csv`. No external data or source lookups.
2. **Libraries ONLY:** Python stdlib + `numpy, pandas, scipy, scikit-learn, statsmodels, imbalanced-learn, xgboost, lightgbm, catboost, networkx, opencv-python, faiss-cpu, matplotlib, seaborn`.
   - **Forbidden:** `torch, tensorflow, keras, jax, joblib, optuna, tqdm, pyyaml, pytest, pyarrow`, any AutoML, or pretrained models. Use JSON configs, `unittest`, and `.npy/.npz/.csv/.pkl` caches.
3. **OpenCV (`cv2`):** Keypoint matching & geometric verification ONLY (`BFMatcher`, `FlannBasedMatcher`, `NORM_L2`, `findHomography`, `estimateAffinePartial2D`, `estimateAffine2D`, `findFundamentalMat`, `perspectiveTransform`, `transform`, `RANSAC`, `USAC_MAGSAC`, `LMEDS`, `setNumThreads`). No detectors (`SIFT_create`), image I/O, or `cv2.dnn`.
4. **Models:** Classical ML, linear metric learning (PCA, LDA, NCA), clustering/graphs, GBDTs, `sklearn.neural_network.MLPClassifier` from scratch, and classical SIFT matching/RANSAC.
5. **Submissions:** Generated ONLY by `scripts/03_make_submission.py`. Never hand-edit predictions. Log any rule ambiguity in `reports/questions_for_TA.md` and stop.

---

we have to make a progressive steps for this competition, there is no need of one shotting in the first try, we will first try somemthing , improve it next time, with different architecture , models, then slowly we will reach the best architecture as the plan or you think.
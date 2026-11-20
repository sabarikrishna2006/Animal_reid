# Questions for the TA team

## Resolved (answered by the student, 2026-11-19)
1. **Train images on disk** (`data/train_images_*`): **not allowed.** The pipeline never reads them; `io.py` reads only
   the 7 competition files.
2. **Unlabelled test features**: **allowed.** Used for PCA-whitening (train ∪ test), the query–query graph, transductive
   features and label propagation. No test labels exist or are used.

## Still open (worth confirming)
3. **Leaderboard probe.** Is one submission of the unchanged `sample_submission.csv` acceptable to read the public
   novelty rate? (It isn't strictly needed any more: three internal estimates agree on π ≈ 0.33 hyena / 0.35 leopard.)

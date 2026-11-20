# Gate 1: EDA and integrity audit

Full EDA with figures: [eda.md](eda.md). This page is the short gate summary.

## What was built
- `src/reid/io.py`: the only reader of raw files. It asserts alignment (MiewID rows ↔ train/test ids), no duplicates,
  finite values, species/identity consistency, and SIFT dtypes/shapes/coordinate ranges. It packs the 19,908 small SIFT
  `.npy` files into three memory-mapped arrays (`data/cache/sift_{xy,desc,off}.npy`): loading went from 123 s to < 1 s.
- `scripts/00_audit.py`: runs the checks and prints keypoint statistics. All asserts pass.
- `eda/eda1…eda5_*.py`: exploratory analysis behind `reports/eda.md`.

## Key facts that drive later design
| Fact | Number | Design consequence |
|---|---|---|
| MiewID vectors are not unit-norm | norms 48–239 | L2-normalise; keep log-norm as a quality feature |
| Test is encounter-disjoint from train | 0 hyena / 2 leopard test images have a train twin at cos > 0.97 | CV must move whole encounters |
| Same animal, opposite flank | mean cos 0.28–0.34 vs 0.54 same flank | viewpoint/side features |
| SIFT: high precision, low recall | ≥ 6 inliers → 98–99.5% same animal; recall 31% hyena / 56% leopard | SIFT promotes candidates, should not demote |
| Leopard long tail | 99 one-shot identities; 43 identities hold 78% of images | retrieval, not per-class softmax |

## Viva questions
1. *Why pack SIFT into three arrays?* Opening ~20k files costs about 2 minutes every run. One contiguous array plus offsets
   is memory-mapped, so a worker reads only the slice for the image it needs.
2. *Why is the embedding norm a feature and not just discarded?* Retrieval accuracy rises with norm (hyena 78% → 92% across
   quartiles), so it carries information about image quality. L2-normalising for cosine removes it from the similarity,
   so we feed it back explicitly.
3. *What is the single most important EDA finding?* The test split is by encounter. Without it, a random-image CV would
   report about 0.85+ top-1 from burst twins that don't exist between train and test.

# Gate 2: Open-set CV and baseline submission

## What was built
- `src/reid/splits.py`: open-set CV. The only module that decides gallery vs query. `FoldData` (seen by every model)
  has no query labels. `FoldTruth` is only read by `metrics.py` and by the second-level training code.
- `src/reid/metrics.py`: MAP@5 (unit-tested on the official example: 0.4583) plus diagnostics (known-only MAP,
  `new_individual` recall/precision at rank 1, P(new) AUC, candidate recall).
- `scripts/tune_splits.py`: chooses split parameters so the CV queries look like the test set.
- `src/reid/decision.py`: the only submission writer. It always runs the validator: 2,791 ids, 1–5 unique labels,
  same species.
- `scripts/03_make_submission.py --baseline`: raw-cosine threshold baseline submission.

## How the CV split works (per species, 5 seeds)
1. **Encounters.** Inside each identity, average-linkage clustering on cosine distance $1-\cos$, cut at `enc_thr`.
   One cluster ≈ one burst/encounter, and clusters always move together.
2. **New animals.** A fraction `f_new_id` of identities (with ≤ 40 images, stratified by shot count) is removed
   from the gallery. All their images become queries with label `new_individual`.
3. **Known queries.** For the other identities with ≥ 2 encounters, each encounter moves to the query set with
   probability `p_move`. At least one stays in the gallery.

**Choosing the parameters (adversarial validation).** For each setting, compute every CV query's top-1 cosine to its
gallery and compare that distribution with the real test images' top-1 cosine to train, using the Kolmogorov–Smirnov
distance $KS=\max_t |F_{cv}(t)-F_{test}(t)|$ (0 = identical distributions). Grid: `enc_thr` ∈ {0, …, 0.6},
`p_move` ∈ {0.1, …, 0.6}, `f_new_id` ∈ {0.1, …, 0.6} (`reports/split_tuning.csv`).

| | chosen `enc_thr` | `p_move` | `f_new_id` | KS | implied novelty rate π_cv | query/gallery |
|---|---|---|---|---|---|---|
| Hyena | 0.2 | 0.3 | 0.15 | ≈ 0.065 | ≈ 0.30 | ≈ 0.6 (test 0.64) |
| Leopard | 0.2 | 0.25 | 0.4 | ≈ 0.08 | ≈ 0.38 | ≈ 0.5 (test 0.30) |

**Reading it.** `enc_thr` = 0.2 means images with cosine > 0.8 are glued together. KS was lowest there for both species
(0.3–0.4 was too strict, 0 too loose). π is not sharply identified: there is a ridge of good settings with
π from 0.15 to 0.30 (hyena) and 0.34 to 0.43 (leopard). This is why P(new) is later re-estimated on test by EM.

## Baseline
Rule: score each identity by the best cosine of its gallery images; output the top identities with score ≥ τ, then
`new_individual`, then the remaining identities (always 5 labels). τ is tuned per species on CV.

| Variant | CV MAP@5 (mean ± sd over 5 splits) | Hyena | Leopard | τ (h / l) |
|---|---|---|---|---|
| raw cosine, max | **0.8486 ± 0.0152** | 0.817 | 0.873 | 0.49 / 0.52 |

The overall CV number is weighted like the test set (43.6% hyena, 56.4% leopard).
Submission: `submissions/sub_20261119-060645_sub_baseline_rawcos.csv`. It puts `new_individual` at rank 1 for 42% of hyena
and 47% of leopard test images.

**Exit check.** The std across splits is 0.015, above the plan's 0.01 target. It is driven by *which* identities are held
out as new (a split-to-split effect, not noise in our models). So decisions use **paired** differences on the same 5
splits: the SE of paired deltas is 0.001–0.003, small enough to separate real gains.
LB check (CV within ±0.03 of LB) is for you to do: submit the baseline and the unchanged `sample_submission.csv`.

## Viva questions
1. *Why tune the split to match the test top-1 distribution?* The CV is only useful if it is as hard as the test. The
   top-1 cosine summarises how close each query's best match is. If CV queries had many burst twins, their top-1 would
   be much higher than the test's, and the KS distance shows it.
2. *Why is the CV std 0.015 not a problem?* Every experiment uses the same 5 splits, so we compare per-split
   differences. Their SE (≈ 0.002) is what decides acceptance.
3. *What does a sample-submission probe measure?* All-`new_individual` scores 1 exactly on new animals, so its public
   score = the fraction of new animals in the public 30%.

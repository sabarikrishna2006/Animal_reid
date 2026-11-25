# Kaggle Submissions Log (Team BLACKbox)

Every official Kaggle submission is tagged in Git with a unique version ID and submission date.

| Git Tag | Date | Kaggle Filename | Repo Submission File | Public LB | Submitted By | Notes |
|---|---|---|---|---|---|---|
| `sub-v01-2026-11-19-sample` | 2026-11-19 | `sample_submission.csv` | — | 0.31113 | Sabari | All `new_individual` (novelty baseline) |
| `sub-v02-2026-11-19-baseline` | 2026-11-19 | `sub_20261119-060645_sub_baseline_rawcos.csv` | `submissions/sub_20261120-205711_sub_baseline_rawcos.csv` | 0.76983 | Sabari | Raw cosine + threshold baseline |
| `sub-v03-2026-11-19-g5-final` | 2026-11-19 | `sub_20261119-132154_sub_g5_final.csv` | `submissions/sub_20261119-132154_sub_g5_final.csv` | 0.84116 | Sabari | Full Gate-5 pipeline (reranker + SIFT + propagation) |
| `sub-v04-2026-11-20-v3-sift` | 2026-11-20 | `sub_20261119-230723_sub_v3_sift.csv` | `submissions/sub_20261119-230723_sub_v3_sift.csv` | 0.83192 | Sabari | Wider SIFT tuned on CV v2 |
| `sub-v05-2026-11-20-team-blend` | 2026-11-20 | `sub_team_blend.csv` | — | 0.85813 | Devyaansh | 50/50 blend of Sabari v8 + Dev v5 |
| `sub-v06-2026-11-20-blend-prior` | 2026-11-20 | `sub_team_blend_prior.csv` | — | 0.85504 | Devyaansh | Prior-shift variant (dropped) |
| `sub-v07-2026-11-21-blend-v2b` | 2026-11-21 | `sub_team_blend_v2b.csv` | — | 0.85698 | Devyaansh | Shared-CV blend weights + original v8 (dropped) |
| `sub-v08-2026-11-21-blend-v2` | 2026-11-21 | `sub_team_blend_v2.csv` | — | 0.85538 | Devyaansh | Shared-CV blend weights + oursplits v8 (dropped) |
| `sub-v09-2026-11-21-blend-pseudo` | 2026-11-21 | `sub_team_blend_pseudo.csv` | — | 0.85829 | Devyaansh | Self-training variant (dropped) |
| `sub-v10-2026-11-22-blend-lnb6` | 2026-11-22 | `sub_team_blend_lnb6.csv` | `submissions/sub_20261122-155330_sub_team_blend_lnb6.csv` | 0.86749 | Devyaansh | Both pipelines + image-level & K=3 LNBNN + kpgraph + Zhou prop (`v9`) |
| **`sub-v11-2026-11-22-blend-lnb7`** | **2026-11-22** | **`sub_team_blend_lnb7.csv`** | **`submissions/sub_20261122-171705_sub_team_blend_lnb7.csv`** | **0.88073** | **Devyaansh** | **+ `lnb_img` candidate expansion + `lnb_img`-guided RANSAC SIFT + k-reciprocal (`v10`)** |

| `sub-v12-2026-11-22-blend-lnb10` | 2026-11-22 | `sub_team_blend_lnb10.csv` | — | 0.87794 | Lokesh | Early `v10` + `lnb_img` blend iteration (dropped) |
| `sub-v13-2026-11-22-blend-lnb11` | 2026-11-22 | `sub_team_blend_lnb11.csv` | — | 0.88034 | Lokesh | `nprobe=64` LNBNN + `lnb_uimg` background-keypoint filtering test |
| `sub-v14-2026-11-22-blend-lnb12` | 2026-11-22 | `sub_team_blend_lnb12.csv` | — | 0.87778 | Lokesh | Alternative `lnb_img` weighting / threshold test (dropped) |

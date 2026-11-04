# Rules v5 + manual corrections (2026-10-20): the current decision

Folder: `decisions/2026-10-20_105845Z_rules_v5_full_mother`, made by `typesafe_image_filter.py decide` on run `runs/2026-10-19_072143Z_full_mother`. It made no new API calls; the Jev answers are the same as for v4.

**The final list is `final_remove_list.txt`: 1,123 images.**
- `remove_list.txt` (1,211) is the rules-only result.
- 88 of those 1,211 were reinstated by the user after looking at them.

## 1. What changed in the factors, and why

| Factor | v4 | v5 | Why |
|---|---|---|---|
| Seams (Noul) | > 0.70 | **> 0.60** | In the score-band review, 0.60–0.70 was still mostly real seams (~8 of 12: strip boundaries, blocky aerial tiles). Below 0.60 they almost vanish (~1 of 12). Seams also show more clearly in the 1 m D3x10 targets. |
| Colour artefacts (Noul) | > 0.65 | **> 0.75** | Real defects are almost all EarthView rainbow halos or pastel casts, but Jev also flags plain haze. The 0.65–0.70 band was only ~3 of 12 real and 0.60–0.65 ~1 of 12. The user asked for this cut-off to be high. |
| P(clouds more than half) | ≥ 0.50 | ≥ 0.50 (same) | Above 0.5 the images are more than half cloud or heavy fog with no ground detail; below 0.5 that is rare. The score is bimodal with a gap here. |
| No-data (Noul) | > 0.70 | > 0.70 (same) | Below 0.7 Jev mostly fires on uniform dark open sea. The real defects in that range are strip seams, which the seams rule already catches. |

- The evidence is in `exploration/2026-10-20_cutoff_band_review/`: 12 random images per score band, fixed seed. The review is Claude's, not blind, with an uncertainty of about ±25% per band.
- Effect: v4 listed 1,207 and v5 lists 1,211.
  - Seams-driven removals went up and colour-driven removals went down (fewer EarthView, more SkySat and aerial). See `thesis_data/<this folder>/tables/t10_v4_v5_final_comparison.csv`.

## 2. Manual corrections by the user (88 reinstated)

The user looked at the rejected images and reinstated 88. All of them are train images, so validation and test were unaffected. Files:
- `manual_corrections_user_list.txt`: the user's list exactly as given (paths in the D3x10 `rejected/` folder).
- `manual_overrides.csv`: the 88 images with their rule reasons and scores.
- `final_decisions.csv`: `decisions.csv` plus `manual_override` and `final_decision`.
- `manual_corrections_summary.json`: the numbers below.

What the corrections say about Jev:

| Rule reason | Reinstated / flagged |
|---|---|
| Colour artefacts | 37 / 312 (12%) |
| Seams | 49 / 587 (8%) |
| No-data | 1 / 32 (3%) |
| Mostly clouds | 1 / 395 (0.3%) |

(An image can have more than one reason.)

- **By source:** aerial 43, EarthView 41, Maxar 3, SkySat 1.
- **Seams:** reinstated at a similar rate in every score band (0.60–0.70: 11/147, 0.70–0.80: 12/139, 0.80–0.90: 15/178, 0.90+: 11/123). So the new 0.60 cut-off did not cause extra wrong rejections near the threshold.
- **Colour:** reinstated most at 0.75–0.80 (15/77) and least at 0.90+ (4/61). This agrees with the review: colour is the least precise factor.
- **Clouds:** the rule was confirmed almost completely.

## 3. Where this was applied

- **Mother:** not changed. These are only lists, as always.
- **D3x10** (the 1 m / 10 m dataset in `Desktop\D3x10 build`): the 1,123 final removals are physically in `D3x10 build/rejected/`. The dataset holds the other 17,071 pairs (train 16,835 / validation 115 / test 121), and its `samples.csv` marks the 88 with `restored_by_manual_review`.
  - The copy on AUTH Aristotle holds the same kept set.

## Caveats for the thesis

- The manual review covered rejected images only; kept images were not re-checked by a human. So it measures false removals, not missed defects.
- The 250-image blind validation set (`thesis_data/review_2026-10-19_095519Z_rules_v4_full_mother`) was drawn from the v4 decision. Its labels still score the Jev answers, but compare them with v5 and the final decision when they are scored.
- Jev scores move by about ±0.05 between calls (test–retest), so images within about 0.05 of a cut-off are uncertain.

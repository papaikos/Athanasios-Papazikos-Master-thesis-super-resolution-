# Lab notebook: Jev image check for the maritime super-resolution dataset

Dataset: `Desktop\Maritime_RGB_Expanded_20261008` (Mother, D1, D2, D3). The dataset is only read, never changed.
Model: TypeSafe Jev (System One), called through `typesafe-sdk` with images enabled (`TypeSafe-Danger-Zone-Images: true`).
Each run is in `runs/`, each set of decisions in `decisions/`. Entries are newest last.

---

## 2026-10-16: Filter design and first D1 pilot

**Goal.** Find HR tiles that are not useful for a marine/ship/shoreline super-resolution model.

**Method.**
- Each HR tile (512x512, 2.5 m per pixel) is sent as a lossless PNG, together with the scale (a 50 m ship is about 20 px long).
- Seven questions per tile, in one request:
  - Nouls: `water`, `shoreline`, `vessels`, `port_structures`, `defects`;
  - Scores: `obscured` (cloud/fog/haze/smoke), `sharpness`.
- The decisions are rules in code, so the thresholds can change without new API calls.

**Live check.** Two tiles answered sensibly (a port, and a river shore). Each tile costs about 1,666 input tokens.

**Pilot 1** (`runs/2026-10-16_pilot1_D1`).
- 300 D1/train tiles, the first tile of each of the first 300 TARs. All 300 calls succeeded in 2 min 12 s.
- 499,661 input tokens, about $0.021.
- One tile per TAR over-sampled SkySat: about 51% of the pilot vs about 15% of the dataset.

Rules v1: discard when defects > 0.7, obscured >= 1.5, sharpness < 0.75, or no maritime subject. Review when defects > 0.3 or the maritime subject is uncertain.
Result: 138 keep, 50 review, 112 discard.

**What the contact sheets showed.**
- `sharpness` measured how much edge content a tile has, not focus. Native-resolution crops of the 17 tiles discarded only as blurry showed none really out of focus; 5 were good coastal tiles. The user agreed blur is not a big problem.
- Cloud tiles with obscured between 1.5 and 2.0 included good coast and reef scenes with partial cloud.
- The "possible defects" review (defects 0.3 to 0.7) held almost only clean port, ship and coast tiles.
- Jev's `water` answer was 0.1 to 0.6 on most grey or dark SkySat open sea, so it cannot separate open sea from inland land.
- Uncertain-maritime tiles with water < 0.5 were inland (desert, lava, forest).

Rules v2 (`decisions/2026-10-16_policy_v2_D1pilot`):
- obscured cut-off 2.0;
- no blur rule;
- no possible-defects review;
- uncertain tiles with water < 0.5 discarded;
- one `no_maritime_subject` label.

Result: 183 keep, 12 review, 105 discard. The v1 decision files were overwritten, but they can be recomputed from the pilot's `judgments.jsonl`.

At the user's request the work was moved to `Desktop\Jev check`.

---

## 2026-10-19: Plan for Mother

**Request.**
- Clean Mother: remove images with defects (fog is not a defect) and images that are mostly cloud.
- Suggest what else to remove and what else to ask Jev.

**Facts checked.**
- Mother has 18,194 images: train 17,952, validation 115, test 127. They are the same image IDs as D1 (checked by listing every TAR).
- Mother stores each HR image as float32. Quantizing it the way D1 does, `floor(clip(x,0,1)*255+0.5)` from `code/src/maritime_uint8.py`, gives D1's uint8 pixels exactly (8 of 8 random images checked).
- Each image links to one native TIF in `Mother/native/objects` (15,273 TIFs, 183 GB). Some SkySat TIFs are shared by several images.
- 12 files in `native/objects` are unfinished downloads (`.tif.partial`).
- In a 60-TAR random sample the images were 44% aerial, 36% earthview, 15% SkySat and 5% Maxar.

**Decisions by the user.**
- Fog and haze are kept, and are never a defect or "cloud".
- Questions removed from the proposed list: `compression`, `fog_haze`, `sun_glint`, `detail`. So blocky compression damage is no longer checked.
- Output: **a text file of the images to remove; nothing in the dataset is removed or changed.** A quarantine plan (rewriting TARs and the manifest) was discussed and then cancelled.
- This is research: every run, decision and change must be logged (timestamped run folders, question set and script snapshot with the API key blanked out, model versions, request ids, image hashes, token counts).

**Question set v2** (11 questions):
- `clouds` Score, 5 levels by coverage; fog/haze excluded.
- Defect Nouls `nodata`, `seams`, `color_artifacts`. Each says that fog, haze, clouds, glint, shadows, waves and calm water are not defects.
- `water`, `shoreline`, `vessels`, `port_structures` (unchanged).
- `scene` Choice; `vessel_count` Score; `large_ship` Noul.

**Rules v3** (to be checked in pilot 2).
- Remove when any defect Noul is above 0.7, or when Jev puts at least a 50% chance on clouds covering more than half of the image.
- Report, but do not remove, `no_maritime_subject` and `uncertain_maritime_subject` (rules as in v2), so the user can decide.

**Next.** Pilot 2: 300 images sampled at random (fixed seed) from all Mother images, so the sources appear in their real proportions; contact sheets per reason; then the full run.

---

## 2026-10-19: Pilot 2 on Mother (question set v2, rules v3)

**Code check.** `test_filter_offline.py` passed before the live run. It uses a mocked API on 6 real Mother images, with output in a temp folder, and checks:
- the request sent and the run logs;
- one deliberate failure being logged;
- resume after the failure and after a simulated interruption;
- `decide`, the contact sheets, and 9 rule cases.

The D1 pilot files were moved into `runs/2026-10-16_pilot1_D1` (logs reconstructed) and `decisions/2026-10-16_policy_v2_D1pilot`.

**Run** `runs/2026-10-19_071327Z_pilot2_mother`.
- 300 images drawn at random (seed 20261019) from all 18,194 Mother images.
- Sources: aerial 137, earthview 106, SkySat 43, Maxar 14, which matches the dataset mix.
- 300/300 answered, 0 failures, 2 min 28 s, model jev-1.13.0.
- 647,786 input tokens (about 2,160 per image), about $0.027.

**Decisions** `decisions/2026-10-19_071602Z_rules_v3_pilot2`.
- 17 of 300 listed for removal (5.7%): seams 9, color artifacts 5, mostly clouds 3, no-data 0.
- All 17 are from train.

**Visual check** (full-resolution views of the uncertain cases).
- Mostly clouds: 3 of 3 correct, all fully clouded SkySat frames. #291 is a flat grey frame with no ground visible (cloud or fog; unusable either way).
- Seams: 7 of 9 show clear strip or mosaic boundaries (#138, #233, #260, #270, #125, #223, #242). #85 is faint and #228 (dark sea) is doubtful.
- Color artifacts:
  - #160, #158 and #255 have rainbow tints or halos;
  - #42 and #44 (earthview sea) have a mild pink cast, which is borderline.
- Missed: #30 (earthview, color 0.67) has obvious rainbow halos but stays under the 0.7 cut-off.
  - A cut-off of 0.65 would add only #30 in this sample.
  - 0.6 would also add #72 and #103 (mild casts).
- Clouds: the highest kept value, 0.45 (#164), is dark open sea with no cloud. Jev gives featureless dark sea some cloud probability, so a cloud cut-off below 0.5 would wrongly remove such images.

**Candidates (reported, not removed).**
- `no_maritime_subject`: 21 images. About 16 are featureless open sea and 5 are inland (#52 desert, #53 and #253 forest, #183 town, #192 industrial yard).
- `uncertain_maritime_subject`: 7 images. Six are open sea; #278 is an atoll reef worth keeping.

**Estimate for all of Mother under rules v3:** about 5.7%, i.e. roughly 1,000 to 1,100 images listed.

**Open for the user:**
- Lower the color cut-off to 0.65?
- Also list inland images?
- Also list featureless open sea?
- Start the full run (about 18,194 images, about $1.65, 1.5 to 2.5 h). The cut-offs can be changed afterwards with `decide` alone.

---

## 2026-10-19: Full run started; rules v4; thesis data tooling

**User decisions.**
- Start the full run.
- Color-artifact cut-off 0.65.
- Report inland and featureless open sea separately.
- This is for the user's thesis: do not touch the original dataset, and collect all the data the thesis needs. Everything stays in `Desktop\Jev check`.

**Full run** `runs/2026-10-19_072143Z_full_mother`.
- Started 07:21 UTC on all 18,194 Mother images (train, validation, test), at 16 concurrent requests.
- Throughput is about 1.7 to 1.9 images/s.
- About 1.2% of requests hit the SDK's default 10 s timeout (`TypeSafeAPITimeoutError`, logged in `errors.jsonl`). They will be retried with `--resume`. `TIMEOUT_S = 60` was added for the retry session and is recorded per session in `run.json`; questions, model and image preparation are unchanged.

**Rules v4.**
- `color_artifacts` cut-off 0.65; `nodata` and `seams` stay at 0.7; clouds stay at 0.5.
- The old `no_maritime_subject` group is split by Jev's `scene` answer into `featureless_open_sea` (scene open_sea) and `inland` (any other scene). This is because `water` cannot separate grey SkySat open sea from land.
- Both groups, and `uncertain_maritime_subject`, are written to `candidates_*.txt`. They are kept, not listed for removal.
- On pilot 2 (`decisions/2026-10-19_072941Z_rules_v4_pilot2`): 18 listed (rules v3 plus #30), inland 5, featureless open sea 16, uncertain 7.

**Dataset metadata** (`thesis_data/mother_metadata.csv`, from `extract_metadata.py`, reading `Mother/manifest.json` only).
- 18,194 images, 76 countries, 410 sites, 407 geographic groups. Acquired 2010 to 2025; 5 images have no date.
- Train: aerial 7,574, earthview 6,466, SkySat 3,399, Maxar 513.
  - **Validation (115) and test (127) are all Maxar.**
  - No geographic group appears in more than one split.
- **Every image is still marked as awaiting review in the dataset's own provenance:**
  - 14,795 aerial, earthview and Maxar images have `requires_native_quality_review` and `requires_semantic_review` set to True, with `final_eligible` False;
  - all 3,399 SkySat images have status `pending_semantic_and_quality_review`.

  The Jev check is a first automated pass of this pending review.
- The country field mixes ISO codes and names; `country` holds the normalized ISO code and `country_raw` the original.

**Tooling for the thesis.**
- `thesis_report.py`: master table (metadata, every Jev answer and the decision, per image); tables t1 to t9; figures f1 to f4 (palette checked with the dataviz validator).
- `make_review_set.py` and `score_review.py`: a blind human-validation set and its scoring (precision, miss rate, recall, Wilson 95% intervals).
- `compare_runs.py`: test-retest agreement between two runs.
- All were tested on the pilot-2 data, with output in a temp folder.
- `references/typesafe_docs`: local copies of the TypeSafe docs used (retrieved 2026-10-16). The models and images pages were re-checked on 2026-10-19 and are unchanged.

**Preliminary test-retest.**
- 18 images were judged in both pilot 2 and the full run (so far). The decisions were identical (Cohen's kappa 1.0).
- The probabilities differ slightly, with a mean absolute difference of 0.004 to 0.03 per question and at most 0.10. So Jev's answers are not bit-identical across calls.
- All 300 pilot-2 images are also in the full run, so the full comparison needs no extra API calls.

---

## 2026-10-19: Full run on Mother: results (rules v4)

**Run** `runs/2026-10-19_072143Z_full_mother`.
- All 18,194 of 18,194 images judged, in two sessions:
  - 07:21 to 09:52 UTC at 16 concurrent requests: 17,962 answered; 232 timed out at the SDK's default 10 s.
  - 09:53 to 09:55 UTC, resumed at 8 concurrent with a 60 s timeout: 232 of 232 answered.
- 39,285,696 input tokens (mean 2,159 per image), about $1.65.
- Every answer came from jev-1.13.0.

**Decisions** `decisions/2026-10-19_095519Z_rules_v4_full_mother`.
- **1,207 images listed for removal (6.63%):** color artifacts 472, seams 428, mostly clouds 395, no-data 32 (images with more than one reason are counted under each; see t6).
- Kept but reported:
  - inland 518;
  - featureless open sea 1,604;
  - uncertain maritime subject 427.

| source | images | listed | % | main reasons |
|---|---|---|---|---|
| aerial (NAIP) | 7,574 | 248 | 3.3 | seams 237 |
| earthview | 6,466 | 428 | 6.6 | color 395 |
| SkySat | 3,399 | 511 | 15.0 | clouds 335, seams 158 |
| Maxar | 755 | 20 | 2.6 | seams 13 |

- Test: 3 of 127 listed (2 seams, 1 no-data). Validation: 0 of 115.
- SkySat makes up 1,354 of the 1,604 featureless-open-sea candidates, i.e. 40% of all SkySat images.

**Visual check of the borderline sheet pages.**
- No-data: mostly real (flat-filled mosaic blocks in aerial images, a cyan-cast earthview port). Uniform dark-blue SkySat tiles are either deep sea or fill; unusable either way.
- Seams, 0.70 to 0.75: most show strip boundaries. Several cloudy SkySat frames are flagged because of cloud edges.
- Color, 0.65 to 0.70: about half show rainbow or pastel casts; the rest look like normal, slightly hazy earthview images.
- Clouds, 0.50 to 0.60: about half are about 50% cloud. The rest are haze, fog or dark sea, which the user's rule does not count as cloud.
- Images listed for a single reason that sits in these borderline bands: color 71, seams 56, clouds 44. With cut-offs of color 0.70, seams 0.75 and clouds 0.60, the list would be 1,029 instead of 1,207.
- Score distributions (f1):
  - Clouds are bimodal, with a peak near 1.0, so the 0.5 cut-off sits in a natural gap.
  - Seams and color have a flat tail with no natural break, so their cut-offs are judgement calls that the human validation must support.
- Inland candidates, page 1 (most borderline): almost all are land (desert, industry, towns, lava, forest). #7828 is a built coast.
- Featureless-open-sea candidates, page 1:
  - mostly open sea, but it includes images with boats (#16234, #4753) and land tiles that Jev called open sea;
  - 54 open-sea candidates have maritime >= 0.4, and 10 have vessels >= 0.5. Check these before removing open sea.

**Test-retest** (`thesis_data/retest_2026-10-19_071327Z_pilot2_mother_vs_2026-10-19_072143Z_full_mother.json`).
- 300 images were judged twice (pilot 2 and the full run, same questions, about 2.5 h apart).
- Decisions agree on 299 of 300 (99.7%, Cohen's kappa 0.97).
- Mean absolute difference per question is 0.004 to 0.03; the largest is 0.22 (`vessel_count`). `scene` was identical for 98.3%.
- The one changed decision is pilot-2 #30 (earthview, rainbow halo): color 0.67 in pilot 2 vs 0.63 in the full run, across the 0.65 cut-off. So it is **kept** in the full-run list.
- Decisions within about 0.05 of a cut-off can change between calls.

**Thesis data** (`thesis_data/2026-10-19_095519Z_rules_v4_full_mother`): master table (18,194 rows), tables t1 to t9, figures f1 to f4, README.

**Blind validation set** (`thesis_data/review_2026-10-19_095519Z_rules_v4_full_mother`).
- 250 images: 100 drawn at random from the 1,207 listed and 150 from the 16,987 kept (seed 20261019), shuffled and renamed R001 to R250.
- `labels_to_fill.csv` is for the user's labels; `answer_key_open_after_labelling.csv` is kept separate.
- Waiting for the user to label it. Then `score_review.py` gives precision, miss rate and recall with 95% intervals.

## 2026-10-20 — cut-off review by score band (Claude, not blind; no dataset change)

Sheets and the sampled list are in `exploration/2026-10-20_cutoff_band_review/` (`band_sheets.py`, seed 20261020+band). There are 12 random images per score band on Mother's 2.5 m HR tiles. The fractions below are Claude's visual judgement and are not blind: about ±25% at n=12.

- **Seams** (share with a visible seam):
  - 0.90+: ~11/12; 0.70–0.80: ~7/12; 0.60–0.70: ~8/12 (strip boundaries, blocky aerial tiles, faint SkySat lines); 0.50–0.60: ~1–2/12; 0.40–0.50: ~1/12.
  - So the knee is near 0.60.
- **Colour artefacts:** mostly EarthView rainbow halos and pastel casts.
  - 0.80+: ~8/12; 0.70–0.80: ~6/12; 0.65–0.70: ~3–4/12; 0.60–0.65: ~1/12.
  - Haze is often mistaken for an artefact at every level.
- **Clouds** (P(more than half)):
  - 0.65–0.80: ~7/12 cloudy, and the rest are heavy haze or fog; 0.50–0.65: ~4/12 cloudy plus ~4/12 heavy haze; 0.35–0.50: ~1–2/12.
  - 0.50 is reasonable.
- **No-data:** at 0.30–0.70 it mostly fires on uniform dark open sea. The real defects in that range are strip seams that the seams score already catches. Keep 0.70.
- **User instruction (mid-review):** "set the colour artifacts high."
- **Recommended rule set (not applied):** no-data > 0.70, seams > 0.60, colour > 0.75, P(clouds > half) ≥ 0.50, giving 1,211 listed (train 1,205, test 6, validation 0). Current v4 lists 1,207.
  - Advice: apply the rules to train only, and leave validation and test fixed.

## 2026-10-20: rules v5 adopted, applied to D3x10, and manual corrections

**Factors changed** (user decision after the score-band review above; the user asked for colour to be high).
- `typesafe_image_filter.py` constants changed from v4 to v5: seams 0.70 → 0.60, colour artefacts 0.65 → 0.75. No-data (0.70) and clouds (≥ 0.50) are unchanged. The v4 values and the reason are kept in a code comment.
- `decide` was rerun on the same full run, with no API calls (typesafe-sdk 0.7.5, the same as the run) → `decisions/2026-10-20_105845Z_rules_v5_full_mother`.
- It lists 1,211 for removal, versus 1,207 in v4. Candidates: inland 519, featureless open sea 1,584, uncertain maritime subject 410.
- This list is identical to the 1,211 the D3x10 build rejected using the same thresholds.

**Why the factors changed** (evidence: `exploration/2026-10-20_cutoff_band_review/`).
- Seams: 0.60–0.70 was still mostly real seams, and below 0.60 they almost vanish.
- Colour: 0.65–0.70 was mostly haze that Jev mistakes for an artefact; real colour defects are mostly EarthView rainbow halos.
- Clouds and no-data: the existing cut-offs sit at natural breaks.

**Applied to D3x10.** The 1,211 rule-rejected pairs were physically moved into `Desktop\D3x10 build\rejected` (the D3x10 dataset only; Mother and D1–D3 untouched).

**Manual corrections by the user (88 reinstated).**
- The user reviewed the rejected images and asked for 88 to be reinstated, all of them train.
- Recorded in the v5 folder:
  - `manual_corrections_user_list.txt`, the list as given;
  - `manual_overrides.csv`;
  - `final_decisions.csv` (`manual_override`, `final_decision`);
  - `final_remove_list.txt` (**1,123**, sha256 `0d33e667…`);
  - `manual_corrections_summary.json`;
  - `MANUAL_CORRECTIONS.md`.
- Reinstated per flagged reason: colour 37/312 (12%), seams 49/587 (8%), no-data 1/32, clouds 1/395. By source: aerial 43, EarthView 41, Maxar 3, SkySat 1.
- Seams overrides are spread evenly over the score bands (about 8% each), so the lower 0.60 cut-off did not add near-threshold mistakes. Colour overrides are highest at 0.75–0.80 (15/77).
- Final: 17,071 keep / 1,123 remove (train 1,117, test 6, validation 0).
- D3x10 now holds the same 17,071, and its AUTH copy matches.

**Thesis data.** `thesis_report.py` and `make_contact_sheets.py` were run for the v5 folder.
- The master table gained `manual_override` and `final_decision`.
- New tables:
  - `t10_v4_v5_final_comparison.csv`;
  - `t11_v4_to_final_flow.csv`: v4 keep → final keep 16,860; v4 keep → final remove 127; v4 remove → final keep 211; v4 remove → final remove 996.

**Caveats.**
- The manual review measured false removals only; kept images were not re-checked.
- The blind 250-image set was drawn from v4 and is still unlabelled.

Earlier decision folders were not changed.

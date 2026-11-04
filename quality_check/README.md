# Jev check: automated quality check of the maritime tiles

This folder checks every HR tile of the maritime dataset (Mother, 18,194 tiles) with the TypeSafe Jev vision model. Jev answers questions about clouds, no-data areas, seams, colour artefacts, scene type, vessels and port structures.
- **The dataset is only read, never changed.** The results are lists of image names (`remove_list.txt`, `candidates_*.txt`).
- Mother and D1/D2/D3 share image IDs, so one list applies to all of them.
- The full history and the reason for each decision are in [`docs/jev_check_notebook.md`](../docs/jev_check_notebook.md).

## Final decision (2026-10-20): rules v5 plus manual corrections
- **Folder:** `decisions/rules_v5/`. Start with `MANUAL_CORRECTIONS.md`.
- **Rules v5:** remove a tile when no-data > 0.70, seams > 0.60, colour artefacts > 0.75, or P(clouds over half) ≥ 0.50 (`policy.json`).
- **Lists:**
  - `remove_list.txt`: the rules remove 1,211 tiles.
  - `manual_overrides.csv` / `manual_restore_list.txt`: 88 of those were reinstated after a manual look.
  - `final_remove_list.txt`: the final 1,123.
- **Where it was applied:** the D3x10 (1 m / 10 m) dataset, through `dataset/d3x10/reject_d3x10.py` and `restore_d3x10.py`. Mother and D1–D3 are unchanged.

## Running it
```bash
pip install -r requirements.txt
export TYPESAFE_API_KEY=...        # never write the key into a file
python typesafe_image_filter.py judge --run-name full_mother --dataset-dir "$MARITIME_DATA_ROOT/Mother"
python typesafe_image_filter.py judge --resume runs/<run folder>            # continue an interrupted run
python typesafe_image_filter.py decide --run-dir runs/<run folder> --name rules_v5
python make_contact_sheets.py decisions/<folder>
python extract_metadata.py "$MARITIME_DATA_ROOT/Mother"
python thesis_report.py decisions/<folder>
```
Judging and deciding are separate steps, so the rules can be re-tuned without paying for inference again. The model version is pinned (`jev-1.13.0`) because the thresholds were tuned against it.

| File | What it does |
|---|---|
| `typesafe_image_filter.py` | `judge`: asks Jev about each image and logs every run in `runs/`. `decide`: applies the rules and writes `decisions/`. |
| `make_contact_sheets.py` | Picture grids for a decisions folder, one per reason |
| `extract_metadata.py` | Per-image metadata from `Mother/manifest.json` → `thesis_data/mother_metadata.csv` |
| `thesis_report.py` | Master table, tables and figures for a decisions folder → `thesis_data/<folder>/` |
| `make_review_set.py`, `score_review.py` | Blind human-validation set and its scoring |
| `compare_runs.py` | Test–retest agreement between two runs |
| `test_filter_offline.py` | Offline test of the whole flow with a mocked API (no cost; needs `MARITIME_DATA_ROOT`) |

- Each run folder (`runs/`) holds `run.json`, `images.tsv`, `questions.json`, `script_snapshot.py`, `environment.txt`, `judgments.jsonl` and `errors.jsonl`.
- `runs/` and `thesis_data/` are outputs and are not in git.
- The key is read from `TYPESAFE_API_KEY` and never reaches a snapshot or a log.

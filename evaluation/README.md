# Evaluation

| Path | What it does |
|---|---|
| `predict.py` | Super-resolves any 8-bit RGB image (or a `(3, H, W)` float `.npy`) with a released model. Full frame, FP32, as in the thesis. |
| `comparisons/infer_sentinel2_d3.py` | D3 models on the 10 preserved native Sentinel-2 port scenes (no ground truth). |
| `comparisons/compare_models_skysat_d3.py` | D3 models on the 10 SkySat crops with a synthetic ×4 input, scored against the SkySat reference. |
| `comparisons/compare_d1_d3.py`, `compare_d2_d3.py` | D1 / D2 models on exactly the same 30 inputs, next to D3. |
| `comparisons/prepare_skysat_x10.py` | Builds the ×10 SkySat inputs (1 m reference, D3x10-degraded 10 m input, 2.5 m reference of the same footprint). |
| `comparisons/compare_x10_d3.py` | D3x10 (×10) against D3 (×4) on the same 30 scenes. Every prediction is computed twice and must match. |
| `analysis/analysis.py` | Thesis tables from the run archives: bootstrap confidence intervals, per-scene and per-site gains, Sentinel-2 diagnostics. |
| `analysis/analysis_x10grid.py` | ×4 and ×10 compared on a common 1 m and 2.5 m grid. |

The comparison scripts were run inside each run's archive folder, which holds `models/`, the comparison inputs and `training-run/`. Point them at such a folder with `RUN_ARCHIVE` (and `D3_ARCHIVE` for the D3 reference). The analysis reads all archives under `MARITIME_ARCHIVE_ROOT` and writes to `ANALYSIS_OUTPUT`.

The comparison runtime was torch 2.8.0+cpu, 6 threads, seed 42, deterministic algorithms. The panels use the Arial font (`C:/Windows/Fonts/arial.ttf`); change `FONT` on other systems.

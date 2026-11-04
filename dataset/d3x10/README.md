# D3x10: 1 m / 10 m maritime super-resolution pairs (×10, D3 recipe)

Final thesis dataset. It contains **only 1 m HR / 10 m LR pairs**. The source datasets (Mother, D1–D3, Jev check) are only read, never changed.

## Where things are
The scripts write next to themselves, so run them from this folder. None of these outputs go into git.

| What | Where |
|---|---|
| Build code | `build_d3x10.py`, `reject_d3x10.py`, `restore_d3x10.py` (this folder) |
| Inputs | the expanded dataset (`MARITIME_DATA_ROOT`) and the Jev master table (`JEV_MASTER_TABLE`, made by `quality_check/thesis_report.py`) |
| Shards, `manifest.json`, `samples.csv` | `Maritime_D3x10_1m10m/{train,validation,test}/d3x10-<split>-NNNNN.tar` |
| Rejected pairs (rules_v5: 1,211, minus 88 restored by manual review = 1,123) | `rejected/` (byte-identical NPZs, `rejected_samples.csv`, `policy.json`) |
| Each build run | `runs/<UTC>_<full\|dryrun>/`: `run.json` (args, environment, input SHA-256s), `shard_plan.json`, `script_snapshot.py`, `progress.log`, `records/` (one JSON per shard), `errors.jsonl`, `summary.json`, `samples.csv`, `manifest.json`, `verify_*.json` |
| Build history | `docs/d3x10_build_notebook.md` |

## Each sample (`<sample_id>.npz` inside a shard)
- `hr`: uint8 (3, 1280, 1280), 1 m, the 1,280 m core of the Mother tile
- `lr`: uint8 (3, 128, 128), 10 m, same core bounds (the same LR grid as D1–D3)
- `hr_mask`, `lr_mask`: all True (full original support)
- `provenance`: JSON. It holds the native object and its window, every D3 draw and seed, transforms, float and uint8 hashes, the checks, references to the Mother and D3 members, and the full Mother provenance.

## How a pair is made
1. Read the original GeoTIFF from `Mother/native/objects`. Its size and SHA-256 must match Mother's pin.
2. Integrate the 1,360 m context by exact physical overlap (`area_matrix`, float64). First to 2.5 m, which must reproduce Mother's stored context (max |Δ| ≤ 1e-6), then to 1 m (1,360 px). EarthView 16-bit TOA gets the same saved display mapping that Mother used.
3. Apply D3 with the PSF kept in metres. D3's σ of 1–3 px at 2.5 m becomes σ of 2.5–7.5 m, so 2.5–7.5 px at 1 m, with radius 30 m (61×61). Then reflect boundary, FFT convolution, an exact 10×10 area mean, and a 4 LR-px halo crop. Noise uses D3's σ and seed, followed by one clip.
4. Draws are re-derived from the seeds and must equal D3's stored draws. Each sample's distance to D3's 10 m LR is recorded; the pilot gave ≤ 2 DN, 57–72 dB.
5. Quantize with half-up rounding (`floor(clip(x)·255+0.5)`). Output NPZs and TARs are deterministic: a rebuild gives identical bytes.

## Excluding samples at load time (nothing is removed)
`samples.csv` holds one row per sample: the Jev master table (all metadata, every Jev answer, `decision`, `remove_reasons`, `candidates`) plus:
- `native_gsd_ge_1m`: True when the HR equals the native 1 m product (EarthView and NAIP 1 m), so there was no downsampling margin.
- `x10_*` build columns: shard and member, LR-vs-D3 agreement, PSF/noise draws, hashes and sizes.

Filter it any way you like and pass the kept `sample_id`s to the loader.

## Commands
```
python build_d3x10.py build  --out Maritime_D3x10_1m10m --workers 8 [--resume runs/<run>]
python build_d3x10.py verify --out Maritime_D3x10_1m10m --run-dir runs/<run>
python reject_d3x10.py --data Maritime_D3x10_1m10m
python restore_d3x10.py --data Maritime_D3x10_1m10m --list rejected/manual_restore_list.txt --full-run runs/<full run>
```
Copy `quality_check/decisions/rules_v5/manual_restore_list.txt` (the 88 images reinstated by hand) into `rejected/` before the restore step.

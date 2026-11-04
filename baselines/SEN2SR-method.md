# WEO-SAS pretrained inference implementation

- Script: `scripts/infer_pretrained_weo.py`.
- HF repository: `WEO-SAS/sen2sr`, revision `5262f02bd454915de7a77cb8627f581255186719`.
- Variant: the repository's `mamba-main` / `referencex4` full ten-band pipeline; this is the published main variant, not a claim of measured superiority against every model.
- SEN2SR source must be installed at `8e21bb669bcc6e8eb953a1fb24dfc5bb59dc18a4`.
- Uses the unmodified official HF `Model` entry point, all three backbones, all three Fourier hard constraints and strict default state-dictionary loads. No adaptation or fine-tuning.
- FP32 model and inference, TF32 disabled, no AMP, no synthetic model fallback.

## Band order and radiometry

The acquired NPZ files store `[B4,B3,B2,B8,B5,B6,B7,B8A,B11,B12]` and are explicitly reordered by their band names. The pinned upstream `referencex4.py` requires `[B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12]`; it extracts RGBN internally with indices `[2,1,0,6]`. Its output returns the same standard ten-band order. Evidence is saved in `weo-referencex4-upstream.py`, downloaded from the pinned ESAOpenSR commit. Do not feed the acquisition order directly into this pipeline.

Input is L2A harmonized surface reflectance (DN / 10000), including six originally 20 m bands bilinearly sampled on the preserved 10 m grid. No input clipping/stretch is applied. Raw ten-band float32 output is saved without clipping. Original and output PNGs use the identical RGB mapping `[B4,B3,B2]`, linear reflectance / 0.3 and clipping only for display; preserved original TCI is saved separately and is not the model input.

Each output directory contains `arrays.npz`, `original.png`, `weo_sen2sr.png`, and `original_tci.png`. Root `result.json` records per-scene source metadata, checksums, shapes implied by validated contracts, timing, peak GPU memory, input/output range, display clipping fraction, model hashes, software and source commit. It becomes `complete` only after all 15 validated model forwards.

## Execution

```sh
python scripts/infer_pretrained_weo.py --model-dir checkpoints/pretrained-weo-sen2sr --inputs outputs/pretrained-comparison-20260929/inputs --output outputs/pretrained-comparison-20260929/weo-results
```

Pre-downloaded model files avoid needing huggingface_hub at inference. Optional `--download-only` uses `snapshot_download` with the pinned revision and explicit allow-list. Runtime requires the established torch 2.5.1 CUDA / Mamba 2.2.4 stack, pinned sen2sr, safetensors, rasterio, numpy and Pillow. The script rejects unpinned sen2sr or a missing CUDA device.

## Local validation

Python syntax compilation passed. The actual completed acquisition manifest and all 15 NPZ files passed checksum, uniqueness, band-order, dtype, finite-value, shape, source and validity checks. Actual CUDA inference completed for all 15 scenes on the Standard NVIDIA L4 worker. The completed result receipt records each forward and its exact input/output hashes; all raw outputs have shape (10, 512, 512) and finite values. See sen2sr-result.json and the final delivery verification.json.


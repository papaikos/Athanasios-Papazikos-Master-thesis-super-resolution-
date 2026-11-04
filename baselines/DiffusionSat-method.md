# DiffusionSat official fMoW checkpoint inference requirements

Verified 2026-09-29. Official source commit: `68259b074dfa567e6cc4667799dab0e37477fcf2`.

## Released model selection and files

The task-compatible release is [fMoW Sentinel -> RGB ControlNet](https://zenodo.org/records/13756246), not text-only image generation or Texas temporal synthesis. It generates 512x512 RGB conditioned on multispectral imagery, text and metadata. One released checkpoint is documented, at 50,000 steps; no evidence ranks multiple released fMoW checkpoints as “best.”

The ZIP is 6,626,209,448 bytes, MD5 `d1becb3a20d2bda29df2c845e083bcb9`. URL: https://zenodo.org/api/records/13756246/files/controlnet_sd21_md7norm_fmow_condres256.zip/content

Remote ZIP central-directory inspection confirmed this archive already contains all inference components. **Do not download the separate 30GB single-image archive.** Its frozen 100,000-step SatUNet is already included in half precision; its config explicitly records that training-checkpoint path. Select the root `unet`, `text_encoder`, `vae`, `tokenizer`, `scheduler`, `feature_extractor`, `model_index.json`, and either `checkpoint-50000/controlnet` or root `controlnet` (only one copy). The former establishes the training step explicitly. Exclude duplicate weights, random states and scalers. Selected weights occupy about4.124GB uncompressed/3.815GB compressed plus tiny configs. `zip-list.json` records offsets, compressed sizes and CRCs. HTTP Range206 was tested successfully, enabling bounded member extraction. Validate each extracted member CRC and record SHA256; the whole-archive MD5 cannot be claimed without downloading the full archive.

Actual config: conditioning_in_channels=10, conditioning_scale=2, num_metadata=7; pipeline scheduler DDIM; custom diffusers version0.17.0.dev0.

## Exact imagery and normalization

[Stanford fMoW-Sentinel catalog](https://purl.stanford.edu/vg497cb6002) describes surface-reflectance imagery and90-day cloud composites. It confusingly calls this13bands although Sentinel L2A lacksB10. The released model discardsB10 plusB1/B9, so all ten surviving bands are available in S2_SR_HARMONIZED. Use the exact original L2A observations and128x12810m grid; do not substitute TCI or L1C. Single observations differ from the training cloud composites; disclose that distribution difference.

Upstream preprocessing: start13-channel order B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12; channel-specific normalize raw scaled DN, clip and quantize to uint8; ToTensor(/255); swap indices1and3; remove0,9,10. Therefore surviving tensor order is **B4,B3,B2,B5,B6,B7,B8,B8A,B11,B12**. This can be computed directly on the ten genuine bands using corresponding mean/std entries; no missing-band padding is necessary.

Mean by original13-band index:
1370.19151926,1184.3824625,1120.77120066,1136.26026392,1263.73947144,1645.40315151,1846.87040806,1762.59530783,1972.62420416,582.72633433,14.77112979,1732.16362238,1247.91870117.

Std by original13-band index:
633.15169573,650.2842772,712.12507725,965.23119807,948.9819932,1108.06650639,1258.36394548,1233.1492281,1364.38688993,472.37967789,14.3114637,1310.36996126,1087.6020813.

For each retained band: uint8(clip((DN-(mean-2std))/(4std)*255,0,255))/255. Input DN corresponds to reflectance*10000; do not divide by10000 before this formula. Then torchvision Resize256 bilinear antialias=True, CenterCrop256. For square128 input this is a full-footprint interpolation, yielding shape1x10x256x256; preserve exact1280m ground footprint.

## Metadata and call

Raw metadata vector: `[lon+180,lat+90,target_gsd,cloud_fraction,year-1980,month,day]`. Normalized vector is respectively raw divided by `[360,180,1,1,120,12,31]` then multiplied by1000. Use actual scene center/date/cloudfraction. Target_gsd=2.5 for512pixels over1280m; conditioning_gsd=10 as training code did (original observed spacing, even though conditioning is interpolated). Record this chosen output-grid setting; it is not an observed2.5m measurement.

For batch1 give `metadata` and `cond_metadata` each flat7-vector. Give image as tensor, **not a list**, to avoid temporal/multi-condition shape logic. Explicit `height=512,width=512` is required; the pipeline otherwise defaults dimensions to the256condition size. Tensor input is not automatically resized by prepare_image. The released condition encoder downsamples256to64 to match512output latent64.

Load custom `SatUNet` and custom `ControlNetModel` from the extracted directories and inject into `DiffusionSatControlNetPipeline.from_pretrained(archive_root,unet=...,controlnet=...)`. Use FP16 on CUDA as upstream notebook. Primary notebook example uses50steps,guidance_scale1.0; retain archive DDIM scheduler. Seed deterministically per image. Use truthful caption matching training grammar, e.g. 'a fmow satellite image of a port in Greece', only when location/class evidence supports it. Do not prompt fabricated vessel counts.

## Dependencies and caveats

Upstream setup: Python3.10,torch2.2.2,torchvision0.17.2,CUDA12.1 (11.8 alternative), editable bundled diffusers fork. requirements_remaining pins accelerate0.21.0,transformers4.31.0,einops0.7.0,torchmetrics1.2.1,pandas2.0.3,rasterio1.3.8,pyproj3.6.0,shapely2.0.2,webdataset0.2.48. Source is custom diffusers0.17.0.dev0, not modern PyPI diffusers. Old dependency ecosystem may require compatible Hugging Face Hub/Numpy pins; verify imports rather than assuming latest packages work. No dependency environment was installed during research.

This is stochastic conditional synthesis; shapes/boats can be invented or moved. 512output maps to2.5m sampling for this1280m footprint, not proven recovered2.5m spatial resolution. No HR truth exists for these examples. Single-frame recent scenes are outside the training90-day-composite/date regime. An L4 runtime estimate requires a one-image benchmark; no measured local GPU throughput was available in this read-only research.

## Pinned source references

- `samar-khanna/DiffusionSat@68259b074dfa567e6cc4667799dab0e37477fcf2:launch_scripts/launch_fmow_md_superres.sh:L6-L27`: model prior,13-3bands,512target/256condition.
- `samar-khanna/DiffusionSat@68259b074dfa567e6cc4667799dab0e37477fcf2:train_controlnet.py:L854-L919`: transforms and conditioning_gsd10.
- `samar-khanna/DiffusionSat@68259b074dfa567e6cc4667799dab0e37477fcf2:diffusionsat/data_util.py:L255-L291`: metadata construction.
- `samar-khanna/DiffusionSat@68259b074dfa567e6cc4667799dab0e37477fcf2:diffusionsat/data_util.py:L445-L453`: metadata scaling.
- `samar-khanna/DiffusionSat@68259b074dfa567e6cc4667799dab0e37477fcf2:diffusionsat/data_util.py:L541-L595`: spectral normalization, drop and RGB reorder.
- `samar-khanna/DiffusionSat@68259b074dfa567e6cc4667799dab0e37477fcf2:diffusionsat/pipeline_controlnet.py:L609-L668`: tensor preparation does not resize.
- `samar-khanna/DiffusionSat@68259b074dfa567e6cc4667799dab0e37477fcf2:diffusionsat/pipeline_controlnet.py:L688-L710`: metadata batch conventions.
- `samar-khanna/DiffusionSat@68259b074dfa567e6cc4667799dab0e37477fcf2:notebooks/controlnet_texas_samples.ipynb`: loading pattern and50step/guidance1example, different task.

Source URLs use https://github.com/samar-khanna/DiffusionSat/blob/68259b074dfa567e6cc4667799dab0e37477fcf2/ followed by the relevant path and line anchors.

## Actual execution and quality review

All 15 CUDA forwards completed on NVIDIA L4 with finite 512 x 512 RGB outputs and saved fixed-seed receipts. The first Piraeus forward took 5.81 seconds. See diffusionsat-result.json and completion.json for actual execution evidence. Desktop checkpoint hashes and sizes match all model members used by cloud inference.

Visual inspection shows substantial invented or distorted structures. The results are poor generative reconstructions, not faithful super-resolution. A bounded independent source review found no concrete mismatch in the inspected preprocessing, band ordering, tensor geometry, metadata vector handling, checkpoint combination or DDIM v_prediction configuration. Domain/scale/date shifts are possible contributors, but the cause is not established by this review. No seed selection, post-processing or replacement imagery was used to conceal this outcome.

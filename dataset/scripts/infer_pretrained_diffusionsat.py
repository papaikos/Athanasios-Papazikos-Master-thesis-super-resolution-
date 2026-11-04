"""Official fMoW DiffusionSat inference on fixed genuine Sentinel-2 scenes.

The 2.5 m output grid is conditional synthesis, not measured spatial resolution.
The original 128x128 footprint is preserved by interpolating the spectral
conditioning tensor to the released model's 256x256 conditioning shape.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import subprocess
import time

COMMIT = '68259b074dfa567e6cc4667799dab0e37477fcf2'
BANDS = ['B4', 'B3', 'B2', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
INDEX13 = [3, 2, 1, 4, 5, 6, 7, 8, 11, 12]
COUNTRIES = {'piraeus': 'Greece', 'zea': 'Greece', 'flisvos': 'Greece',
             'barcelona': 'Spain', 'monaco': 'Monaco', 'marseille': 'France',
             'valencia': 'Spain', 'palma': 'Spain', 'genoa': 'Italy',
             'livorno': 'Italy', 'naples': 'Italy', 'split': 'Croatia',
             'dubrovnik': 'Croatia', 'malta': 'Malta', 'limassol': 'Cyprus'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=15)
    parser.add_argument('--seed', type=int, default=20260929)
    args = parser.parse_args()
    actual_commit = subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip()
    assert actual_commit == COMMIT, 'Official source revision differs from verified recipe'
    # Exact checked-out official source and bundled diffusers fork take precedence.
    sys.path[:0] = [str(args.source.resolve()), str((args.source / 'src').resolve())]
    import numpy as np
    import torch
    import torchvision
    from PIL import Image
    from diffusionsat import SatUNet, ControlNetModel, DiffusionSatControlNetPipeline
    from diffusionsat.data_util import SentinelNormalize, metadata_normalize
    assert torch.cuda.is_available(), 'A CUDA runtime is required for this bounded inference job'
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.inputs / 'manifest.json'
    manifest = json.loads(manifest_path.read_text('utf-8'))
    assert manifest['completed'] == 15
    download = json.loads((args.weights / 'download-receipt.json').read_text())
    assert download['status'] == 'complete' and download['source_commit'] == COMMIT
    for item in download['members']:
        assert sha(args.weights / item['path']) == item['sha256'], item['path']
    dtype = torch.float16
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    unet = SatUNet.from_pretrained(str(args.weights / 'unet'), torch_dtype=dtype, local_files_only=True)
    controlnet = ControlNetModel.from_pretrained(str(args.weights / 'checkpoint-50000/controlnet'),
                                                torch_dtype=dtype, local_files_only=True)
    assert controlnet.config.conditioning_in_channels == 10
    assert controlnet.config.conditioning_scale == 2
    pipe = DiffusionSatControlNetPipeline.from_pretrained(
        str(args.weights), unet=unet, controlnet=controlnet,
        torch_dtype=dtype, local_files_only=True).to('cuda')
    assert pipe.scheduler.__class__.__name__ == 'DDIMScheduler'
    # Attention slicing avoids materializing very large attention score tensors.
    pipe.enable_attention_slicing()
    resize = torchvision.transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                                            antialias=True)
    means = np.asarray(SentinelNormalize.mean)[INDEX13, None, None]
    stds = np.asarray(SentinelNormalize.std)[INDEX13, None, None]
    common = {'model': 'DiffusionSat fMoW SentinelRGB ControlNet checkpoint-50000',
              'source_commit': COMMIT, 'input_manifest_sha256': sha(manifest_path),
              'weight_download_receipt_sha256': sha(args.weights / 'download-receipt.json'),
              'script_sha256': sha(__file__), 'steps': 50, 'guidance_scale': 1.0,
              'scheduler': 'DDIMScheduler', 'dtype': 'float16', 'bands': BANDS,
              'native_input_grid_m': 10, 'interpolated_condition_grid_m': 5,
              'conditioning_metadata_gsd_m': 10, 'output_sampling_m': 2.5,
              'conditioning_shape': [1, 10, 256, 256], 'output_shape': [512, 512, 3],
              'normalization': 'official per-band mean +/- 2 std, clip and quantize uint8 / 255',
              'cloud_metadata': 'local SCL cloud classes 8/9/10 fraction, not scene cloud percentage',
              'no_ground_truth': True, 'generative_details_are_not_measurements': True,
              'native20m_bands_were_resampled_to10m': True,
              'distribution_shift': 'single recent L2A observation, not training 90-day composite',
              'environment': {name: importlib.metadata.version(name) for name in
                              ['torch', 'torchvision', 'diffusers', 'transformers', 'numpy']},
              'gpu': torch.cuda.get_device_name()}
    records = []
    for i, row in enumerate(manifest['examples'][:args.limit]):
        receipt_path = args.output / (row['id'] + '.json')
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            assert receipt['input_sha256'] == row['sha256']
            assert receipt['script_sha256'] == common['script_sha256']
            assert sha(args.output / receipt['png']) == receipt['png_sha256']
            assert sha(args.output / receipt['npz']) == receipt['npz_sha256']
            records.append(receipt)
            continue
        source_path = args.inputs / row['path']
        assert sha(source_path) == row['sha256']
        with np.load(source_path, allow_pickle=False) as data:
            source_bands = data['bands'].tolist()
            reflectance = data['reflectance'][[source_bands.index(b) for b in BANDS]]
            assert reflectance.shape == (10, 128, 128) and np.isfinite(reflectance).all()
            dn = reflectance * np.float32(10000)
            quantized = np.clip((dn - (means - 2 * stds)) / (4 * stds) * 255, 0, 255).astype(np.uint8)
            condition = resize(torch.from_numpy(quantized).float() / 255).unsqueeze(0)
            cloud = float(np.isin(data['scl'], [8, 9, 10]).mean())
        p = row['provenance']
        date = datetime.strptime(p['date'], '%Y%m%d')
        raw = [p['longitude'] + 180, p['latitude'] + 90, 2.5, cloud,
               date.year - 1980, date.month, date.day]
        cond_raw = [*raw[:2], 10.0, *raw[3:]]
        metadata = metadata_normalize(raw).tolist()
        cond_metadata = metadata_normalize(cond_raw).tolist()
        prompt = f"a fmow satellite image of a port in {COUNTRIES[row['id']]}"
        seed = args.seed + i
        generator = torch.Generator(device='cuda').manual_seed(seed)
        torch.cuda.synchronize()
        start = time.monotonic()
        with torch.inference_mode():
            result = pipe(prompt=prompt, image=condition.to(device='cuda', dtype=dtype),
                          metadata=metadata, cond_metadata=cond_metadata,
                          height=512, width=512, num_inference_steps=50, guidance_scale=1.0,
                          generator=generator, output_type='np').images[0]
        torch.cuda.synchronize()
        seconds = time.monotonic() - start
        result = np.asarray(result, dtype=np.float32)
        assert result.shape == (512, 512, 3) and np.isfinite(result).all()
        png = args.output / (row['id'] + '-DiffusionSat.png')
        npz = args.output / (row['id'] + '-DiffusionSat.npz')
        Image.fromarray(np.round(np.clip(result, 0, 1) * 255).astype(np.uint8)).save(png)
        np.savez_compressed(npz, rgb=result)
        receipt = {**common, 'id': row['id'], 'input_sha256': row['sha256'],
                   'source': row,
                   'prompt': prompt, 'seed': seed, 'seconds': seconds,
                   'metadata_raw': raw, 'conditioning_metadata_raw': cond_raw,
                   'metadata_normalized': metadata, 'conditioning_metadata_normalized': cond_metadata,
                   'png': png.name, 'png_sha256': sha(png), 'npz': npz.name, 'npz_sha256': sha(npz),
                   'files': {png.name: {'path': png.name, 'sha256': sha(png)},
                             npz.name: {'path': npz.name, 'sha256': sha(npz)}},
                   'completed_utc': datetime.now(timezone.utc).isoformat(),
                   'min': float(result.min()), 'max': float(result.max())}
        write_json(receipt_path, receipt)
        records.append(receipt)
        write_json(args.output / 'results.json', {'status': 'complete' if len(records) == 15 else 'in_progress',
                                                'completed': len(records), 'examples': records})
        print(json.dumps({'completed': len(records), 'id': row['id'], 'seconds': seconds}), flush=True)
    write_json(args.output / 'results.json', {'status': 'complete' if len(records) == 15 else 'partial',
                                            'completed': len(records), 'examples': records})


if __name__ == '__main__':
    main()

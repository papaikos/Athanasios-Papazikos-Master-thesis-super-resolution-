"""Direct native Sentinel-2 RGB -> frozen D3 x4 models; no reference metrics."""
import html
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(os.environ.get('RUN_ARCHIVE', Path(__file__).resolve().parents[1]))  # archive folder of one run, e.g. 'D1 run'
CHECKOUT = ROOT / 'reproducibility/checkout'
if not (CHECKOUT / 'research/model_factory.py').is_file():
    CHECKOUT = Path(__file__).resolve().parents[2] / 'training/x4/source'  # same SwinIR code in this repo
sys.path.insert(0, str(CHECKOUT))
from research.checkpointing import sha256
from research.model_factory import make_model


def image(array):
    return Image.fromarray(np.rint(np.clip(array.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8))


def panel(path, sample_id, lr, final, best, detail=False):
    # Corresponding central crop fixed before prediction inspection: 32 input pixels.
    if detail:
        lr = lr[:, 48:80, 48:80]
        final = final[:, 192:320, 192:320]
        best = best[:, 192:320, 192:320]
    canvas = Image.new('RGB', (1536, 610), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 19)
    labels = ['Original Sentinel-2 input (10 m)', 'Model 1: final, step 100000', 'Model 2: best, step 82000']
    for i, (label, array) in enumerate(zip(labels, [lr, final, best])):
        draw.text((i * 512 + 8, 8), label, fill='black', font=font)
        draw.text((i * 512 + 8, 34), 'Input enlarged for display only' if i == 0 else 'Nominal 2.5 m output grid', fill='#555555', font=font)
        canvas.paste(image(array).resize((512, 512), Image.Resampling.NEAREST), (i * 512, 64))
    draw.text((8, 581), sample_id + (' | corresponding central detail' if detail else '') + ' | Inferred detail; no independent HR ground truth.', fill='black', font=font)
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', default='sentinel2-comparisons', help='Input/output group folder within this archive')
    args = parser.parse_args()
    group = (ROOT / args.folder).resolve()
    if group.parent != ROOT.resolve():
        raise ValueError('Group must be a direct archive subfolder')
    source_path = group / 'inputs/sources.json'
    sources = json.loads(source_path.read_text(encoding='utf-8'))['sources']
    models = json.loads((ROOT / 'models/model-identities.json').read_text(encoding='utf-8'))
    if len(sources) != 10:
        raise ValueError('Exactly ten preserved Sentinel-2 inputs required')
    torch.set_num_threads(6)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.use_deterministic_algorithms(True)
    model, model_info = make_model(CHECKOUT / 'vendor/network_swinir.py')
    model.eval().requires_grad_(False)
    records = []
    for source in sources:
        sample_id = source['id']
        if Path(sample_id).name != sample_id:
            raise ValueError('Invalid sample ID')
        source_file = Path(source['reference_path'])
        if sha256(source_file) != source['reference_png']['sha256']:
            raise ValueError('Original source changed')
        with Image.open(source_file) as original:
            if original.mode != 'RGB' or original.size != (128, 128):
                raise ValueError('Native 128x128 RGB source required')
            pixels = np.asarray(original).copy()
        lr = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255
        out = group / 'results' / sample_id
        out.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(out / 'LR-input.png')
        np.save(out / 'LR-input-fp32.npy', lr.numpy())
        assert np.array_equal(np.asarray(Image.open(out / 'LR-input.png')), pixels)
        predictions = {}
        runtimes = {}
        for label in ['final', 'best']:
            identity = models[label]
            weights = ROOT / 'models' / Path(identity['weights_path']).name
            if sha256(weights) != identity['weights_sha256']:
                raise ValueError('Pinned weights changed')
            state = torch.load(weights, map_location='cpu', weights_only=True)
            model.load_state_dict(state, strict=True)
            started = time.perf_counter()
            with torch.inference_mode():
                prediction = model(lr[None])[0].float().cpu()
            if prediction.shape != (3, 512, 512) or not torch.isfinite(prediction).all():
                raise ValueError('Invalid direct reconstruction')
            if any(not torch.equal(state[k], value) for k, value in model.state_dict().items()):
                raise ValueError('Inference mutated model state')
            runtimes[label] = time.perf_counter() - started
            predictions[label] = prediction.numpy()
            np.save(out / (label + '-raw-fp32.npy'), prediction.numpy())
            image(prediction.numpy()).save(out / (label + '.png'))
            print(json.dumps({'sample': sample_id, 'model': label, 'output_shape': list(prediction.shape), 'seconds': runtimes[label]}), flush=True)
        panel(out / 'comparison.png', sample_id, lr.numpy(), predictions['final'], predictions['best'])
        panel(out / 'detail.png', sample_id, lr.numpy(), predictions['final'], predictions['best'], detail=True)
        record = {'id': sample_id, 'source_path': str(source_file), 'source_sha256': sha256(source_file),
                  'source_provenance': source['provenance'], 'input_shape': list(lr.shape), 'output_shape': [3, 512, 512],
                  'input_pixel_m': 10, 'nominal_output_grid_m': 2.5, 'seconds': runtimes,
                  'detail_crop_input_xyxy': [48, 48, 80, 80], 'detail_crop_output_xyxy': [192, 192, 320, 320],
                  'outputs': {p.name: {'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in out.iterdir() if p.is_file() and p.name != 'provenance.json'}}
        (out / 'provenance.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        records.append(record)
    protocol = {'schema': 'd3-sentinel2-direct-inference-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'count': len(records), 'mode': 'direct native Sentinel-2 inference',
                'preprocessing': 'Original uint8 TCI RGB /255; no degradation, resizing, cropping, enhancement or fitting before inference.',
                'inference': 'CPU FP32, full-frame, eval+inference_mode, seed42, deterministic algorithms, pinned strict model weights, model-state immutability checked. One forward per model and scene; repeat inference not performed.',
                'ground_truth': 'Unavailable. No HR reference, error maps or PSNR/SSIM/LPIPS scores computed.',
                'resolution_limit': '128x128 native10m input ->512x512 nominal2.5m output grid; no independent evidence of2.5m resolving power or true added physical detail.',
                'runtime': {'torch': torch.__version__, 'numpy': np.__version__, 'threads': torch.get_num_threads()},
                'source_manifest_sha256': sha256(source_path), 'model_architecture': model_info, 'models': models, 'samples': records}
    (group / 'inference.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    cards = []
    for record in records:
        sample_id = record['id']
        cards.append('<h2>' + html.escape(sample_id) + '</h2><img style="max-width:100%" src="results/' + sample_id + '/comparison.png"><p><a href="results/' + sample_id + '/detail.png">Corresponding detail</a> | <a href="results/' + sample_id + '/provenance.json">Provenance</a></p>')
    (group / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Direct Sentinel-2 inference</title><body style="font-family:Arial;max-width:1550px;margin:2em auto"><h1>Direct Sentinel-2 inference: final and best D3 models</h1><p>Original native128x12810m RGB inputs, directly reconstructed to512x512 nominal2.5m output grids. Input enlarged for display only. Inferred details are unvalidated; no independent HR reference, error maps or quality metrics.</p>' + ''.join(cards) + '</body>', encoding='utf-8')
    print(json.dumps({'status': 'COMPLETE', 'count': len(records), 'mode': 'direct inference'}), flush=True)


if __name__ == '__main__':
    main()

"""Reproduce the user-approved external synthetic x4 demonstration, locally."""
import argparse
import csv
import hashlib
import html
import json
import os
import sys
import time
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
from research.metrics import score_image, aggregate
from research.lpips_metric import LPIPSMetric


def degrade(reference):
    if reference.ndim != 3 or reference.shape[0] != 3:
        raise ValueError('CHW RGB reference required')
    height, width = reference.shape[-2:]
    height, width = height // 4 * 4, width // 4 * 4
    if min(height, width) < 64:
        raise ValueError('Reference is too small for LPIPS')
    hr = reference[:, :height, :width].contiguous()
    lr = torch.nn.functional.interpolate(hr[None], size=(height // 4, width // 4),
        mode='bicubic', align_corners=False, antialias=True)[0]
    # The actual network input has a lossless PNG representation.
    lr = lr.clamp(0, 1).mul(255).round().div(255)
    return hr, lr, [0, 0, width, height]


def error_map(prediction, reference):
    error = np.abs(np.clip(prediction, 0, 1) - reference).mean(axis=0)
    return np.rint(np.clip(error / 0.25, 0, 1) * 255).astype(np.uint8)


def output_inventory(directory):
    return {p.name: {'bytes': p.stat().st_size, 'sha256': sha256(p)}
            for p in directory.iterdir() if p.is_file() and p.name != 'provenance.json'}


def detail_crop(sample_id, height, width):
    # Illustrative display crops identified from source-reference maritime content.
    # They affect no full-frame metrics or sample/model selection.
    maritime = {'skysat-04':[192,16,256,80], 'skysat-05':[0,96,64,160],
                'skysat-06':[192,64,256,128], 'skysat-07':[72,192,136,256],
                'skysat-08':[80,112,144,176], 'skysat-09':[40,0,104,64]}
    if sample_id in maritime:
        return maritime[sample_id]
    side = min(64, height, width)
    x, y = (width-side)//2, (height-side)//2
    return [x,y,x+side,y+side]


def rgb(array):
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    return Image.fromarray(np.rint(np.clip(array.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8))


def font(size):
    path = Path('C:/Windows/Fonts/arial.ttf')
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def panel(path, sample_id, hr, lr, final, best, crop=None):
    hr = hr.numpy()
    enlarged = torch.nn.functional.interpolate(lr[None], size=hr.shape[-2:],
        mode='bicubic', align_corners=False, antialias=False)[0].numpy()
    arrays = [rgb(hr), rgb(final), rgb(best), rgb(enlarged),
              Image.fromarray(error_map(final, hr)).convert('RGB'),
              Image.fromarray(error_map(best, hr)).convert('RGB')]
    if crop:
        arrays = [a.crop(crop) for a in arrays]
    side = 256
    canvas = Image.new('RGB', (side * 3, (side + 54) * 2 + 100), 'white')
    draw = ImageDraw.Draw(canvas)
    labels = ['HR source reference', 'Model 1: final (100000)', 'Model 2: best (82000)',
              'LR: bicubic display x4', 'Mean absolute error: final', 'Mean absolute error: best']
    for index, (label, image) in enumerate(zip(labels, arrays)):
        x, y = index % 3 * side, index // 3 * (side + 54)
        draw.text((x + 5, y + 5), label, fill='black', font=font(16))
        draw.text((x + 5, y + 27), 'Display only; no added resolution', fill='#555555', font=font(12))
        canvas.paste(image.resize((side, side), Image.Resampling.NEAREST), (x, y + 54))
    y = (side + 54) * 2
    draw.text((5, y + 5), sample_id + (' | reference-detail crop' if crop else ''), fill='black', font=font(16))
    draw.text((5, y + 30), 'Synthetic x4 experiment; source is reference, not independent HR satellite truth.', fill='black', font=font(15))
    draw.text((5, y + 54), 'Both error panels: mean |RGB error|, fixed 0 (black) to 0.25 (white); clipped above 0.25.', fill='black', font=font(14))
    bar = Image.fromarray(np.tile(np.arange(256, dtype=np.uint8), (12, 1))).convert('RGB')
    canvas.paste(bar, (5, y + 80))
    draw.text((270, y + 76), '0 -------------------- 0.25', fill='black', font=font(13))
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', required=True, choices=['skysat'])
    parser.add_argument('--sources', type=Path, required=True)
    args = parser.parse_args()
    root = ROOT
    group = root / (args.group + '-comparisons')
    models = json.loads((root / 'models/model-identities.json').read_text(encoding='utf-8'))
    source_data = json.loads(args.sources.read_text(encoding='utf-8'))
    sources = source_data if isinstance(source_data, list) else source_data['sources']
    if len(sources) != 10:
        raise ValueError('Exactly ten documented source references required')
    torch.set_num_threads(6)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.use_deterministic_algorithms(True)
    metric = LPIPSMetric('cpu')
    model, model_info = make_model(CHECKOUT / 'vendor/network_swinir.py')
    model.eval().requires_grad_(False)
    all_rows, sample_records = [], []
    for source in sources:
        sample_id = str(source['id'])
        if Path(sample_id).name != sample_id:
            raise ValueError('Unsafe sample ID')
        input_path = Path(source['reference_path'])
        source_hash = sha256(input_path)
        with Image.open(input_path) as image:
            if image.mode != 'RGB':
                raise ValueError('Explicit RGB preprocessing must precede inference')
            reference = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
        hr, lr, crop = degrade(reference)
        out = group / 'results' / sample_id
        out.mkdir(parents=True, exist_ok=True)
        rgb(hr).save(out / 'HR-reference.png')
        rgb(lr).save(out / 'LR-x4.png')
        np.save(out / 'LR-x4-fp32.npy', lr.numpy())
        # Verify the saved lossless LR really is the exact network input.
        saved_lr = torch.from_numpy(np.asarray(Image.open(out / 'LR-x4.png')).copy()).permute(2, 0, 1).float() / 255
        if not torch.equal(lr, saved_lr):
            raise ValueError('Saved LR differs from network input')
        predictions = {}
        for label in ['final', 'best']:
            identity = models[label]
            checkpoint = root / 'models' / Path(identity['weights_path']).name
            if sha256(checkpoint) != identity['weights_sha256']:
                raise ValueError('Model weights changed')
            weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
            model.load_state_dict(weights, strict=True)
            before = {k: v.detach().clone() for k, v in model.state_dict().items()}
            started = time.perf_counter()
            with torch.inference_mode():
                prediction = model(lr[None])[0].float().cpu()
                repeat = model(lr[None])[0].float().cpu()
            if prediction.shape != hr.shape or not torch.isfinite(prediction).all():
                raise ValueError('Invalid model reconstruction')
            if not torch.equal(prediction, repeat):
                raise ValueError('Inference is not deterministic')
            if any(not torch.equal(before[k], value) for k, value in model.state_dict().items()):
                raise ValueError('Inference mutated model')
            predictions[label] = prediction.numpy()
            np.save(out / (label + '-raw-fp32.npy'), prediction.numpy())
            rgb(prediction).save(out / (label + '.png'))
            np.save(out / (label + '-absolute-error.npy'), np.abs(prediction.clamp(0, 1).numpy() - hr.numpy()).mean(axis=0))
            Image.fromarray(error_map(prediction.numpy(), hr.numpy())).save(out / (label + '-error-0-to-0.25.png'))
            mask = torch.ones(hr.shape[-2:], dtype=torch.bool)
            scores = score_image(prediction.numpy(), hr.numpy(), mask.numpy())
            scores['lpips'] = metric.score(prediction, hr, mask)
            scores.update(sample_id=sample_id, model=label, step=identity['step'],
                          checkpoint_sha256=identity['checkpoint_sha256'], seconds=time.perf_counter()-started)
            all_rows.append(scores)
            print(json.dumps({'group': args.group, 'sample': sample_id, 'model': label,
                              'psnr': scores['psnr'], 'ssim': scores['ssim'], 'lpips': scores['lpips']}), flush=True)
        panel(out / 'comparison.png', sample_id, hr, lr, predictions['final'], predictions['best'])
        height, width = hr.shape[-2:]
        detail = detail_crop(sample_id, height, width)
        panel(out / 'detail.png', sample_id, hr, lr, predictions['final'], predictions['best'], detail)
        record = dict(source, source_sha256=source_hash, aligned_hr_shape=list(hr.shape),
                      actual_lr_shape=list(lr.shape), alignment_crop_xyxy=crop,
                      detail_crop_xyxy=detail,
                      outputs=output_inventory(out))
        sample_records.append(record)
        (out / 'provenance.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    with (group / 'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    summary = {}
    for label in ['final', 'best']:
        rows = [r for r in all_rows if r['model'] == label]
        summary[label] = dict(aggregate(rows), mean_per_image_lpips=float(np.mean([r['lpips'] for r in rows])))
    protocol = {'scope':'external synthetic x4 demonstration; not D3 sealed-test evaluation',
        'source_manifest_sha256':sha256(args.sources), 'source_reference':'source RGB image, not independent HR acquisition',
        'downsample':'torch.interpolate bicubic, align_corners=False, antialias=True; clip [0,1]; round to uint8; divide by255',
        'alignment':'top-left native crop to dimensions divisible by4; no reference resizing',
        'inference':'full-frame FP32 CPU; eval+inference_mode; exact pinned architecture; no tiling; seed42; deterministic algorithms; repeated output equality and unchanged-state assertions',
        'runtime':{'torch':torch.__version__,'numpy':np.__version__,'threads':torch.get_num_threads()},
        'metrics':'RGB PSNR over all pixels and population-covariance 11x11 Gaussian SSIM at valid window centers per original code; full support; raw predictions clipped [0,1]. LPIPS0.1.4 AlexNetv0.1 FP32 normalized [-1,1] over full image. Arithmetic means per image.',
        'error_display':'mean absolute RGB error at native reference pixels; shared fixed scale0..0.25; display clipped; numeric arrays retained',
        'model':model_info,'lpips':metric.provenance,'models':models,'summary':summary,'samples':sample_records}
    (group / 'evaluation.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    cards = []
    for source in sources:
        sample_id = source['id']
        cards.append('<h2>'+html.escape(sample_id)+'</h2><a href="results/'+sample_id+'/comparison.png"><img style="max-width:100%" src="results/'+sample_id+'/comparison.png"></a><p><a href="results/'+sample_id+'/detail.png">Detail crop</a> | <a href="results/'+sample_id+'/provenance.json">Provenance</a></p>')
    page = '<!doctype html><meta charset="utf-8"><title>D3 '+args.group+' comparisons</title><body style="max-width:1100px;margin:2em auto;font-family:Arial"><h1>D3 '+args.group+' synthetic x4 comparisons</h1><p>Source image = reference. This is a synthetic degradation experiment, not independent higher-resolution ground truth. Model1=final100000; model2=PSNR-best82000.</p><pre>'+html.escape(json.dumps(summary, indent=2))+'</pre>'+''.join(cards)+'</body>'
    (group / 'index.html').write_text(page, encoding='utf-8')
    print(json.dumps({'group':args.group,'status':'COMPLETE','summary':summary}), flush=True)


if __name__ == '__main__':
    main()

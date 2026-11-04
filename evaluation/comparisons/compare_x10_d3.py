"""D3x10 (x10) vs D3 (x4) on the 30 preserved demonstration scenes, in the D1/D2 comparison layout.

Sentinel-2 groups (real 10 m inputs, no ground truth): exact preserved D3 LR arrays are reused; D3 x4 predictions are
copied byte-for-byte; D3x10 best/final predict 1280x1280 (nominal 1 m). SkySat: the x10 synthetic set made by
prepare_skysat_x10.py (1 m reference, D3x10-degraded 10 m input); D3x10 and bicubic x10 are scored at 1 m; the D3 x4
model and bicubic x4 run on the SAME 10 m input and are scored against a 2.5 m reference of the same footprint.
Runtime = the original D3/D1 comparison runtime (torch 2.8.0+cpu, numpy 2.2.6, 6 threads, seed 42, deterministic,
eval/inference_mode, full frame, no tiling); every prediction is computed twice and must match exactly.
"""
import csv
import hashlib
import html
import importlib.util
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(os.environ.get('RUN_ARCHIVE', Path(__file__).resolve().parents[1]))  # archive folder of one run, e.g. 'D1 run'
D3 = Path(os.environ.get('D3_ARCHIVE', ROOT.parent / 'D3 run'))
SOURCE = ROOT / 'training-run/project/source'
os.environ['TORCH_HOME'] = str(ROOT / 'training-run/environment/torch-cache')  # pinned AlexNet for LPIPS
sys.path.insert(0, str(SOURCE))
from research.model_factory import make_model as make_x10, MODEL as X10_MODEL  # noqa: E402
from research.metrics import score_image, aggregate  # noqa: E402
from research.lpips_metric import LPIPSMetric  # noqa: E402

S2_GROUPS = ['sentinel2-comparisons', 'sentinel2-additional-ports']
SKY = 'skysat-comparisons'
ARCH_SHA = '9e143898679ebeebc5d2fc94ad1b89c38aa4a4d43da4e0fcba0f93e476994913'
FONT = 'C:/Windows/Fonts/arial.ttf'


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(8 << 20), b''):
            h.update(c)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def checked_copy(src, dst, expected=None):
    digest = sha256(src)
    if expected and digest != expected:
        raise ValueError(f'Source hash mismatch: {src}')
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if sha256(dst) != digest:
        raise ValueError(f'Copy mismatch: {dst}')
    return {'sha256': digest, 'bytes': dst.stat().st_size, 'original_path': str(src)}


def rgb(a):
    return Image.fromarray(np.rint(np.clip(a.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8))


def panel(path, title, items, side, footer, crop=None):
    """items: (array CHW or HW error map, label, grid_note, crop_box_in_that_array_or_None). Nearest display enlargement."""
    rows = (len(items) + 2) // 3
    canvas = Image.new('RGB', (side * 3, rows * (side + 54) + 74), 'white')
    draw = ImageDraw.Draw(canvas)
    big, small = ImageFont.truetype(FONT, 18), ImageFont.truetype(FONT, 13)
    for i, (a, label, note, box) in enumerate(items):
        pic = rgb(a) if a.ndim == 3 else Image.fromarray(np.rint(np.clip(a / .25, 0, 1) * 255).astype(np.uint8)).convert('RGB')
        if box:
            pic = pic.crop(box)
        x, y = i % 3 * side, i // 3 * (side + 54)
        draw.text((x + 5, y + 6), label, fill='black', font=big)
        draw.text((x + 5, y + 31), f'{note}; display only, nearest-neighbour {side / pic.width:g}x', fill='#555555', font=small)
        canvas.paste(pic.resize((side, side), Image.Resampling.NEAREST), (x, y + 54))
    draw.text((5, rows * (side + 54) + 8), title, fill='black', font=big)
    draw.text((5, rows * (side + 54) + 34), footer, fill='black', font=big)
    canvas.save(path)


def load_x4_model():
    """D3 x4 SwinIR from the same pinned source and constructor (strict load of D3 weights proves identity)."""
    spec = importlib.util.spec_from_file_location('pinned_swinir_x4', SOURCE / 'vendor/network_swinir.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kwargs = {k: v for k, v in X10_MODEL.items()}
    kwargs['upscale'] = 4
    return module.SwinIR(norm_layer=torch.nn.LayerNorm, **kwargs)


def predict(model, weights, lr):
    model.load_state_dict(weights, strict=True)
    started = time.perf_counter()
    with torch.inference_mode():
        a = model(lr[None])[0].float().cpu()
        b = model(lr[None])[0].float().cpu()
    if not torch.isfinite(a).all() or not torch.equal(a, b):
        raise ValueError('Invalid or non-repeatable prediction')
    if any(not torch.equal(weights[k], v) for k, v in model.state_dict().items()):
        raise ValueError('Inference mutated model')
    return a.numpy(), time.perf_counter() - started


def main():
    if torch.__version__ != '2.8.0+cpu' or np.__version__ != '2.2.6':
        raise ValueError('Original D3/D1 comparison runtime required (torch 2.8.0+cpu, numpy 2.2.6)')
    torch.set_num_threads(6); torch.manual_seed(42); np.random.seed(42); torch.use_deterministic_algorithms(True)
    if sha256(SOURCE / 'vendor/network_swinir.py') != ARCH_SHA:
        raise ValueError('Architecture source changed')
    ids = json.loads((ROOT / 'models/model-identities.json').read_text())
    d3ids = json.loads((D3 / 'models/model-identities.json').read_text())
    x10w, d3w = {}, {}
    for label in ('best', 'final'):
        wp = ROOT / ids[label]['weights_file']
        if sha256(wp) != ids[label]['weights_sha256']:
            raise ValueError('D3x10 weights changed')
        x10w[label] = torch.load(wp, map_location='cpu', weights_only=True)
        dp = D3 / 'models' / Path(d3ids[label]['weights_path']).name
        if sha256(dp) != d3ids[label]['weights_sha256']:
            raise ValueError('D3 weights changed')
        d3w[label] = torch.load(dp, map_location='cpu', weights_only=True)
    x10, info = make_x10(SOURCE / 'vendor/network_swinir.py'); x10.eval().requires_grad_(False)
    x4 = load_x4_model(); x4.eval().requires_grad_(False)
    steps = {'best': ids['best']['step'], 'final': ids['final']['step']}
    d3steps = {k: d3ids[k]['step'] for k in ('best', 'final')}
    lpips = LPIPSMetric('cpu')
    records, cards, metric_rows = [], [], []

    # ---------------- Sentinel-2: real 10 m inputs, no ground truth
    for group in S2_GROUPS:
        src_group, dst_group = D3 / group, ROOT / group
        if (src_group / 'inputs').exists() and not (dst_group / 'inputs').exists():
            shutil.copytree(src_group / 'inputs', dst_group / 'inputs', ignore=shutil.ignore_patterns('__pycache__'))
        samples = sorted(p for p in (src_group / 'results').iterdir() if p.is_dir())
        if len(samples) != 10:
            raise ValueError('Exactly ten preserved scenes per group required')
        group_records = []
        for src in samples:
            sample, dest = src.name, dst_group / 'results' / src.name
            dest.mkdir(parents=True, exist_ok=True)
            prov = json.loads((src / 'provenance.json').read_text(encoding='utf-8'))
            inv, copies = prov['outputs'], {}
            for fn in ('LR-input-fp32.npy', 'LR-input.png'):
                copies[fn] = checked_copy(src / fn, dest / fn, inv[fn]['sha256'])
            lr = np.load(dest / 'LR-input-fp32.npy', allow_pickle=False)
            if lr.dtype != np.float32 or lr.shape != (3, 128, 128) or not np.array_equal(lr, np.asarray(Image.open(dest / 'LR-input.png')).transpose(2, 0, 1).astype(np.float32) / 255):
                raise ValueError('Saved LR is not the exact D3 input')
            lr_t = torch.from_numpy(lr)
            preds, timings = {}, {}
            for label in ('final', 'best'):
                for fn in (label + '-raw-fp32.npy', label + '.png'):
                    copies['D3-' + fn] = checked_copy(src / fn, dest / ('D3-' + fn), inv[fn]['sha256'])
                preds['D3-' + label] = np.load(dest / f'D3-{label}-raw-fp32.npy', allow_pickle=False)
                p, timings[label] = predict(x10, x10w[label], lr_t)
                if p.shape != (3, 1280, 1280):
                    raise ValueError('x10 output shape')
                preds['D3x10-' + label] = p
                np.save(dest / f'{label}-raw-fp32.npy', p); rgb(p).save(dest / f'{label}.png')
                print(json.dumps({'group': group, 'sample': sample, 'model': 'D3x10-' + label, 'seconds_two_forwards': round(timings[label], 2)}), flush=True)
            note_in = 'Sentinel-2 input, 10 m grid'
            footer = 'No independent HR ground truth: nominal 1 m (x10) and 2.5 m (x4) grids do not prove that resolving power.'
            # detail = the same 320 m square as D1/D2/D3 (input 48:80) in every grid
            for mode in ('pair', 'best', 'final'):
                if mode == 'pair':
                    items = [(lr, note_in, 'input', None), (preds['D3x10-final'], f'D3x10 final ({steps["final"]:,})', 'x10 output, 1 m grid', None),
                             (preds['D3x10-best'], f'D3x10 best ({steps["best"]:,})', 'x10 output, 1 m grid', None)]
                    base = 'comparison'
                else:
                    items = [(lr, note_in, 'input', None), (preds['D3x10-' + mode], f'D3x10 {mode} ({steps[mode]:,}), x10', '1 m grid', None),
                             (preds['D3-' + mode], f'D3 {mode} ({d3steps[mode]:,}), x4', '2.5 m grid', None)]
                    base = f'D3x10-vs-D3-{mode}'
                panel(dest / f'{base}.png', f'{group} / {sample}', items, 1280, footer)
                boxes = {'input': (48, 48, 80, 80), '1 m': (480, 480, 800, 800), '2.5 m': (192, 192, 320, 320)}
                ditems = [(a, l, n, boxes['input'] if a.shape[-1] == 128 else boxes['1 m'] if a.shape[-1] == 1280 else boxes['2.5 m']) for a, l, n, _ in items]
                panel(dest / ('detail.png' if mode == 'pair' else f'{base}-detail.png'), f'{group} / {sample} | same 320 m detail square as D1-D3', ditems, 640, footer)
            rec = {'group': group, 'id': sample, 'original_D3_provenance': prov, 'original_provenance_sha256': sha256(src / 'provenance.json'),
                   'copied_inputs_and_D3_outputs': copies, 'exact_D3_input_reused': True, 'repeat_equal': True, 'model_state_unchanged': True,
                   'input_shape': [3, 128, 128], 'output_shape_x10': [3, 1280, 1280], 'output_shape_D3_x4': [3, 512, 512],
                   'seconds_two_forwards': timings, 'detail_square_m': 320,
                   'outputs': {p.name: {'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in dest.iterdir() if p.is_file() and p.name != 'provenance.json'}}
            write(dest / 'provenance.json', rec); group_records.append(rec); records.append(rec)
            cards.append(f'<h2>{html.escape(group + " / " + sample)}</h2><a href="../{group}/results/{sample}/D3x10-vs-D3-best.png"><img style="max-width:100%" src="../{group}/results/{sample}/D3x10-vs-D3-best-detail.png"></a>'
                         f'<p><a href="../{group}/results/{sample}/D3x10-vs-D3-best.png">D3x10 best vs D3 best (full)</a> | <a href="../{group}/results/{sample}/D3x10-vs-D3-final.png">finals</a> | '
                         f'<a href="../{group}/results/{sample}/comparison.png">D3x10 final/best</a> | <a href="../{group}/results/{sample}/provenance.json">provenance</a></p>')
        write(dst_group / 'inference.json', {'samples': group_records, 'count': 10, 'ground_truth': 'Unavailable; no HR quality metrics'})
        (dst_group / 'README.txt').write_text(
            f'D3x10 (x10) inference on the exact preserved D3 Sentinel-2 inputs (128x128 at 10 m -> 1280x1280, nominal 1 m). D3x10 best={steps["best"]}, '
            f'final={steps["final"]}; D3 x4 predictions (best={d3steps["best"]}, final={d3steps["final"]}; 512x512, nominal 2.5 m) copied byte-for-byte, not rerun. '
            'CPU FP32 full-frame, repeated inference equal, weights unchanged. No independent HR ground truth; nominal output grids do not establish resolving power. '
            'Detail panels show the same 320 m square as the D1/D2/D3 detail crops.\n', encoding='utf-8')
        (dst_group / 'index.html').write_text(f'<!doctype html><meta charset="utf-8"><title>{group} x10</title><body style="font-family:Arial;max-width:1500px;margin:2em auto">'
                                              f'<h1>{group}: D3x10 (x10) vs D3 (x4)</h1>' + ''.join(c.replace(f'../{group}/', '') for c in cards if f'>{group} /' in c) + '</body>', encoding='utf-8')

    # ---------------- SkySat x10 synthetic
    sky_in, sky_out = ROOT / SKY / 'inputs', ROOT / SKY / 'results'
    sources = json.loads((sky_in / 'sources.json').read_text())
    group_records = []
    for s in sources['samples']:
        sid, folder = s['id'], sky_in / s['id']
        for fn, meta in s['files'].items():
            if sha256(folder / fn) != meta['sha256']:
                raise ValueError(f'Changed SkySat input {sid}/{fn}')
        dest = sky_out / sid
        dest.mkdir(parents=True, exist_ok=True)
        for fn in ('LR-10m-fp32.npy', 'LR-10m.png', 'HR-reference-1m.png', 'HR-reference-2.5m.png'):
            checked_copy(folder / fn, dest / fn, s['files'][fn]['sha256'])
        lr = np.load(dest / 'LR-10m-fp32.npy', allow_pickle=False)
        hr1 = np.asarray(Image.open(dest / 'HR-reference-1m.png')).transpose(2, 0, 1).astype(np.float32) / 255
        hr25 = np.asarray(Image.open(dest / 'HR-reference-2.5m.png')).transpose(2, 0, 1).astype(np.float32) / 255
        lr_t = torch.from_numpy(lr)
        preds, timings = {}, {}
        for label in ('best', 'final'):
            preds['D3x10-' + label], timings['D3x10-' + label] = predict(x10, x10w[label], lr_t)
            preds['D3-x4-' + label], timings['D3-x4-' + label] = predict(x4, d3w[label], lr_t)
        preds['bicubic-x10'] = torch.nn.functional.interpolate(lr_t[None], size=hr1.shape[-2:], mode='bicubic', align_corners=False, antialias=False)[0].numpy()
        preds['bicubic-x4'] = torch.nn.functional.interpolate(lr_t[None], size=hr25.shape[-2:], mode='bicubic', align_corners=False, antialias=False)[0].numpy()
        for name, p in preds.items():
            ref = hr1 if p.shape[-1] == 240 else hr25
            if p.shape != ref.shape:
                raise ValueError(f'{name} shape {p.shape} vs {ref.shape}')
            np.save(dest / f'{name}-raw-fp32.npy', p.astype(np.float32)); rgb(p).save(dest / f'{name}.png')
            err = np.abs(np.clip(p, 0, 1) - ref).mean(0)
            np.save(dest / f'{name}-absolute-error.npy', err.astype(np.float32))
            Image.fromarray(np.rint(np.clip(err / .25, 0, 1) * 255).astype(np.uint8)).save(dest / f'{name}-error-0-to-0.25.png')
            mask = torch.ones(ref.shape[-2:], dtype=torch.bool)
            row = score_image(p, ref, mask.numpy())
            row['lpips'] = lpips.score(torch.from_numpy(np.ascontiguousarray(p)), torch.from_numpy(ref), mask)
            row.update(sample_id=sid, model=name, reference_grid_m=1.0 if ref is hr1 else 2.5,
                       step=steps.get(name.split('-')[-1]) if name.startswith('D3x10') else d3steps.get(name.split('-')[-1]) if name.startswith('D3-') else None)
            metric_rows.append(row)
        footer = 'Synthetic x10 demonstration (D3x10 degradation of a known-source SkySat scene); error scale 0..0.25. x10 scored at 1 m, x4 at 2.5 m.'
        for mode in ('best', 'final'):
            items = [(hr1, 'HR reference, 1 m', 'reference', None), (preds['D3x10-' + mode], f'D3x10 {mode} ({steps[mode]:,}), x10', '1 m grid', None),
                     (preds['D3-x4-' + mode], f'D3 {mode} ({d3steps[mode]:,}), x4', '2.5 m grid, same input', None),
                     (preds['bicubic-x10'], 'LR input, bicubic x10', '10 m input', None),
                     (np.abs(np.clip(preds['D3x10-' + mode], 0, 1) - hr1).mean(0), 'D3x10 error vs 1 m reference', '1 m grid', None),
                     (np.abs(np.clip(preds['D3-x4-' + mode], 0, 1) - hr25).mean(0), 'D3 x4 error vs 2.5 m reference', '2.5 m grid', None)]
            panel(dest / f'D3x10-vs-D3-{mode}.png', f'skysat / {sid} ({s["content"]})', items, 480, footer)
            box = lambda a: (60, 60, 180, 180) if a.shape[-1] == 240 else (24, 24, 72, 72)
            panel(dest / f'D3x10-vs-D3-{mode}-detail.png', f'skysat / {sid} | central 120 m', [(a, l, n, box(a)) for a, l, n, _ in items], 480, footer)
        panel(dest / 'comparison.png', f'skysat / {sid}', [(hr1, 'HR reference, 1 m', 'reference', None),
              (preds['D3x10-final'], f'D3x10 final ({steps["final"]:,})', '1 m grid', None), (preds['D3x10-best'], f'D3x10 best ({steps["best"]:,})', '1 m grid', None)], 480, footer)
        rec = {'group': SKY, 'id': sid, 'source_record': s, 'repeat_equal': True, 'model_state_unchanged': True, 'seconds_two_forwards': timings,
               'outputs': {p.name: {'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in dest.iterdir() if p.is_file() and p.name != 'provenance.json'}}
        write(dest / 'provenance.json', rec); group_records.append(rec); records.append(rec)
        cards.append(f'<h2>{html.escape(SKY + " / " + sid + " - " + s["content"])}</h2><a href="../{SKY}/results/{sid}/D3x10-vs-D3-best.png"><img style="max-width:100%" src="../{SKY}/results/{sid}/D3x10-vs-D3-best.png"></a>'
                     f'<p><a href="../{SKY}/results/{sid}/D3x10-vs-D3-best-detail.png">central detail</a> | <a href="../{SKY}/results/{sid}/D3x10-vs-D3-final.png">finals</a> | <a href="../{SKY}/results/{sid}/provenance.json">provenance</a></p>')
        print(json.dumps({'group': SKY, 'sample': sid, 'done': True}), flush=True)
    fields = list(metric_rows[0])
    with (ROOT / SKY / 'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(metric_rows)
    labels = sorted({r['model'] for r in metric_rows})
    summary = {l: dict(aggregate([r for r in metric_rows if r['model'] == l]), mean_per_image_lpips=float(np.mean([r['lpips'] for r in metric_rows if r['model'] == l])),
                       reference_grid_m=next(r['reference_grid_m'] for r in metric_rows if r['model'] == l)) for l in labels}
    for scale, base in (('x10', 'bicubic-x10'), ('x4', 'bicubic-x4')):
        for l in labels:
            if summary[l]['reference_grid_m'] == summary[base]['reference_grid_m'] and l != base:
                summary[l]['psnr_gain_over_bicubic_same_grid_db'] = summary[l]['mean_per_image_rgb_psnr'] - summary[base]['mean_per_image_rgb_psnr']
    write(ROOT / SKY / 'inference.json', {'samples': group_records, 'count': 10, 'ground_truth': 'Synthetic references (1 m and 2.5 m) from the native 0.8 m scene'})
    write(ROOT / SKY / 'evaluation.json', {'scope': 'external synthetic demonstration; not sealed test; known-source acquisition',
                                           'metrics': 'Identical score_image and LPIPSMetric of the training code; full RGB support; arithmetic per-image means',
                                           'grids': 'D3x10 and bicubic-x10 vs 1 m reference (240x240); D3 x4 and bicubic-x4 vs 2.5 m reference (96x96), same 10 m input',
                                           'summary': summary, 'lpips': lpips.provenance})
    (ROOT / SKY / 'README.txt').write_text(
        'x10 SkySat synthetic demonstration (rebuilt for x10). Same New Bedford scene and the same 10 spots as the D1-D3 x4 SkySat set, but the x4 set\'s '
        '0.8 m/3.2 m pairs do not fit x10, so each spot was cut from the native 0.8 m scene as a 320 m context, area-integrated to 1 m and degraded with the '
        'exact D3x10 recipe to a 10 m input (see inputs/prepare_skysat_x10.py and inputs/sources.json). D3x10 best/final and bicubic x10 are scored against '
        'the 1 m reference; the D3 x4 models and bicubic x4 run on the same 10 m input and are scored against a 2.5 m reference of the same footprint. '
        'Scores at different grids are not directly comparable; gains over the same-grid bicubic are. Known-source acquisition; not independent validation.\n',
        encoding='utf-8')
    (ROOT / SKY / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>SkySat x10</title><body style="font-family:Arial;max-width:1500px;margin:2em auto">'
                                           '<h1>SkySat x10 synthetic demonstration</h1><pre>' + html.escape(json.dumps(summary, indent=2)) + '</pre>'
                                           + ''.join(c.replace(f'../{SKY}/', '') for c in cards if f'>{SKY} /' in c) + '</body>', encoding='utf-8')
    for label in ('best', 'final'):
        if sha256(ROOT / ids[label]['weights_file']) != ids[label]['weights_sha256']:
            raise ValueError('D3x10 weights changed during comparison')
    protocol = {'status': 'PASS', 'created_utc': datetime.now(timezone.utc).isoformat(), 'count': len(records), 'adapted_script_sha256': sha256(Path(__file__)),
                'reference_script': 'D1 run/tools/compare_d1_d3.py (same runtime, verification and panel conventions)',
                'inference': 'CPU FP32, full frame, no tiling, eval/inference_mode, seed 42, 6 threads, deterministic; two identical forwards per model and image; weights unchanged',
                'runtime': {'torch': torch.__version__, 'numpy': np.__version__, 'threads': torch.get_num_threads()},
                'D3x10_models': ids, 'D3_models': d3ids, 'architecture_x10': info, 'lpips': lpips.provenance, 'SkySat_summary': summary, 'samples': records}
    if len(records) != 30:
        raise ValueError('Not all 30 scenes complete')
    write(ROOT / 'reports/comparison-verification.json', protocol)
    (ROOT / 'reports/comparison-index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>D3x10 vs D3</title><body style="font-family:Arial;max-width:1550px;margin:2em auto">'
        f'<h1>D3x10 (x10, 1 m) vs D3 (x4, 2.5 m): the 30 preserved comparison scenes</h1><p>D3x10 best {steps["best"]:,} / final {steps["final"]:,}; '
        f'D3 best {d3steps["best"]:,} / final {d3steps["final"]:,}. Sentinel-2 scenes have no independent HR ground truth. SkySat metrics use the rebuilt x10 '
        'synthetic references (x10 scored at 1 m, x4 at 2.5 m).</p><pre>' + html.escape(json.dumps(summary, indent=2)) + '</pre>' + ''.join(cards) + '</body>',
        encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'count': len(records), 'summary': summary}, indent=1), flush=True)


if __name__ == '__main__':
    main()

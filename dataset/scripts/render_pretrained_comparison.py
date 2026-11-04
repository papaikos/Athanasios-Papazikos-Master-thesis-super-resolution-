"""Verify genuine completed model outputs, then assemble the 15-scene delivery.

No inference or image synthesis occurs here. Original imagery and float outputs
are preserved, and model receipts bind each result to its exact source hash.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def font(size, bold=False):
    return ImageFont.truetype('C:/Windows/Fonts/' + ('arialbd.ttf' if bold else 'arial.ttf'), size)


def load(path):
    return json.loads(Path(path).read_text('utf-8'))


def safe_file(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f'Invalid or missing result path: {relative}')
    return path


def receipt(root, manifest, is_diffusion=False):
    data = load(root / ('results.json' if is_diffusion else 'result.json'))
    if data.get('status') != 'complete' or len(data.get('examples', [])) != 15:
        raise ValueError(f'Not 15 completed model results: {root}')
    by_id = {r['id']: r for r in data['examples']}
    if len(by_id) != 15 or set(by_id) != {r['id'] for r in manifest['examples']}:
        raise ValueError('Model example IDs do not exactly match the source manifest')
    for source in manifest['examples']:
        row = by_id[source['id']]
        source_hash = row['input_sha256'] if is_diffusion else row['source']['sha256']
        if source_hash != source['sha256']:
            raise ValueError(f"Model source hash mismatch: {source['id']}")
        files = ({'arrays.npz': {'path': row['npz'], 'sha256': row['npz_sha256']},
                  'output.png': {'path': row['png'], 'sha256': row['png_sha256']}}
                 if is_diffusion else row['files'])
        for item in files.values():
            if sha(safe_file(root, item['path'])) != item['sha256']:
                raise ValueError(f"Model output checksum mismatch: {item['path']}")
        row['verified_files'] = files
    return data, by_id


def rgb_image(rgb, white=1):
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[0] != 3 or not np.isfinite(rgb).all():
        raise ValueError('Invalid RGB tensor')
    return Image.fromarray(np.round(np.clip(rgb / white, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8))


def tensor_rgb(arrays, key):
    bands = arrays['bands'].tolist()
    return arrays[key][[bands.index(b) for b in ['B4', 'B3', 'B2']]]


def render(args):
    manifest = load(args.inputs / 'manifest.json')
    if manifest.get('status') != 'complete' or len(manifest.get('examples', [])) != 15:
        raise ValueError('Expected 15 completed source examples')
    weo_receipt, weo = receipt(args.weo, manifest)
    diffusion_receipt, diffusion = receipt(args.diffusionsat, manifest, is_diffusion=True)
    # Validate all tensors before creating the final desktop package.
    prepared = []
    for source in manifest['examples']:
        key = source['id']
        input_path = safe_file(args.inputs, source['path'])
        if sha(input_path) != source['sha256']:
            raise ValueError(f'Input checksum mismatch: {key}')
        wp = safe_file(args.weo, weo[key]['files']['arrays.npz']['path'])
        dp = safe_file(args.diffusionsat, diffusion[key]['verified_files']['arrays.npz']['path'])
        with np.load(input_path, allow_pickle=False) as inp, np.load(wp, allow_pickle=False) as w, np.load(dp, allow_pickle=False) as d:
            sr = w['sen2sr']
            if sr.shape != (10, 512, 512) or not np.isfinite(sr).all():
                raise ValueError(f'Invalid WEO output: {key}')
            if not np.array_equal(tensor_rgb(w, 'original'), inp['reflectance'][:3]):
                raise ValueError(f'WEO/source pixel correspondence mismatch: {key}')
            generated = d['rgb']
            if generated.shape != (512, 512, 3) or not np.isfinite(generated).all():
                raise ValueError(f'Invalid DiffusionSat output: {key}')
            if generated.min() < 0 or generated.max() > 1:
                raise ValueError(f'Unexpected DiffusionSat RGB range: {key}')
            prepared.append((source, rgb_image(inp['reflectance'][:3], .3),
                             rgb_image(tensor_rgb(w, 'sen2sr'), .3), rgb_image(generated.transpose(2, 0, 1))))
    args.output.mkdir(parents=True, exist_ok=True)
    audit = {'created_utc': datetime.now(timezone.utc).isoformat(), 'count': 15,
             'source_manifest_sha256': sha(args.inputs / 'manifest.json'),
             'weo_result_sha256': sha(args.weo / 'result.json'),
             'diffusionsat_result_sha256': sha(args.diffusionsat / 'results.json'),
             'examples': [], 'files': {}}
    for source, original, wimage, dimage in prepared:
        key = source['id']
        dest = args.output / 'examples' / key
        dest.mkdir(parents=True, exist_ok=True)
        original.save(dest / 'original-reflectance.png')
        wimage.save(dest / 'sen2sr-released.png')
        dimage.save(dest / 'diffusionsat-generative.png')
        shutil.copy2(args.inputs / f'{key}-original.png', dest / 'original-tci.png')
        shutil.copy2(args.inputs / source['path'], dest / 'input-multispectral.npz')
        for name, root, records in [('sen2sr', args.weo, weo), ('diffusionsat', args.diffusionsat, diffusion)]:
            shutil.copy2(safe_file(root, records[key]['verified_files']['arrays.npz']['path']), dest / f'{name}-arrays.npz')
        audit['examples'].append({'id': key, 'scene_id': source['provenance']['scene_id'],
                                  'source_sha256': source['sha256'],
                                  'transform': source['provenance']['transform']})
    evidence = args.output / 'provenance'
    evidence.mkdir(exist_ok=True)
    shutil.copy2(args.inputs / 'manifest.json', evidence / 'input-manifest.json')
    shutil.copy2(args.weo / 'result.json', evidence / 'sen2sr-result.json')
    shutil.copy2(args.diffusionsat / 'results.json', evidence / 'diffusionsat-result.json')
    panel, margin, gap = 512, 28, 18
    width, height = 1628, 1988
    for page in range(5):
        canvas = Image.new('RGB', (width, height), '#101722')
        draw = ImageDraw.Draw(canvas)
        draw.text((margin, 22), f'Sentinel-2 boats and marinas | Released models | {page + 1}/5', font=font(32, True), fill='white')
        draw.text((margin, 70), 'Same 15 source scenes and footprints. Released pretrained models; no new fine-tuning.', font=font(24), fill='#c3d1df')
        labels = ['Original | native 10 m', 'SEN2SR | inferred 2.5 m grid', 'DiffusionSat | generative 2.5 m grid']
        for col, label in enumerate(labels):
            draw.text((margin + col * (panel + gap), 118), label, font=font(23, True), fill='#8edbd2' if col else 'white')
        for local, (source, original, wimage, dimage) in enumerate(prepared[page * 3:(page + 1) * 3]):
            y = 173 + local * 577
            date = source['provenance']['date']
            date = f'{date[:4]}-{date[4:6]}-{date[6:8]}'
            draw.text((margin, y), f"{page * 3 + local + 1:02d}  {source['title']}  |  {date}", font=font(25, True), fill='white')
            for col, image in enumerate([original.resize((512, 512), Image.Resampling.NEAREST), wimage, dimage]):
                assert image.size == (512, 512)
                canvas.paste(image, (margin + col * (panel + gap), y + 38))
        draw.text((margin, 1918), 'Original and SEN2SR: fixed reflectance / 0.3. DiffusionSat produces generative RGB appearance.', font=font(22), fill='#c3d1df')
        draw.text((margin, 1951), 'DiffusionSat visibly alters structures. No matching HR ground truth. Original enlarged with nearest-neighbour.', font=font(21), fill='#c3d1df')
        canvas.save(args.output / f'comparison-{page + 1:02d}.png')
    (args.output / 'README.txt').write_text(
        'RELEASED PRETRAINED MODEL COMPARISON\n\n'
        '15 original Sentinel-2 scenes, identical locations/dates/1280 m footprints. '
        'These are released pretrained SEN2SR and DiffusionSat, not our earlier fine-tuned models.\n\n'
        'Five comparison PNG sheets contain three examples each: original native 10 m imagery, '
        'SEN2SR 4x output, and DiffusionSat generative output. Original pixels are enlarged using '
        'nearest-neighbour only. Generated outputs are shown at 512 x 512 without resampling. '
        'A 2.5 m output sampling grid is not measured 2.5 m resolution. No matching HR reference exists.\n\n'
        'Original-reflectance and SEN2SR PNGs use RGB B4/B3/B2 divided by 0.3 then clipped to [0, 1], '
        'with no image-specific stretch or gamma. DiffusionSat RGB is generative appearance and '
        'is not a calibrated reflectance measurement; structures/detail may be invented. '
        'The actual DiffusionSat results visibly distort structures and should be treated as poor '
        'generative reconstructions, not faithful recovered detail.\n\n'
        'Each example includes untouched original-tci.png for the earlier official TCI display. '
        'TCI uses a different rendering than reflectance/0.3 and must not be interpreted as '
        'the same display mapping. Input multispectral float data, masks and both model float '
        'outputs are preserved in NPZ archives. Native 20 m input bands were bilinearly resampled '
        'onto the 10 m grid; this does not create new measured detail.\n\n'
        'Provenance receipts record model identities/settings and bind results to source SHA256. '
        'verification.json audits every delivered file (excluding itself).\n', encoding='utf-8')
    for path in sorted(args.output.rglob('*')):
        if path.is_file() and path.name != 'verification.json':
            audit['files'][path.relative_to(args.output).as_posix()] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    audit.update(status='verified', comparison_sheets=5, model_outputs=30, source_examples=15)
    (args.output / 'verification.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'verified', 'output': str(args.output), 'examples': 15, 'sheets': 5}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--weo', type=Path, required=True)
    p.add_argument('--diffusionsat', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--report', type=Path)
    render(p.parse_args())


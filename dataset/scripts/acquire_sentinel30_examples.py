"""Extend the preserved Sentinel-2 inference set to 30 native 10 m port scenes."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import acquire_sentinel_examples as base
from PIL import Image, ImageDraw

OUT = Path('outputs/sentinel30-comparison-20260928/inputs')
EXTRA = [
 ('cannes', 'Cannes old harbour, France', 7.0148, 43.5483),
 ('antibes', 'Port Vauban, Antibes', 7.1283, 43.5854),
 ('nice', 'Nice harbour, France', 7.2849, 43.6950),
 ('toulon', 'Toulon harbour, France', 5.9294, 43.1177),
 ('ajaccio', 'Ajaccio harbour, Corsica', 8.7419, 41.9264),
 ('cagliari', 'Cagliari port, Sardinia', 9.1095, 39.2079),
 ('palermo', 'Palermo harbour, Sicily', 13.3698, 38.1293),
 ('catania', 'Catania harbour, Sicily', 15.0932, 37.4963),
 ('corfu', 'Corfu ferry port, Greece', 19.9027, 39.6303),
 ('rhodes', 'Mandraki harbour, Rhodes', 28.2275, 36.4512),
 ('kos', 'Kos harbour, Greece', 27.2901, 36.8987),
 ('chania', 'Chania old harbour, Crete', 24.0190, 35.5186),
 ('heraklion', 'Heraklion harbour, Crete', 25.1451, 35.3445),
 ('bodrum', 'Bodrum marina, Turkey', 27.4219, 37.0326),
 ('marmaris', 'Marmaris marina, Turkey', 28.2770, 36.8508),
]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    previous = Path('outputs/sentinel-examples')
    old = json.loads((previous / 'manifest.json').read_text(encoding='utf-8'))
    for row in old['examples']:
        src = previous / row['path']
        assert hashlib.sha256(src.read_bytes()).hexdigest() == row['sha256']
        for name in [row['path'], row['id'] + '-original.png']:
            target = OUT / name
            if target.exists():
                assert target.read_bytes() == (previous / name).read_bytes()
            else:
                shutil.copy2(previous / name, target)
        metadata = OUT / (row['id'] + '.json')
        if not metadata.exists():
            metadata.write_text(json.dumps(row, indent=2), encoding='utf-8')
    base.OUT = OUT
    base.SITES = base.SITES + EXTRA
    base.main()
    manifest = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
    rows = manifest['examples']
    assert len(rows) == 30
    assert len({r['sha256'] for r in rows}) == 30
    assert len({(r['provenance']['scene_id'], tuple(r['provenance']['transform'])) for r in rows}) == 30
    canvas = Image.new('RGB', (5 * 276, 6 * 300), (15, 23, 35))
    draw = ImageDraw.Draw(canvas)
    for i, row in enumerate(rows):
        x, y = (i % 5) * 276, (i // 5) * 300
        image = Image.open(OUT / (row['id'] + '-original.png')).resize((256, 256), Image.Resampling.NEAREST)
        canvas.paste(image, (x + 10, y + 30))
        draw.text((x + 10, y + 7), str(i + 1) + '. ' + row['title'], fill='white')
    canvas.save(OUT / 'originals-contact-sheet.png')
    print('Verified 30 distinct, real native 10 m Sentinel-2 inputs.', flush=True)

if __name__ == '__main__':
    main()

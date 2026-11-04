"""Contact sheets for one decisions folder: one sheet per reason, highest deciding value first.

Usage: python make_contact_sheets.py decisions/<folder>
Sheets go to <folder>/sheets/. Images are read (read only) from the dataset the run used.
#N on a sheet is the row number in that folder's decisions.csv.
"""
import collections
import csv
import io
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from typesafe_image_filter import CANDIDATES, DEFECTS, to_uint8

folder = Path(sys.argv[1])
policy = json.loads((folder / 'policy.json').read_text(encoding='utf-8'))
dataset_dir = Path(json.loads((Path(policy['run_dir']) / 'run.json').read_text(encoding='utf-8'))['dataset_dir'])
rows = list(csv.DictReader((folder / 'decisions.csv').open(encoding='utf-8')))
for number, row in enumerate(rows, 1):
    row['row'] = number
    row['max_defect'] = max(float(row[key]) for key in DEFECTS)

kept = [row for row in rows if row['decision'] == 'keep']
groups = {}
for reason, key in [*((f'defect_{name}', name) for name in DEFECTS), ('mostly_clouds', 'clouds_over_half')]:
    groups[f'remove_{reason}'] = (key, [row for row in rows if reason in row['remove_reasons'].split(';')])
for candidate in CANDIDATES:
    groups[f'candidate_{candidate}'] = ('maritime', [row for row in kept if candidate in row['candidates'].split(';')])
groups['kept_near_cloud_threshold'] = ('clouds_over_half', [row for row in kept if float(row['clouds_over_half']) >= 0.25])
groups['kept_near_defect_threshold'] = ('max_defect', [row for row in kept if row['max_defect'] > 0.5])

needed = collections.defaultdict(set)
for _, items in groups.values():
    for row in items:
        needed[(row['split'], row['tar'])].add(row['image'])
thumbs = {}
for (split, tar), names in needed.items():
    with tarfile.open(dataset_dir / split / tar, 'r') as archive:
        for member in archive:
            if member.name in names:
                with np.load(io.BytesIO(archive.extractfile(member).read()), allow_pickle=False) as data:
                    hr = to_uint8(data['hr']).transpose(1, 2, 0)
                thumbs[member.name] = Image.fromarray(hr).resize((200, 200), Image.LANCZOS)

SOURCE = {'skysat_raw_window': 'skysat', 'aerial_rgb': 'aerial', 'earthview': 'earthview', 'maxar': 'maxar'}
font, head = ImageFont.load_default(size=12), ImageFont.load_default(size=20)
COLS, PER, W, H = 8, 64, 204, 236
out = folder / 'sheets'
out.mkdir(exist_ok=True)
for name, (key, items) in groups.items():
    items.sort(key=lambda row: float(row[key]), reverse=True)
    for page in range(0, len(items), PER):
        chunk = items[page:page + PER]
        sheet = Image.new('RGB', (COLS * W, 40 + -(-len(chunk) // COLS) * H), 'white')
        draw = ImageDraw.Draw(sheet)
        pages = f'  page {page // PER + 1} of {-(-len(items) // PER)}' if len(items) > PER else ''
        draw.text((8, 8), f'{name}  ({len(items)} images, sorted by {key}, highest first){pages}', fill='black', font=head)
        for k, row in enumerate(chunk):
            x, y = (k % COLS) * W + 2, 40 + (k // COLS) * H
            sheet.paste(thumbs[row['image']], (x, y))
            draw.text((x, y + 202), f"#{row['row']} {SOURCE.get(row['source_kind'], row['source_kind'])}  "
                                    f"{key}={float(row[key]):.2f}", fill='black', font=font)
            draw.text((x, y + 217), f"cl{float(row['clouds_over_half']):.2f} nd{float(row['nodata']):.2f} "
                                    f"se{float(row['seams']):.2f} co{float(row['color_artifacts']):.2f} {row['scene']}",
                      fill='#444', font=font)
        sheet.save(out / (name + (f'_p{page // PER + 1}' if len(items) > PER else '') + '.png'))
    print(f'{name}: {len(items)}')

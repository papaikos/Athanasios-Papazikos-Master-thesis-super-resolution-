"""Blind human-validation set for one decisions folder (dataset read only).

Usage: python make_review_set.py decisions/<folder> [--removed 100] [--kept 150] [--seed 20261019]
Draws a simple random sample from each stratum (images listed for removal, images kept), shuffles them
together and writes thesis_data/review_<folder>/:
  images/R001.png ...                   full-resolution HR images, named by review ID only (no Jev answers)
  labels_to_fill.csv                    one row per image; fill in the label columns while looking at the images
  answer_key_open_after_labelling.csv   stratum and Jev decision per review ID
  review_info.json                      seed, sample sizes and stratum sizes (needed to weight the results)
Score the filled labels with score_review.py.
"""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import random
import tarfile

import numpy as np
from PIL import Image

from typesafe_image_filter import HERE, to_uint8

LABELS = {
    'has_defect': 'y/n: no-data areas, mosaic seams or stripes, or color artifacts (fog/haze is not a defect)',
    'defect_type': 'if y: nodata / seams / color / other',
    'mostly_clouds': 'y/n: clouds cover more than half of the image (fog/haze does not count)',
    'maritime_subject': 'y/n: shows a shoreline, vessels, or port/waterfront structures',
    'scene': 'open_sea / coast / port_or_marina / river_or_lake / inland / cloud',
    'notes': 'free text',
}


def main(folder, removed_n, kept_n, seed):
    folder = folder.resolve()
    policy = json.loads((folder / 'policy.json').read_text(encoding='utf-8'))
    dataset_dir = Path(json.loads((Path(policy['run_dir']) / 'run.json').read_text(encoding='utf-8'))['dataset_dir'])
    rows = list(csv.DictReader((folder / 'decisions.csv').open(encoding='utf-8')))
    strata = {'removed': [r for r in rows if r['decision'] == 'remove'], 'kept': [r for r in rows if r['decision'] == 'keep']}
    rng = random.Random(seed)
    sample = [dict(row, stratum=name) for name, size in (('removed', removed_n), ('kept', kept_n))
              for row in rng.sample(strata[name], min(size, len(strata[name])))]
    rng.shuffle(sample)
    out = HERE / 'thesis_data' / f'review_{folder.name}'
    (out / 'images').mkdir(parents=True)  # never overwrite a review set that may already hold labels
    wanted = {}
    for number, row in enumerate(sample, 1):
        row['review_id'] = f'R{number:03d}'
        wanted.setdefault((row['split'], row['tar']), {})[row['image']] = row['review_id']
    for (split, tar), names in wanted.items():
        with tarfile.open(dataset_dir / split / tar, 'r') as archive:
            for member in archive:
                if member.name in names:
                    with np.load(io.BytesIO(archive.extractfile(member).read()), allow_pickle=False) as data:
                        hr = to_uint8(data['hr']).transpose(1, 2, 0)
                    Image.fromarray(hr).save(out / 'images' / f'{names[member.name]}.png')
    with (out / 'labels_to_fill.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['review_id', *LABELS])
        writer.writerows([[row['review_id']] + [''] * len(LABELS) for row in sample])
    with (out / 'answer_key_open_after_labelling.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['review_id', 'image', 'source_kind', 'stratum', 'jev_decision', 'remove_reasons', 'candidates'])
        writer.writerows([[r['review_id'], r['image'], r['source_kind'], r['stratum'], r['decision'],
                           r['remove_reasons'], r['candidates']] for r in sample])
    (out / 'review_info.json').write_text(json.dumps(dict(
        created_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), decisions=folder.name, seed=seed,
        sample_sizes={name: sum(r['stratum'] == name for r in sample) for name in strata},
        stratum_sizes={name: len(items) for name, items in strata.items()}, labels=LABELS,
        instructions='Label every image in images/ in labels_to_fill.csv without opening the answer key. '
                     'Then run score_review.py on this folder.'), indent=2), encoding='utf-8')
    print(f'{len(sample)} images -> {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--removed', type=int, default=100)
    parser.add_argument('--kept', type=int, default=150)
    parser.add_argument('--seed', type=int, default=20261019)
    args = parser.parse_args()
    main(args.folder, args.removed, args.kept, args.seed)

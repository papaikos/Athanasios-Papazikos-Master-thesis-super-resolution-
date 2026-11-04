"""Score a labelled review set (from make_review_set.py) against Jev's decisions.

Usage: python score_review.py thesis_data/review_<folder>
Writes review_results.json in that folder: precision of the remove list, the miss rate among kept images,
and the estimated recall over the whole dataset (weighted by stratum size), each with a Wilson 95% interval.
"""
import csv
import json
import math
from pathlib import Path
import sys


def wilson(hits, n, z=1.96):
    if n == 0:
        return None
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return dict(rate=round(p, 4), hits=hits, n=n, ci95=[round(centre - half, 4), round(centre + half, 4)])


def main(folder):
    info = json.loads((folder / 'review_info.json').read_text(encoding='utf-8'))
    labels = {r['review_id']: r for r in csv.DictReader((folder / 'labels_to_fill.csv').open(encoding='utf-8'))}
    key = {r['review_id']: r for r in csv.DictReader((folder / 'answer_key_open_after_labelling.csv').open(encoding='utf-8'))}
    missing = [rid for rid, r in labels.items() if r['has_defect'].strip().lower() not in ('y', 'n')
               or r['mostly_clouds'].strip().lower() not in ('y', 'n')]
    if missing:
        raise SystemExit(f'{len(missing)} images still need has_defect and mostly_clouds labels (y/n), e.g. {missing[:5]}')

    def bad(rid):  # what the user's rules say should be removed: a defect, or mostly clouds
        return labels[rid]['has_defect'].strip().lower() == 'y' or labels[rid]['mostly_clouds'].strip().lower() == 'y'

    removed = [rid for rid in key if key[rid]['stratum'] == 'removed']
    kept = [rid for rid in key if key[rid]['stratum'] == 'kept']
    precision = wilson(sum(bad(r) for r in removed), len(removed))
    miss = wilson(sum(bad(r) for r in kept), len(kept))
    sizes = info['stratum_sizes']
    true_removed = precision['rate'] * sizes['removed'] if precision else 0
    missed = miss['rate'] * sizes['kept'] if miss else 0
    per_reason = {}
    for reason, label in (('mostly_clouds', 'mostly_clouds'), ('defect', 'has_defect')):
        flagged = [r for r in removed if reason in key[r]['remove_reasons']]
        per_reason[reason] = wilson(sum(labels[r][label].strip().lower() == 'y' for r in flagged), len(flagged))
    maritime = [r for r in kept if labels[r]['maritime_subject'].strip().lower() in ('y', 'n')]
    candidates = {}
    for candidate in ('inland', 'featureless_open_sea'):
        flagged = [r for r in maritime if candidate in key[r]['candidates']]
        candidates[candidate] = wilson(sum(labels[r]['maritime_subject'].strip().lower() == 'n' for r in flagged), len(flagged))
    results = dict(
        review=folder.name, decisions=info['decisions'], stratum_sizes=sizes, sample_sizes=info['sample_sizes'],
        precision_of_remove_list=precision, miss_rate_among_kept=miss,
        estimated_images_that_should_be_removed=round(true_removed + missed),
        estimated_recall=round(true_removed / (true_removed + missed), 4) if true_removed + missed else None,
        precision_by_reason=per_reason,
        candidate_agreement_no_maritime_subject=candidates,
        note='Precision and miss rate are per stratum (simple random samples); recall weights them by stratum size. '
             'Labels follow the user rules: fog/haze is neither a defect nor cloud.')
    (folder / 'review_results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())

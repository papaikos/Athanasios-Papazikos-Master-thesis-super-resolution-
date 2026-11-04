"""Test-retest agreement between two Jev runs over the images they share (no API calls).

Usage: python compare_runs.py runs/<run A> runs/<run B>
Writes thesis_data/retest_<A>_vs_<B>.json: per question, the mean and largest absolute difference
(Nouls and Scores), the share of identical Choice answers, and agreement on the final decision
under the current rules (with Cohen's kappa).
"""
import json
from pathlib import Path
import sys

from typesafe_image_filter import HERE, classify


def load(run_dir):
    return {r['sample_id']: r for r in map(json.loads, (run_dir / 'judgments.jsonl').read_text(encoding='utf-8').splitlines())}


def value(answer):
    return answer['noul'] if answer['type'] == 'noul' else answer['score'] if answer['type'] == 'score' else answer['choice']


def main(run_a, run_b):
    a, b = load(run_a), load(run_b)
    shared = sorted(set(a) & set(b))
    questions = {}
    for key in a[shared[0]]['answers']:
        pairs = [(value(a[s]['answers'][key]), value(b[s]['answers'][key])) for s in shared]
        if isinstance(pairs[0][0], str):
            questions[key] = dict(type='choice', identical_share=round(sum(x == y for x, y in pairs) / len(pairs), 4))
        else:
            diffs = [abs(x - y) for x, y in pairs]
            questions[key] = dict(type=a[shared[0]]['answers'][key]['type'], mean_abs_diff=round(sum(diffs) / len(diffs), 5),
                                  max_abs_diff=round(max(diffs), 5), identical_share=round(sum(d == 0 for d in diffs) / len(diffs), 4))
    decisions = [(bool(classify(a[s]['answers'])[0]), bool(classify(b[s]['answers'])[0])) for s in shared]
    agree = sum(x == y for x, y in decisions) / len(decisions)
    pa, pb = sum(x for x, _ in decisions) / len(decisions), sum(y for _, y in decisions) / len(decisions)
    chance = pa * pb + (1 - pa) * (1 - pb)
    kappa = (agree - chance) / (1 - chance) if chance < 1 else None
    result = dict(run_a=run_a.name, run_b=run_b.name, shared_images=len(shared),
                  models=sorted({a[s]['model'] for s in shared} | {b[s]['model'] for s in shared}),
                  decision_agreement=round(agree, 4), cohens_kappa=round(kappa, 4) if kappa is not None else None,
                  removed_in_a=sum(x for x, _ in decisions), removed_in_b=sum(y for _, y in decisions),
                  questions=questions)
    out = HERE / 'thesis_data' / f'retest_{run_a.name}_vs_{run_b.name}.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'questions'}, indent=2), '->', out)


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())

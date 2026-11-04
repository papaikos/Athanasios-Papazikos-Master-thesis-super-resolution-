"""Offline check of typesafe_image_filter: mocked API, real Mother images (read only), output in a temp folder.

Covers the request sent, the run logs, error logging, resuming, decide, the contact sheets and the rules.
"""
import asyncio, functools, hashlib, io, json, os, random, subprocess, sys, tempfile
from pathlib import Path
import httpx2
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import typesafe_image_filter as f

MOTHER = Path(os.environ.get('MARITIME_DATA_ROOT', 'Maritime_RGB_Expanded_20261008')) / 'Mother'
requests, fail_next = [], [True]


def fake_answer(question, rnd):
    if question['type'] == 'noul':
        return {'type': 'noul', 'noul': rnd.random()}
    if question['type'] == 'choice':
        weights = {option: rnd.random() for option in question['criteria']}
        total = sum(weights.values()); probs = {k: v / total for k, v in weights.items()}
        best = max(probs, key=probs.get)
        return {'type': 'choice', 'choice': best, 'confidence': probs[best], 'probabilities': probs}
    p = [rnd.random() for _ in question['criteria']]; total = sum(p); p = [x / total for x in p]
    return {'type': 'score', 'score': sum(i * x for i, x in enumerate(p)), 'confidence': max(p),
            'legend': {str(i): c for i, c in enumerate(question['criteria'])},
            'probabilities': {str(i): x for i, x in enumerate(p)}}


def handler(request):
    assert request.url.path == '/v1/systemone' and request.headers['TypeSafe-Danger-Zone-Images'] == 'true'
    boundary = request.headers['content-type'].split('boundary=')[1].encode()
    parts = {}
    for part in request.read().split(b'--' + boundary):
        if part.strip() in (b'', b'--'):
            continue
        head, _, data = part.partition(b'\r\n\r\n')
        parts[head.split(b'name="')[1].split(b'"')[0].decode()] = data.rstrip(b'\r\n')
    body = json.loads(parts.pop('request'))
    assert body['model'] == f.MODEL and set(body['questions']) == set(f.QUESTIONS)
    (ref,) = body['assets']
    assert body['state']['image'] == ref
    png = parts[body['assets'][ref]['part']]
    assert png[:8] == b'\x89PNG\r\n\x1a\n' and Image.open(io.BytesIO(png)).size == (512, 512)
    requests.append(hashlib.sha256(png).hexdigest())
    headers = {'x-typesafe-request-id': f'req_{len(requests)}'}
    if fail_next[0]:  # one deliberate non-retryable failure, to check errors.jsonl and resume
        fail_next[0] = False
        return httpx2.Response(400, headers=headers, json={'error': 'test failure'})
    rnd = random.Random(png)
    return httpx2.Response(200, headers=headers, json={
        'model': f.MODEL, 'usage': {'input_tokens': 1666, 'output_tokens': 9},
        'answers': {key: fake_answer(q, rnd) for key, q in body['questions'].items()}})


f.AsyncTypeSafeClient = functools.partial(f.AsyncTypeSafeClient, api_key='ts-test-key',
                                          transport=httpx2.MockTransport(handler))
tmp = Path(tempfile.mkdtemp())
run_dir = f.new_run(MOTHER, 'offline_test', 6, 7, tmp / 'runs')
for name in ('run.json', 'images.tsv', 'questions.json', 'script_snapshot.py', 'environment.txt'):
    assert (run_dir / name).exists(), name
snapshot = (run_dir / 'script_snapshot.py').read_text(encoding='utf-8')
assert 'apikey_' not in snapshot and "os.environ.get('TYPESAFE_API_KEY'" in snapshot

asyncio.run(f.judge(run_dir, 3))
log = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
errors = [json.loads(line) for line in (run_dir / 'errors.jsonl').read_text().splitlines()]
assert log['status'].startswith('incomplete') and log['images_judged'] == 5, log['status']
assert len(errors) == 1 and errors[0]['status'] == 400 and errors[0]['request_id'] == 'req_1', errors

asyncio.run(f.judge(run_dir, 3))  # resume retries only the failed image
lines = (run_dir / 'judgments.jsonl').read_text(encoding='utf-8').splitlines()
assert len(requests) == 7 and len(lines) == 6
(run_dir / 'judgments.jsonl').write_text('\n'.join(lines[:4]) + '\n', encoding='utf-8')  # simulate an interruption
asyncio.run(f.judge(run_dir, 3))
log = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
records = [json.loads(line) for line in (run_dir / 'judgments.jsonl').read_text(encoding='utf-8').splitlines()]
assert len(requests) == 9 and log['status'] == 'complete' and len(log['sessions']) == 3
assert len({r['sample_id'] for r in records}) == 6 and log['input_tokens_total'] == 6 * 1666
record = records[0]
assert record['request_id'].startswith('req_') and record['png_sha256'] in requests and record['native_tif'].endswith('.tif')
assert record['hr_sha256'] and record['split'] in f.SPLITS and set(record['answers']) == set(f.QUESTIONS)
assert (MOTHER / 'native' / 'objects' / record['native_tif']).exists()
print('judge, logs, errors and resume OK:', run_dir)

out = f.decide(run_dir, 'offline_test', tmp / 'decisions')
removed = (out / 'remove_list.txt').read_text(encoding='utf-8').split()
policy = json.loads((out / 'policy.json').read_text(encoding='utf-8'))
assert all(name.endswith('.npz') for name in removed) and policy['images'] == 6
assert sum(json.loads((out / 'summary.json').read_text())['decision'].values()) == 6
subprocess.run([sys.executable, str(HERE / 'make_contact_sheets.py'), str(out)], check=True)
print('decide and contact sheets OK:', sorted(p.name for p in (out / 'sheets').glob('*.png')))


def answers(over_half, nodata=.1, seams=.1, color=.1, water=.9, shore=.9, vessels=.2, port=.2, scene='natural_coast'):
    rest = (1 - over_half) / 3
    return {'clouds': {'score': 0, 'probabilities': {'0': rest, '1': rest, '2': rest, '3': over_half, '4': 0}},
            'nodata': {'noul': nodata}, 'seams': {'noul': seams}, 'color_artifacts': {'noul': color},
            'water': {'noul': water}, 'shoreline': {'noul': shore}, 'vessels': {'noul': vessels},
            'port_structures': {'noul': port}, 'large_ship': {'noul': .1},
            'scene': {'choice': scene, 'confidence': .8}, 'vessel_count': {'score': 0}}


cases = {
    'clean coast':            (answers(.05), [], []),
    'mostly clouds':          (answers(.60), ['mostly_clouds'], []),
    'clouds just under half': (answers(.49), [], []),
    'seams':                  (answers(.05, seams=.8), ['defect_seams'], []),
    'seams at cut-off':       (answers(.05, seams=.7), [], []),
    'no-data and clouds':     (answers(.70, nodata=.71), ['defect_nodata', 'mostly_clouds'], []),
    'color above 0.65':       (answers(.05, color=.66), ['defect_color_artifacts'], []),
    'color at cut-off':       (answers(.05, color=.65), [], []),
    'open sea':               (answers(.05, shore=.05, vessels=.1, port=.05, scene='open_sea'), [], ['featureless_open_sea']),
    'forest called coast':    (answers(.05, water=.3, shore=.45), [], ['inland']),
    'inland town':            (answers(.05, water=.05, shore=.05, scene='inland'), [], ['inland']),
    'uncertain coast':        (answers(.05, water=.9, shore=.5), [], ['uncertain_maritime_subject']),
}
for name, (a, remove, candidates) in cases.items():
    got = f.classify(a)[:2]
    assert got == (remove, candidates), (name, got)
print('rules OK:', len(cases), 'cases')

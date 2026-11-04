"""Ask TypeSafe Jev about each HR image and list the images to remove.

Two steps, so the rules can be re-tuned without paying for inference again:

  judge   send each HR image (512x512 RGB) to Jev with QUESTIONS. Every run gets
          its own folder in runs/ with the answers and a full log; an
          interrupted run continues with --resume.
  decide  apply the rules in code. Every pass gets its own folder in decisions/
          with remove_list.txt (one image name per line), decisions.csv,
          policy.json and summary.json.

The dataset is only read, never changed: nothing is removed, the result is a
list. Mother and D1/D2/D3 share image IDs, so the list applies to all of them.

Usage (pip install typesafe-sdk numpy pillow):
  python typesafe_image_filter.py judge --run-name pilot2_mother --dataset-dir ../Maritime_RGB_Expanded_20261008/Mother --sample 300
  python typesafe_image_filter.py judge --resume runs/<run folder>
  python typesafe_image_filter.py decide --run-dir runs/<run folder> --name rules_v3
"""
import argparse
import asyncio
import collections
import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import io
import json
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import tarfile

import numpy as np
from PIL import Image as PILImage
from typesafe_sdk import AsyncTypeSafeClient, Choice, Image, Noul, NoulCriteria, Score, TypeSafeError

API_KEY = os.environ.get('TYPESAFE_API_KEY', '')  # set it in the environment, never in this file
MODEL = 'jev-1.13.0'  # pinned: thresholds below are tuned against one version
IMAGES_HEADER = {'TypeSafe-Danger-Zone-Images': 'true'}  # images are opt-in ("dreamer-ready")
PRICE_USD_PER_MTOK = 0.042  # Jev input price, docs.typesafe.ai/models (read 2026-10-16); output tokens are free
TIMEOUT_S = 60.0  # per request; the SDK default of 10 s timed out ~2% of image requests at concurrency 16
SPLITS = ('train', 'validation', 'test')
HERE = Path(__file__).resolve().parent

NOT_A_DEFECT = ('A natural-looking image. Fog, haze, clouds, sun glint, shadows, waves, and calm uniform '
                'water are not defects.')

QUESTIONS = {
    'clouds': Score(
        instructions='How much of `image` is covered by clouds? Fog, haze, mist, and smoke do not count as cloud.',
        criteria=[
            'No clouds.',
            'A few clouds covering less than a quarter of the image.',
            'Clouds covering roughly a quarter to a half of the image.',
            'Clouds covering more than half of the image, with some ground or water still visible.',
            'Clouds covering almost all of the image, so the ground or water is barely visible.',
        ]),
    'nodata': Noul(
        instructions='Does `image` contain no-data areas?',
        criteria=NoulCriteria(
            true='Areas of solid black, solid white, or flat single-color fill with no texture at all, or missing '
                 'strips where the sensor recorded nothing.',
            false=NOT_A_DEFECT + ' Dark or uniform but real water is not no-data.')),
    'seams': Noul(
        instructions='Does `image` show mosaic seams or sensor stripes?',
        criteria=NoulCriteria(
            true='Straight lines where image strips meet with misaligned content or a sudden step in brightness or '
                 'color, or regular stripes or banding across the image.',
            false=NOT_A_DEFECT)),
    'color_artifacts': Noul(
        instructions='Does `image` show color artifacts?',
        criteria=NoulCriteria(
            true='Color fringes or rainbow halos along edges, red, green, and blue channels visibly shifted from '
                 'each other, or unnatural color blotches.',
            false=NOT_A_DEFECT)),
    'water': Noul(
        instructions='Is open water (sea, surf, harbor basin, river, or lake surface) visible anywhere in `image`?'),
    'shoreline': Noul(
        instructions='Does a boundary between land and water run through `image`?',
        criteria=NoulCriteria(
            true='A coastline, beach, surf line, cliff edge, riverbank, or harbor edge where land meets water is visible.',
            false='The image is all water or all land, with no land-water edge.')),
    'vessels': Noul(
        instructions='Are boats or ships, or their wakes, visible on the water in `image`?',
        criteria=NoulCriteria(
            true='At least one vessel hull, moored or moving boat, or wake trail on the water.',
            false='No vessels and no wakes.')),
    'port_structures': Noul(
        instructions='Does `image` show man-made waterfront structures such as docks, piers, quays, jetties, '
                     'marinas, breakwaters, or container terminals?'),
    'scene': Choice(
        instructions='What is the main scene in `image`?',
        criteria={
            'open_sea': 'Open water only, with no land or coastline in view.',
            'natural_coast': 'A natural coastline: beach, cliffs, rocky shore, reef, or wetland meeting the water.',
            'built_coast': 'A built-up shoreline without a port: seawalls, promenades, waterfront buildings, or '
                           'roads along the water.',
            'port': 'A port or industrial waterfront: quays, cranes, container yards, shipyards, or large docks.',
            'marina': 'A marina or small-boat harbor with pontoons and moored boats.',
            'river_or_estuary': 'A river, canal, or estuary channel.',
            'lake': 'A lake or reservoir.',
            'inland': 'Land with no visible water.',
            'mostly_cloud': 'Clouds hide most of the view.',
        }),
    'vessel_count': Score(
        instructions='How many boats or ships are visible in `image`?',
        criteria=[
            'No vessels visible.',
            'One or two vessels.',
            'Three to ten vessels.',
            'More than ten vessels, such as a busy marina, anchorage, or port.',
        ]),
    'large_ship': Noul(
        instructions='Is there a large ship in `image`, about 50 m or longer, such as a cargo ship, tanker, ferry, '
                     'or cruise ship?',
        criteria=NoulCriteria(
            true='At least one ship about 50 m or longer (roughly 20 pixels or more at this scale).',
            false='Only small boats, or no vessels.')),
}
DEFECTS = ('nodata', 'seams', 'color_artifacts')

# Rules v5 (2026-10-20). Raw answers stay in the run folder, so changing these needs only `decide`.
# v4 was nodata 0.7, seams 0.7, color_artifacts 0.65. Seams went down to 0.6 because the 0.60-0.70 band
# was still mostly real seams; colour went up to 0.75 (the user asked for it high) because 0.65-0.70 was
# mostly haze. See LAB_NOTEBOOK 2026-10-20 and exploration/2026-10-20_cutoff_band_review.
DEFECT_REMOVE = dict(nodata=0.7, seams=0.6, color_artifacts=0.75)  # remove when a defect Noul is above its cut-off
CLOUDS_MOSTLY = ('3', '4')   # `clouds` levels "more than half" and "almost all"
CLOUDS_REMOVE = 0.5          # remove when P(clouds cover more than half) is at least this
YES, NO = 0.7, 0.3           # maritime-subject bands for the reported (not removed) candidates
UNCERTAIN_WATER_MIN = 0.5
CANDIDATES = ('inland', 'featureless_open_sea', 'uncertain_maritime_subject')
NO_MARITIME = (f'max(shoreline, vessels, port_structures) <= {NO}, or < {YES} with water < {UNCERTAIN_WATER_MIN}')
RULES = dict(
    remove=[f'defect_<name>: {name} Noul > {cut}' for name, cut in DEFECT_REMOVE.items()]
           + [f'mostly_clouds: P(clouds level 3 or 4, i.e. more than half covered) >= {CLOUDS_REMOVE}'],
    candidates_not_removed=[
        f'featureless_open_sea: {NO_MARITIME}, and scene is open_sea',
        f'inland: {NO_MARITIME}, and scene is not open_sea',
        f'uncertain_maritime_subject: {NO} < max(shoreline, vessels, port_structures) < {YES} '
        f'with water >= {UNCERTAIN_WATER_MIN}'])


def now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def stamp():
    return datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M%SZ')


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def questions_json():
    return json.dumps({key: q.model_dump(exclude_none=True) for key, q in QUESTIONS.items()}, indent=2, sort_keys=True)


def redacted_script():
    """This script's source with the API key blanked out, for the logs."""
    source = Path(__file__).read_text(encoding='utf-8')
    return re.sub(r"^API_KEY = '.*'$", "API_KEY = '<REDACTED>'", source, flags=re.M)


def to_uint8(hr):
    """Mother stores float32 in [0, 1]; quantize exactly as D1 does (code/src/maritime_uint8.py)."""
    if hr.dtype == np.uint8:
        return hr
    return np.floor(np.clip(hr.astype(np.float64), 0, 1) * 255 + 0.5).astype(np.uint8)


def encode_png(hr):
    buffer = io.BytesIO()
    PILImage.fromarray(np.ascontiguousarray(hr.transpose(1, 2, 0))).save(buffer, 'PNG')
    return buffer.getvalue()


def tile_state(png, provenance, width_px):
    """Top-down HR image plus the scale facts Jev needs to judge it."""
    gsd = float(provenance['hr_transform'][0])
    return {
        'image': Image(png),
        'about_image': {
            'view': 'Top-down satellite or aerial RGB tile, north up',
            'ground_sample_distance_m': gsd,
            'tile_width_m': gsd * width_px,
            'scale': f'One pixel is {gsd:g} m: a 50 m ship is about {50 / gsd:.0f} pixels long; '
                     'boats under 10 m are only a few pixels.',
        },
    }


def native_tif(provenance):
    """The image's native TIF in native/objects, found as code/scripts/maritime_uint8_download.native_references does."""
    window = provenance.get('native_window_ref')
    reference = window['object'] if window else provenance.get('native_context')
    return reference['sha256'] + '.tif' if reference else None


def index_images(dataset_dir):
    """(split, tar, sample_id) for every image in the dataset's split TARs, in file order."""
    images = []
    for split in SPLITS:
        for shard in sorted((dataset_dir / split).glob('*.tar')):
            with tarfile.open(shard, 'r') as archive:
                images += [(split, shard.name, Path(member.name).stem) for member in archive]
    return images


def iter_images(dataset_dir, selected):
    """Yield (split, tar, provenance, hr_uint8) for the selected images, reading each TAR once (read only)."""
    wanted = collections.defaultdict(set)
    for split, shard, sample_id in selected:
        wanted[(split, shard)].add(sample_id)
    for (split, shard), ids in wanted.items():
        with tarfile.open(dataset_dir / split / shard, 'r') as archive:
            for member in archive:
                if Path(member.name).stem in ids:
                    with np.load(io.BytesIO(archive.extractfile(member).read()), allow_pickle=False) as data:
                        provenance = json.loads(str(data['provenance']))
                        hr = to_uint8(data['hr'])
                    yield split, shard, provenance, hr


def new_run(dataset_dir, name, sample, seed, runs_root):
    """Create a never-reused run folder with everything needed to reproduce the run."""
    run_dir = runs_root / f'{stamp()}_{name}'
    run_dir.mkdir(parents=True)
    index = index_images(dataset_dir)
    selected = random.Random(seed).sample(index, sample) if sample else index
    with (run_dir / 'images.tsv').open('w', encoding='utf-8', newline='') as handle:
        csv.writer(handle, delimiter='\t').writerows([('split', 'tar', 'sample_id'), *selected])
    questions, script = questions_json(), redacted_script()
    (run_dir / 'questions.json').write_text(questions, encoding='utf-8')
    (run_dir / 'script_snapshot.py').write_text(script, encoding='utf-8')
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True).stdout
    (run_dir / 'environment.txt').write_text(f'python {sys.version}\n{platform.platform()}\n\n{freeze}', encoding='utf-8')
    manifest = dataset_dir / 'manifest.json'
    write_json(run_dir / 'run.json', dict(
        status='created', created_utc=now(), dataset_dir=str(dataset_dir),
        dataset_manifest_sha256=sha256_file(manifest) if manifest.exists() else None,
        selection=dict(mode=f'random sample of {sample}' if sample else 'all images', seed=seed if sample else None,
                       splits=SPLITS, images_in_dataset=len(index), images_selected=len(selected)),
        model_requested=MODEL, models_answered=[], images_header=IMAGES_HEADER,
        questions_sha256=hashlib.sha256(questions.encode()).hexdigest(),
        script_snapshot_sha256=hashlib.sha256(script.encode()).hexdigest(),
        versions=dict(python=platform.python_version(),
                      **{package: metadata.version(package) for package in ('typesafe-sdk', 'numpy', 'pillow')}),
        image_prep='HR quantized to uint8 with floor(clip(x,0,1)*255+0.5) (identical to D1 storage), sent as a '
                   'lossless 512x512 PNG with the scale facts from tile_state()',
        price_usd_per_mtok=PRICE_USD_PER_MTOK, sessions=[]))
    return run_dir


async def judge(run_dir, concurrency):
    log = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
    if log['questions_sha256'] != hashlib.sha256(questions_json().encode()).hexdigest() or log['model_requested'] != MODEL:
        raise SystemExit('QUESTIONS or MODEL changed since this run started: start a new run instead of mixing answers')
    with (run_dir / 'images.tsv').open(encoding='utf-8', newline='') as handle:
        selected = [tuple(row) for row in csv.reader(handle, delimiter='\t')][1:]
    path = run_dir / 'judgments.jsonl'
    done = set()
    if path.exists():
        done = {json.loads(line)['sample_id'] for line in path.read_text(encoding='utf-8').splitlines() if line}
    todo = [image for image in selected if image[2] not in done]
    session = dict(started_utc=now(), argv=sys.argv, concurrency=concurrency, timeout_s=TIMEOUT_S,
                   already_done=len(done), to_judge=len(todo), judged=0, failed=0)
    log['sessions'].append(session)
    log['status'] = 'running'
    write_json(run_dir / 'run.json', log)
    models = set(log['models_answered'])
    slots = asyncio.Semaphore(concurrency)
    pending = set()
    try:
        async with AsyncTypeSafeClient(api_key=API_KEY, model=MODEL, headers=IMAGES_HEADER, timeout=TIMEOUT_S) as client:
            with path.open('a', encoding='utf-8') as sink, (run_dir / 'errors.jsonl').open('a', encoding='utf-8') as errors:
                async def one(split, shard, provenance, hr):
                    png = encode_png(hr)
                    try:
                        response = await client.system_one(tile_state(png, provenance, hr.shape[2]), QUESTIONS)
                    except TypeSafeError as error:
                        session['failed'] += 1
                        errors.write(json.dumps(dict(
                            sample_id=provenance['sample_id'], split=split, tar=shard, utc=now(),
                            error=type(error).__name__, status=getattr(error, 'status', None),
                            request_id=getattr(error, 'request_id', None), message=str(error))) + '\n')
                        errors.flush()
                        return
                    finally:
                        slots.release()
                    models.add(response.model)
                    session['judged'] += 1
                    sink.write(json.dumps(dict(
                        sample_id=provenance['sample_id'], split=split, tar=shard,
                        source_kind=provenance.get('source_kind'), native_tif=native_tif(provenance),
                        hr_sha256=provenance.get('hr_sha256'), png_sha256=hashlib.sha256(png).hexdigest(),
                        request_id=response.raw_http_response.headers.get('x-typesafe-request-id'),
                        model=response.model, answered_utc=now(), input_tokens=response.usage.input_tokens,
                        answers={key: answer.model_dump() for key, answer in response.answers.items()})) + '\n')
                    sink.flush()

                for split, shard, provenance, hr in iter_images(Path(log['dataset_dir']), todo):
                    await slots.acquire()  # bounds both in-flight requests and decoded images in memory
                    task = asyncio.create_task(one(split, shard, provenance, hr))
                    pending.add(task)
                    task.add_done_callback(pending.discard)
                await asyncio.gather(*pending)
        log['status'] = 'complete' if len(done) + session['judged'] == len(selected) else 'incomplete (resume to retry failures)'
    except BaseException:
        log['status'] = 'interrupted (resume to continue)'
        raise
    finally:
        session['ended_utc'] = now()
        records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line] if path.exists() else []
        tokens = sum(record['input_tokens'] for record in records)
        log.update(models_answered=sorted(models), images_judged=len(records), input_tokens_total=tokens,
                   estimated_cost_usd=round(tokens / 1e6 * PRICE_USD_PER_MTOK, 4))
        write_json(run_dir / 'run.json', log)
    print(f'{run_dir.name}: {log["status"]}; {len(records)}/{len(selected)} images judged, '
          f'{session["failed"]} failed this session, {tokens} input tokens (~${log["estimated_cost_usd"]})')


def classify(answers):
    """Return (remove_reasons, candidate_reasons, values) from raw Jev answers. Rules only; no inference."""
    def noul(key):
        return answers[key]['noul']
    clouds_over_half = sum(answers['clouds']['probabilities'][level] for level in CLOUDS_MOSTLY)
    maritime = max(noul('shoreline'), noul('vessels'), noul('port_structures'))
    remove = [f'defect_{key}' for key in DEFECTS if noul(key) > DEFECT_REMOVE[key]]
    if clouds_over_half >= CLOUDS_REMOVE:
        remove.append('mostly_clouds')
    candidates = []
    if maritime <= NO or (maritime < YES and noul('water') < UNCERTAIN_WATER_MIN):
        # `water` cannot separate grey open sea from land (pilot 1), so the scene answer splits them
        candidates.append('featureless_open_sea' if answers['scene']['choice'] == 'open_sea' else 'inland')
    elif maritime < YES:
        candidates.append('uncertain_maritime_subject')
    values = dict(clouds_over_half=clouds_over_half, clouds_score=answers['clouds']['score'],
                  **{key: noul(key) for key in (*DEFECTS, 'water', 'shoreline', 'vessels', 'port_structures', 'large_ship')},
                  maritime=maritime, scene=answers['scene']['choice'], scene_confidence=answers['scene']['confidence'],
                  vessel_count=answers['vessel_count']['score'])
    return remove, candidates, values


def decide(run_dir, name, decisions_root):
    run_log = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
    judgments = run_dir / 'judgments.jsonl'
    records = [json.loads(line) for line in judgments.read_text(encoding='utf-8').splitlines() if line]
    out = decisions_root / f'{stamp()}_{name}'
    out.mkdir(parents=True)
    rows, removed = [], []
    reported = {candidate: [] for candidate in CANDIDATES}
    counts = collections.defaultdict(collections.Counter)
    for record in records:
        remove, candidates, values = classify(record['answers'])
        decision = 'remove' if remove else 'keep'
        if remove:
            removed.append(record['sample_id'] + '.npz')
        else:
            for candidate in candidates:
                reported[candidate].append(record['sample_id'] + '.npz')
        rows.append(dict(image=record['sample_id'] + '.npz', split=record['split'], tar=record['tar'],
                         native_tif=record['native_tif'], source_kind=record['source_kind'], decision=decision,
                         remove_reasons=';'.join(remove), candidates=';'.join(candidates),
                         **{key: round(value, 4) if isinstance(value, float) else value for key, value in values.items()}))
        counts['decision'][decision] += 1
        counts['remove_reason'].update(remove)
        counts[f'candidates_among_{decision}'].update(candidates)
        counts[f'source_kind_{decision}'][record['source_kind']] += 1
        counts[f'split_{decision}'][record['split']] += 1
        counts['scene'][values['scene']] += 1
    with (out / 'decisions.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / 'remove_list.txt').write_text(''.join(f'{image}\n' for image in removed), encoding='utf-8')
    for candidate, images in reported.items():  # kept images, reported for the user to decide on
        (out / f'candidates_{candidate}.txt').write_text(''.join(f'{image}\n' for image in images), encoding='utf-8')
    (out / 'script_snapshot.py').write_text(redacted_script(), encoding='utf-8')
    write_json(out / 'policy.json', dict(
        name=name, created_utc=now(), run_dir=str(run_dir.resolve()), run_status=run_log['status'],
        judgments_sha256=sha256_file(judgments), images=len(records), rules=RULES,
        thresholds=dict(DEFECT_REMOVE=DEFECT_REMOVE, CLOUDS_MOSTLY=CLOUDS_MOSTLY, CLOUDS_REMOVE=CLOUDS_REMOVE,
                        YES=YES, NO=NO, UNCERTAIN_WATER_MIN=UNCERTAIN_WATER_MIN),
        note='remove_list.txt and candidates_*.txt only list image names; nothing in the dataset is removed or '
             'changed. Candidate lists hold kept images only.'))
    write_json(out / 'summary.json', {key: dict(counter) for key, counter in counts.items()})
    print(f'{out.name}: {len(records)} images, {len(removed)} listed for removal -> {out / "remove_list.txt"}; '
          + ', '.join(f'{candidate} {len(images)}' for candidate, images in reported.items()))
    if run_log['status'] != 'complete':
        print(f'warning: run status is "{run_log["status"]}"', file=sys.stderr)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('judge', help='ask Jev about each image; logs everything in a run folder')
    start = run.add_mutually_exclusive_group(required=True)
    start.add_argument('--run-name', help='start a new run (needs --dataset-dir)')
    start.add_argument('--resume', type=Path, help='continue an existing run folder')
    run.add_argument('--dataset-dir', type=Path, help='e.g. ../Maritime_RGB_Expanded_20261008/Mother (read only)')
    run.add_argument('--sample', type=int, default=0, help='judge a random sample of N images (0 = all)')
    run.add_argument('--seed', type=int, default=20261019)
    run.add_argument('--concurrency', type=int, default=8)
    rules = commands.add_parser('decide', help='apply the rules to a run; writes a new decisions folder')
    rules.add_argument('--run-dir', type=Path, required=True)
    rules.add_argument('--name', required=True)
    args = parser.parse_args(argv)
    if args.command == 'judge':
        if args.run_name and not args.dataset_dir:
            parser.error('--run-name needs --dataset-dir')
        run_dir = args.resume or new_run(args.dataset_dir.resolve(), args.run_name, args.sample, args.seed, HERE / 'runs')
        asyncio.run(judge(run_dir, args.concurrency))
    else:
        decide(args.run_dir, args.name, HERE / 'decisions')


if __name__ == '__main__':
    main()

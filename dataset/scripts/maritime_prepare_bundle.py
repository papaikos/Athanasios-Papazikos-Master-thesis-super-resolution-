"""Package code and metadata only, then upload create-only for a cloud CPU pilot."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

from scripts.maritime_build_control import BUCKET, client_from_gcloud


def main():
    root = Path('.')
    report = root / 'reports/maritime-build-20261006'
    inventory = json.loads((report / 'control/source-inventory.json').read_text())
    # Rejoin fresh inventory to propagate newly recorded export-mask semantics.
    from scripts.maritime_build_control import join_sources
    manifests = [json.loads((root / p).read_text()) for p in inventory['ancestor_manifests']]
    assets = {a['id']: a for a in json.loads((root / 'reports/skysat-inventory.json').read_text())}
    sources = join_sources(manifests, json.loads((report / 'control/raw-objects.json').read_text()), BUCKET, assets)
    pilot = [r for r in sources if '/expansion-20260929/raw/' in r['uri']]
    files = {}
    for pattern in ('src/maritime*.py', 'scripts/maritime*.py', 'scripts/maritime*.sh',
                    'scripts/maritime*.Dockerfile', 'tests/test_maritime*.py'):
        for path in sorted(root.glob(pattern)):
            files[path.as_posix()] = path.read_bytes()
    for name, payload in [('source-inventory.json', sources), ('pilot-sources.json', pilot)]:
        files['inputs/' + name] = json.dumps(payload, sort_keys=True, indent=2).encode()
    files['inputs/plan.md'] = (report / 'PLAN.md').read_bytes()
    files['inputs/catalog-inventory.json'] = (root / 'reports/skysat-inventory.json').read_bytes()
    files['src/skysat_prepare.py'] = (root / 'src/skysat_prepare.py').read_bytes()
    files['inputs/pilot-review-references.json'] = (report / 'control/pilot-review-references.json').read_bytes()
    files['inputs/planet-sources.json'] = (report / 'control/planet-sources.json').read_bytes()
    files['inputs/disaster-skysat-sources.json'] = (report / 'control/disaster-skysat-sources.json').read_bytes()
    files['inputs/reuse-approved-assets.json'] = (report / 'control/reuse-approved-assets.json').read_bytes()
    files['inputs/site-metadata.json'] = (report / 'control/site-metadata.json').read_bytes()
    files['inputs/config.json'] = json.dumps({
        'project': inventory['project'], 'bucket': BUCKET, 'zone': 'europe-west4-a',
        'release_prefix': 'skysat/thesis-datasets/v1', 'run_id': 'maritime-20261006',
        'release_seed': 20261006, 'mother_target_bytes': 60000000000,
        'mother_max_bytes': 60100000000, 'pilot_only': True,
        'required_sites': 30, 'required_countries': 3,
        'minimum_validation_sites': 5, 'minimum_test_sites': 10,
        'native_gsd_m': 0.8, 'core_pixels': 1600, 'halo_pixels': 50,
        'source_policy': 'free_or_free_academic_skysat_only',
        'final_ready_markers_authorized_only_after_all_gates': True,
    }, sort_keys=True, indent=2).encode()
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode='w') as archive:
            for name, payload in sorted(files.items()):
                member = tarfile.TarInfo(name)
                member.size = len(payload)
                member.mode = 0o644
                archive.addfile(member, io.BytesIO(payload))
    payload = buf.getvalue()
    sha = hashlib.sha256(payload).hexdigest()
    local = report / 'control' / ('code-' + sha[:16] + '.tar.gz')
    local.write_bytes(payload)
    bucket = client_from_gcloud().bucket(BUCKET)
    name = 'skysat/thesis-datasets/v1/control/runtime/code-' + sha + '.tar.gz'
    blob = bucket.blob(name)
    if not blob.exists():
        blob.upload_from_string(payload, if_generation_match=0, checksum='crc32c')
    blob.reload()
    if hashlib.sha256(blob.download_as_bytes(if_generation_match=int(blob.generation))).hexdigest() != sha:
        raise ValueError('Code bundle readback mismatch')
    receipt = {'uri': 'gs://' + BUCKET + '/' + name, 'generation': str(blob.generation),
               'bytes': blob.size, 'sha256': sha, 'crc32c': blob.crc32c,
               'code_files': {n: hashlib.sha256(p).hexdigest() for n, p in files.items()},
               'local_path': str(local.resolve()), 'source_objects': len(sources), 'pilot_objects': len(pilot)}
    (report / 'control/runtime-bundle.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()

"""GCE-only additive uint8 release preserving immutable float baseline data.

SkySat, Maxar, EarthView TOA mosaics and audited tier-B academic rasters are accepted.
Baseline children are converted from pinned float shards; baseline Mother
shards/native references stay unchanged. New source contexts use the original
processing recipes before final, explicitly lossy uint8 quantization.
"""
import argparse
import copy
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import hashlib
import io
import json
import math
from pathlib import Path
import re
import tarfile

import numpy as np

from src.maritime_dataset import MaritimeDataset
from src.maritime_recipes import RECIPE_VERSIONS, array_hash, degrade
from src.maritime_uint8 import (QUANTIZATION, RGB_PAYLOAD_BYTES, STORAGE_VERSION,
                              encode_uint8_pair)
from src.maritime_uint8_dataset import MaritimeUint8Dataset
from src.cloud_config import GCP_PROJECT


EXPANSION_GATE_NAMES = (
    'source_rights_and_resolution', 'source_support_and_geometry',
    'marine_content_and_quality_audit', 'duplicate_and_leakage_audit',
)


def validate_expansion_gate_evidence(document, gate, *, population_sha256,
                                     baseline_index_sha256, code_sha256):
    """Bind an explicit review decision to this exact release population.

    Structural checks cannot establish the truth of an audit. The controller
    must inspect substantive evidence before declaring each named check true.
    Native replay and numerical verification remain mandatory in this worker.
    """
    expected = dict(selected_population_sha256=population_sha256,
                    baseline_index_sha256=baseline_index_sha256,
                    code_sha256=code_sha256)
    if (gate not in EXPANSION_GATE_NAMES or not isinstance(document, dict)
            or document.get('schema') != 'maritime-expansion-gate-evidence-v1'
            or document.get('gate') != gate or document.get('passed') is not True
            or any(not re.fullmatch('[0-9a-f]{64}', str(value))
                   or document.get(key) != value for key, value in expected.items())
            or not isinstance(document.get('scope'), str) or not document['scope'].strip()
            or not isinstance(document.get('checks'), dict) or not document['checks']
            or any(not isinstance(key, str) or not key or value is not True
                   for key, value in document['checks'].items())):
        raise ValueError('Expansion gate evidence is incomplete or belongs to another release')


def preserve_baseline(baseline, document):
    """Keep every baseline ID/source/split; add training rows only."""
    rows = document['selection']
    by_id = {row['sample_id']: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError('Duplicate expanded sample ID')
    original = {row['sample_id']: row for row in baseline['selection']['selection']}
    if set(original) != {sample['sample_id'] for sample in baseline['samples']}:
        raise ValueError('Baseline selection and manifest identities disagree')
    if not set(original).issubset(by_id):
        raise ValueError('Missing preserved baseline sample')
    for identity, old in original.items():
        for key in ('source', 'source_kind', 'split', 'site_id', 'geographic_group_id', 'acquisition_id'):
            if by_id[identity].get(key) != old.get(key):
                raise ValueError('Baseline source/identity/split replacement: ' + identity + ':' + key)
    old_assignments = baseline['selection'].get('splits', {}).get('assignment', {})
    new_assignments = document.get('splits', {}).get('assignment', {})
    if any(new_assignments.get(group) != split for group, split in old_assignments.items()):
        raise ValueError('Frozen baseline geographic assignments changed')
    added = [row for row in rows if row['sample_id'] not in original]
    if any(row['split'] != 'train' for row in added):
        raise ValueError('Expansion must preserve evaluation population; new rows must be train')
    return added


def stratified_recompute(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['split'], row.get('primary_category'), row['source_kind'])].append(row['sample_id'])
    return {identity for identities in groups.values() for identity in
            sorted(identities, key=lambda value: hashlib.sha256(value.encode()).hexdigest())[:math.ceil(len(identities)*.05)]}


def added_batches(rows, shard_samples):
    """Order aerial cores by pinned full parent to reuse verified source bytes.

    This affects shard scheduling only. Every core still undergoes original
    source replay; SkySat original-window ownership stays in separate batches.
    Multiple aerial parents may share a bounded shard, avoiding tiny shards.
    """
    if not 1 <= shard_samples <= 32:
        raise ValueError('Shard samples must be 1..32')
    def original_ref(row):
        source = row['source']
        if row['source_kind'] == 'aerial_rgb':
            return source['source_original_object']
        return source.get('source_object') or source.get('mother') or source['native_context']
    def order(row):
        ref = original_ref(row)
        return row['split'], ref['uri'], str(ref['generation']), row['sample_id']
    batches, current, previous = [], [], None
    for row in sorted(rows, key=order):
        ref = original_ref(row)
        key = ((row['split'], ref['uri'], str(ref['generation']))
               if row['source_kind'] == 'skysat_raw_window' else (row['split'], 'materialized'))
        if current and (len(current) == shard_samples or key != previous):
            batches.append(current)
            current = []
        current.append(row)
        previous = key
    if current:
        batches.append(current)
    return batches


def register_hr_hash(seen, sample, *, baseline):
    """Block identical added uint8 targets, preserving reported legacy repeats."""
    hashes = {sample['uint8_storage'][name]['hr_sha256'] for name in RECIPE_VERSIONS}
    if len(hashes) != 1:
        raise ValueError('Shared uint8 HR hash mismatch')
    digest = next(iter(hashes))
    prior = seen.get(digest)
    if prior is not None and not baseline:
        raise ValueError('Duplicate added uint8 HR: ' + prior + ' and ' + sample['sample_id'])
    seen.setdefault(digest, sample['sample_id'])
    return prior


def _npz(pair, metadata):
    from scripts.maritime_pilot import deterministic_npz
    return deterministic_npz(dict(**pair, provenance=np.asarray(json.dumps(metadata, sort_keys=True))))


def _member(archive, identity, payload):
    from scripts.maritime_ultimate_release import tar_member
    tar_member(archive, identity+'.npz', payload)


def _encode(pair, metadata, common, recipe, seed):
    if (metadata['recipe']['recipe'] != RECIPE_VERSIONS[recipe]
            or metadata['recipe'].get('release_seed') != seed
            or metadata['recipe'].get('variant_id') != '0'):
        raise ValueError('Baseline/original recipe seed or variant mismatch')
    for key in ('sample_id', 'split', 'site_id', 'geographic_group_id', 'hr_sha256', 'hr_transform', 'lr_transform'):
        if metadata[key] != common[key]:
            raise ValueError('Pair/common provenance mismatch: ' + key)
    if metadata.get('context_sha256') is not None and metadata['context_sha256'] != common.get('context_sha256'):
        raise ValueError('Pair/common context hash mismatch')
    enriched = dict(common)
    enriched.update(metadata)  # Preserve every original float recipe field.
    encoded, provenance = encode_uint8_pair(pair['hr'], pair['lr'], enriched,
                                           hr_mask=pair['hr_mask'], lr_mask=pair['lr_mask'])
    return _npz(encoded, provenance), provenance['storage']


def convert_baseline_shard(source_path, output_path, recipe, sample_ids, common_by_id, seed):
    """Strictly decode old float pairs and preserve their original recipe."""
    dataset = MaritimeDataset([source_path], recipe)
    if [Path(name).stem for _, name in dataset.members] != sample_ids:
        raise ValueError('Baseline shard member identities/order mismatch')
    storage = {}
    with tarfile.open(output_path, 'w') as archive:
        for index in range(len(dataset)):
            pair, metadata = dataset.read_arrays(index)
            identity = metadata['sample_id']
            payload, info = _encode(pair, metadata, common_by_id[identity], recipe, seed)
            _member(archive, identity, payload)
            storage[identity] = info
    return storage


def verify_uint8_shard(path, recipe, contexts, seed):
    """Decode all samples, reproduce selected original recipes, collate Torch."""
    dataset = MaritimeUint8Dataset([path], recipe)
    checked = set()
    storage = {}
    for index in range(len(dataset)):
        arrays, metadata = dataset.read_arrays(index)
        identity = metadata['sample_id']
        storage[identity] = metadata['storage']
        if identity in contexts:
            context = contexts[identity]
            expected = degrade(context, identity, recipe, release_seed=seed)
            # Original metadata equality proves seeded processing before encoding.
            if expected['metadata'] != metadata['recipe']:
                raise ValueError('Original float recipe recomputation mismatch')
            encoded, _ = encode_uint8_pair(np.ascontiguousarray(context[:,16:-16,16:-16]),
                expected['lr'], metadata, hr_mask=expected['hr_mask'], lr_mask=expected['lr_mask'])
            if not all(np.array_equal(encoded[key], arrays[key]) for key in ('hr', 'lr')):
                raise ValueError('Quantized recipe recomputation mismatch')
            checked.add(identity)
    if checked != set(contexts):
        raise ValueError('Missing stratified numeric recomputation')
    import torch
    from torch.utils.data import DataLoader, Subset
    batch = next(iter(DataLoader(Subset(dataset, range(min(2, len(dataset)))), batch_size=2, num_workers=0)))
    if (batch['hr'].dtype != torch.float32 or batch['lr'].dtype != torch.float32
            or not torch.isfinite(batch['hr']).all() or not torch.isfinite(batch['lr']).all()):
        raise ValueError('Real uint8 training loader smoke failed')
    return dict(verified_samples=len(dataset), numeric_recomputed_samples=len(checked),
        real_torch_loader=True, torch_version=torch.__version__,
        hr_shape=list(batch['hr'].shape), lr_shape=list(batch['lr'].shape)), storage


def release_status(samples, gates, verified_evidence, formats_verified,
                   minimum=10000000000, target=15000000000):
    counts, seen = Counter(), set()
    for sample in samples:
        if sample['sample_id'] in seen:
            raise ValueError('Duplicate release sample ID')
        seen.add(sample['sample_id'])
        info = sample['uint8_storage']
        if set(info) != set(RECIPE_VERSIONS) or len({info[name]['hr_sha256'] for name in info}) != 1:
            raise ValueError('Shared uint8 HR hash mismatch across children')
        if any(info[name].get('version') != STORAGE_VERSION for name in info):
            raise ValueError('Invalid uint8 storage version')
        if any(info[name]['rgb_payload_bytes'] != RGB_PAYLOAD_BYTES for name in info):
            raise ValueError('Invalid decoded RGB pair payload count')
        if sample['split'] not in ('train', 'validation', 'test'):
            raise ValueError('Invalid release split')
        counts[sample['split']] += 1
    totals = {split: counts[split]*RGB_PAYLOAD_BYTES for split in ('train', 'validation', 'test')}
    gaps = []
    if not formats_verified: gaps.append('training_format_verification')
    if totals['train'] < minimum: gaps.append('minimum_train_uint8_rgb_bytes')
    if not isinstance(gates, dict) or not gates: gaps.append('missing_expansion_gates')
    for key in EXPANSION_GATE_NAMES:
        if not isinstance(gates, dict) or gates.get(key) is not True:
            gaps.append('expansion_gate:'+key)
        if key not in verified_evidence:
            gaps.append('expansion_gate_evidence:'+key)
    if isinstance(gates, dict):
        for key, passed in sorted(gates.items()):
            if passed is not True: gaps.append('expansion_gate:'+key)
            if key not in verified_evidence: gaps.append('expansion_gate_evidence:'+key)
    return dict(status='READY' if not gaps else 'REVIEW_PENDING', training_format_verified=bool(formats_verified),
        readiness_gaps=sorted(set(gaps)), split_counts=dict(counts), rgb_bytes_per_child_by_split=totals,
        train_rgb_bytes_per_child=totals['train'], minimum_train_rgb_bytes=minimum,
        target_train_rgb_bytes=target, target_train_rgb_bytes_met=totals['train'] >= target,
        target_shortfall_rgb_bytes_per_child=max(0, target-totals['train']),
        rgb_accounting='decoded uint8 HR plus synthetic LR RGB; masks, archives and metadata excluded',
        rgb_payload_bytes_per_pair=RGB_PAYLOAD_BYTES, sample_count=len(samples))


def load_baseline(store, index_pin):
    from scripts.maritime_cloud_inventory import digest_bytes, json_bytes
    index = json.loads(store.get(index_pin))
    receipt = json.loads(store.get(index['release_receipt']))
    if receipt.get('state') != 'COMPLETE' or not receipt.get('training_format_verified'):
        raise ValueError('Verified complete float baseline required')
    if receipt['manifests'] != index['datasets']:
        raise ValueError('Baseline index/receipt manifest pins disagree')
    manifests = {name: json.loads(store.get(index['datasets'][name])) for name in ['mother', *RECIPE_VERSIONS]}
    mother = manifests['mother']
    if not mother.get('training_format_verified') or mother['sample_count'] != index['sample_count']:
        raise ValueError('Baseline population/format mismatch')
    samples = mother['samples']
    ids = [sample['sample_id'] for sample in samples]
    if len(set(ids)) != len(ids) or len(ids) != mother['sample_count']:
        raise ValueError('Baseline duplicate/count mismatch')
    expected_layout = [(shard['split'], shard['sample_ids']) for shard in mother['shards']]
    if set(ids) != {identity for _, group in expected_layout for identity in group} or sum(len(group) for _, group in expected_layout) != len(ids):
        raise ValueError('Baseline Mother shard membership mismatch')
    for name, manifest in manifests.items():
        if (manifest['dataset'] != name or manifest['samples'] != samples
                or manifest['configuration'] != mother['configuration']
                or [(shard['split'], shard['sample_ids']) for shard in manifest['shards']] != expected_layout
                or (name != 'mother' and manifest['recipe'] != RECIPE_VERSIONS[name])):
            raise ValueError('Baseline four-dataset structure/provenance mismatch')
        manifest['samples'] = samples  # Share the identical large population in memory.
    selection_bytes = store.get(mother['selection'])
    if digest_bytes(selection_bytes) != mother['configuration']['selection_sha256']:
        raise ValueError('Baseline frozen selection SHA mismatch')
    selection = json.loads(selection_bytes)
    row_map = {row['sample_id']: row for row in selection['selection']}
    if set(row_map) != set(ids):
        raise ValueError('Baseline source selection population mismatch')
    for sample in samples:
        row = row_map[sample['sample_id']]
        if row['source'] != sample['source_provenance']:
            raise ValueError('Baseline native source provenance mismatch')
        for key in ('split', 'source_kind', 'site_id', 'geographic_group_id', 'acquisition_id'):
            if row[key] != sample[key]: raise ValueError('Baseline sample identity mismatch')
    return dict(index=index, index_pin=index_pin, receipt=receipt, manifests=manifests,
        samples=samples, selection=selection, samples_sha256=digest_bytes(json_bytes(samples)))


def _verify_ref_metadata(client, ref):
    from scripts.maritime_cloud_inventory import parse_gs
    bucket, name = parse_gs(ref['uri'])
    blob = client.bucket(bucket).blob(name, generation=int(ref['generation']))
    blob.reload()
    if str(blob.generation) != str(ref['generation']) or int(blob.size) != ref['bytes'] or blob.crc32c != ref['crc32c']:
        raise ValueError('Immutable resumed/referenced shard metadata mismatch')


def _validate_resume(receipt, config_sha, identities, store):
    if (receipt.get('state') != 'COMPLETE' or receipt['configuration_sha256'] != config_sha
            or receipt['sample_ids'] != identities
            or [sample['sample_id'] for sample in receipt['samples']] != identities):
        raise ValueError('Immutable batch resume conflict')
    for ref in receipt['objects'].values(): _verify_ref_metadata(store.client, ref)
    for name in RECIPE_VERSIONS:
        report = receipt['verification'][name]
        if report['verified_samples'] != len(identities) or not report['real_torch_loader']:
            raise ValueError('Missing resumed batch verification')


def run(args):
    from scripts.maritime_cloud_inventory import require_cloud_vm
    instance = require_cloud_vm()  # No baseline/source access outside GCE.
    from scripts.maritime_cloud_inventory import digest_bytes, file_sha256, json_bytes
    from scripts.maritime_ultimate_release import CloudStore, process_sample, validate_selection
    from google.cloud import storage
    import torch
    torch.set_num_threads(1)
    if not 1 <= args.workers <= 8 or not 1 <= args.shard_samples <= 32:
        raise ValueError('Workers must be 1..8; shard samples 1..32')
    if args.min_train_rgb_bytes < 10000000000 or args.target_train_rgb_bytes < args.min_train_rgb_bytes:
        raise ValueError('Production minimum must be >=10GB; target must be >=minimum')
    payload = Path(args.selection).read_bytes()
    document = json.loads(payload.decode('utf-8-sig'))
    rows, original_native_target_gaps = validate_selection(document)
    client = storage.Client(project=args.project)
    store = CloudStore(client, args.output_prefix, args.scratch)
    index_pin = json.loads(Path(args.baseline_index_pin).read_text(encoding='utf-8-sig'))
    baseline_root = index_pin['uri'].rsplit('/', 1)[0]
    if args.output_prefix.rstrip('/') == baseline_root or args.output_prefix.rstrip('/').startswith(baseline_root+'/'):
        raise ValueError('A separate expansion prefix is required; baseline is immutable')
    baseline = load_baseline(store, index_pin)
    added_ids = {row['sample_id'] for row in preserve_baseline(baseline, document)}
    added = [row for row in rows if row['sample_id'] in added_ids]
    seed = baseline['manifests']['mother']['configuration']['release_seed']
    if document.get('release_seed', seed) != seed:
        raise ValueError('Expansion cannot change the baseline release seed')
    baseline_layout = baseline['manifests']['mother']['shards']
    if any(len(shard['sample_ids']) > args.shard_samples for shard in baseline_layout):
        raise ValueError('Shard sample limit must accommodate immutable baseline layout')
    evidence, verified_evidence = document.get('expansion_gate_evidence', {}), set()
    if not isinstance(evidence, dict) or set(evidence) != set(EXPANSION_GATE_NAMES):
        raise ValueError('All four expansion gate evidence documents are required')
    population_sha = digest_bytes(json_bytes(document['selection']))
    for key, ref in evidence.items():
        audit = json.loads(store.get(ref))  # Verify generation/CRC/SHA before interpretation.
        validate_expansion_gate_evidence(audit, key, population_sha256=population_sha,
            baseline_index_sha256=index_pin['sha256'], code_sha256=args.code_sha256)
        verified_evidence.add(key)
    config = dict(schema='maritime-uint8-expansion-config-v1', baseline_index=index_pin,
        baseline_receipt=baseline['index']['release_receipt'], baseline_manifests=baseline['index']['datasets'],
        baseline_selection=baseline['manifests']['mother']['selection'], baseline_samples_sha256=baseline['samples_sha256'],
        selection_sha256=digest_bytes(payload), selected_population_sha256=population_sha,
        code_sha256=args.code_sha256, image_id=args.image_id,
        worker_sha256=file_sha256(__file__), release_seed=seed, recipes=RECIPE_VERSIONS,
        storage_version=STORAGE_VERSION, quantization=QUANTIZATION,
        quantization_arithmetic='float64 on original float32 values',
        min_train_rgb_bytes=args.min_train_rgb_bytes, target_train_rgb_bytes=args.target_train_rgb_bytes,
        workers=args.workers, shard_samples=args.shard_samples,
        batch_order_version='original-parent-reuse-v1',
        expansion_gates=document.get('expansion_gates', {}), expansion_gate_evidence=evidence,
        splits=document['splits'], review_policy=document['review_policy'])
    config_sha = digest_bytes(json_bytes(config))
    store.publish('configuration.json', json_bytes(config))
    selection_ref = store.publish('frozen-selection.json', payload)
    common_by_id = {sample['sample_id']: sample for sample in baseline['samples']}
    completed, samples, decoded_seen = [], [], {}
    uint8_hr_seen, baseline_hr_duplicates = {}, []
    mother_shards = list(baseline_layout)
    def finish_batch(receipt):
        for sample in receipt['samples']:
            decoded = sample['decoded_native_core_sha256']
            if decoded in decoded_seen: raise ValueError('Duplicate decoded native core pixels')
            decoded_seen[decoded] = sample['sample_id']
            prior = register_hr_hash(uint8_hr_seen, sample,
                                     baseline=receipt['origin']=='baseline_float_conversion')
            if prior is not None:
                baseline_hr_duplicates.append(dict(sample_id=sample['sample_id'], same_hr_as=prior))
        completed.append(receipt); samples.extend(receipt['samples'])
        print(json.dumps(dict(completed_batches=len(completed), completed_samples=len(samples),
            completed_train_rgb_bytes=sum(sample['split']=='train' for sample in samples)*RGB_PAYLOAD_BYTES)), flush=True)
    for index, shard in enumerate(baseline_layout):
        ids = shard['sample_ids']
        name = f'control/batches/baseline-{index:06d}.json'
        receipt = store.existing(name)
        if receipt is not None:
            _validate_resume(receipt, config_sha, ids, store)
        else:
            objects, reports = {}, {}
            batch_samples = [dict(common_by_id[identity], uint8_storage={}) for identity in ids]
            for recipe in RECIPE_VERSIONS:
                old = baseline['manifests'][recipe]['shards'][index]['object']
                source_path = store.scratch/f'baseline-{recipe}-{index:06d}.tar'
                output_path = store.scratch/f'uint8-{recipe}-baseline-{index:06d}.tar'
                try:
                    source_path.write_bytes(store.get(old))
                    info = convert_baseline_shard(source_path, output_path, recipe, ids, common_by_id, seed)
                    report, verified = verify_uint8_shard(output_path, recipe, {}, seed)
                    if info != verified: raise ValueError('Converted serialized byte metadata mismatch')
                    for sample in batch_samples: sample['uint8_storage'][recipe] = info[sample['sample_id']]
                    reports[recipe] = report
                    objects[recipe] = store.publish(f'{recipe}/shards/baseline-{index:06d}.tar', output_path.read_bytes())
                finally:
                    source_path.unlink(missing_ok=True); output_path.unlink(missing_ok=True)
            receipt = dict(state='COMPLETE',configuration_sha256=config_sha, sample_ids=ids,
                split=shard['split'], origin='baseline_float_conversion', samples=batch_samples,
                objects=objects, verification=reports)
            store.publish(name, json_bytes(receipt))
        finish_batch(receipt)
    recompute = stratified_recompute(added)
    batches = added_batches(added, args.shard_samples)
    with ThreadPoolExecutor(max_workers=args.workers) as workers:
        for index, batch in enumerate(batches):
            ids = [row['sample_id'] for row in batch]
            name = f'control/batches/added-{index:06d}.json'
            receipt = store.existing(name)
            if receipt is not None:
                _validate_resume(receipt, config_sha, ids, store)
            else:
                processed = list(workers.map(lambda row: process_sample(store, row, seed), batch))
                paths = {name: store.scratch/f'{name}-added-{index:06d}.tar' for name in ['mother', *RECIPE_VERSIONS]}
                objects, reports = {}, {}
                contexts = {common['sample_id']:context for common,context,_,_ in processed if common['sample_id'] in recompute}
                batch_samples = []
                try:
                    with ExitStack() as stack:
                        archives = {name: stack.enter_context(tarfile.open(path,'w')) for name,path in paths.items()}
                        for common, context, shared, children in processed:
                            _member(archives['mother'], common['sample_id'], shared)
                            sample = dict(common, uint8_storage={})
                            for recipe, original in children.items():
                                with np.load(io.BytesIO(original), allow_pickle=False) as arrays:
                                    pair = {key:np.array(arrays[key],copy=True) for key in ('hr','lr','hr_mask','lr_mask')}
                                    metadata = json.loads(str(arrays['provenance']))
                                encoded, info = _encode(pair, metadata, common, recipe, seed)
                                _member(archives[recipe], common['sample_id'], encoded)
                                sample['uint8_storage'][recipe] = info
                            batch_samples.append(sample)
                    for recipe in RECIPE_VERSIONS:
                        report, info = verify_uint8_shard(paths[recipe],recipe,contexts,seed)
                        if any(info[sample['sample_id']] != sample['uint8_storage'][recipe] for sample in batch_samples):
                            raise ValueError('Added serialized byte metadata mismatch')
                        reports[recipe] = report
                    with tarfile.open(paths['mother'],'r') as archive:
                        for common, context, _, _ in processed:
                            with np.load(io.BytesIO(archive.extractfile(common['sample_id']+'.npz').read()),allow_pickle=False) as arrays:
                                if (array_hash(arrays['context']) != common['context_sha256'] or array_hash(arrays['hr']) != common['hr_sha256']
                                        or arrays['hr_mask'].shape != (512,512) or arrays['hr_mask'].dtype != np.bool_ or not arrays['hr_mask'].all()):
                                    raise ValueError('Added Mother serialized hash/support mismatch')
                    for name,path in paths.items(): objects[name] = store.publish(f'{name}/shards/added-{index:06d}.tar',path.read_bytes())
                    receipt = dict(state='COMPLETE',configuration_sha256=config_sha,sample_ids=ids,split='train',
                        origin='added_native_processing',samples=batch_samples,objects=objects,verification=reports)
                    store.publish(f'control/batches/added-{index:06d}.json',json_bytes(receipt))
                finally:
                    for path in paths.values(): path.unlink(missing_ok=True)
                del processed, contexts
            mother_shards.append(dict(split='train',sample_ids=ids,object=receipt['objects']['mother']))
            finish_batch(receipt); store.trim_cache()
    if set(common_by_id) | added_ids != {sample['sample_id'] for sample in samples}:
        raise ValueError('Expanded release population mismatch')
    # Preserve Mother objects by metadata verification, without native reads.
    for shard in baseline_layout: _verify_ref_metadata(client, shard['object'])
    state = release_status(samples, document.get('expansion_gates',{}), verified_evidence, True,
                           args.min_train_rgb_bytes,args.target_train_rgb_bytes)
    manifests = {}
    for name in ['mother',*RECIPE_VERSIONS]:
        shards = mother_shards if name=='mother' else [dict(split=receipt['split'],sample_ids=receipt['sample_ids'],
            object=receipt['objects'][name]) for receipt in completed]
        manifest = dict(schema='maritime-uint8-expansion-manifest-v1',dataset=name,configuration=config,
            baseline_index=index_pin,selection=selection_ref,samples=samples,shards=shards,**state,
            recipe=RECIPE_VERSIONS.get(name),sample_id_split_and_hr_shared_across_recipes=True,
            storage_version=STORAGE_VERSION if name!='mother' else 'unchanged-float-context-native-references',
            useful_native_rgb_bytes=sum(sample['useful_native_rgb_bytes'] for sample in samples),
            loader='src.maritime_uint8_dataset.MaritimeUint8Dataset' if name!='mother' else 'src.maritime_native_dataset.NativeMaritimeDataset')
        manifests[name]=store.publish(name+'/manifest.json',json_bytes(manifest))
        card=(f'# Maritime {name} additive release\n\n{state["status"]}; {len(samples)} samples. '
            f'Each child training split contains {state["train_rgb_bytes_per_child"]} decoded uint8 HR+LR RGB bytes. '
            'HR 512x512 at 2.5 m; synthetic LR 128x128 at 10 m. Masks, metadata and archive padding are excluded. '
            'Quantization is lossy and occurs after the unchanged original float recipes. '
            'Baseline Mother shards/native pixels, sample IDs, splits and original seeds are preserved. '
            'Mother retains observed float contexts and generation-pinned native source references. '
            'Per-sample source licenses, masks and radiometric/compression history remain in source_provenance. '
            'Use manifest splits and the versioned uint8 child loader; tensors become float32/255 at load. '
            'All new child samples were decoded, hash/grid/support checked, with a real Torch batch per shard. '
            'Added sources had native support validated; at least 5% per added split/category/source stratum '
            'were numerically recomputed. Baseline native sources were not reread. '
            'Quality-gate evidence is supplied by the selection and pinned; its substantive truth requires '
            'the documented audits. Unsupported sources require a separate adapter.\n\n'
            'Readiness gaps: '+json.dumps(state['readiness_gaps'])+'\n\n'
            f'Target shortfall: {state["target_shortfall_rgb_bytes_per_child"]} RGB bytes per child.\n')
        store.publish(name+'/DATASET.md',card.encode())
    receipt=dict(schema='maritime-uint8-expansion-receipt-v1',state='COMPLETE',configuration_sha256=config_sha,
        configuration=config,vm_instance_id=instance,manifests=manifests,**state,
        baseline_sample_count=len(common_by_id),added_sample_count=len(added),
        verification=dict(all_child_samples_decoded=True,real_torch_loader=True,
            added_uint8_hr_exact_duplicates_rejected=True,
            preserved_baseline_uint8_hr_duplicates=baseline_hr_duplicates,
            baseline_sources_reread=False,baseline_float_pairs_converted_without_recipe_changes=True,
            added_native_sources_verified=True,added_numeric_recomputed_samples_per_recipe=len(recompute),
            independent_audit=False),legacy_original_plan_target_gaps_not_used_for_uint8_readiness=original_native_target_gaps,
        expansion_gate_evidence_verified=sorted(verified_evidence))
    receipt_ref=store.publish('release-receipt.json',json_bytes(receipt))
    index=dict(schema='maritime-super-resolution-training-index-uint8-v1',datasets=manifests,
        release_receipt=receipt_ref,baseline_index=index_pin,state=state['status'],
        child_loader=dict(module='src.maritime_uint8_dataset.MaritimeUint8Dataset',recipes=list(RECIPE_VERSIONS),use_manifest_splits=True),
        mother_loader=baseline['index'].get('mother_loader'),
        shared_pair_format=dict(hr_shape=[3,512,512],lr_shape=[3,128,128],stored_dtype='uint8',
            hr_pixel_spacing_m=2.5,lr_pixel_spacing_m=10,train_tensor_dtype='float32',normalization='uint8 /255 on load'),
        release_summary=state,source_resolution_limit_m=1,
        labeled_ship_supplement=baseline['index'].get('labeled_ship_supplement'),
        baseline_interpretation=baseline['index'].get('interpretation'),
        notes='Baseline audit evidence describes baseline only; new release independent audit is not performed.')
    ref=store.publish('training-index.json',json_bytes(index))
    print(json.dumps(dict(training_index=ref,status=state['status'],train_rgb_bytes_per_child=state['train_rgb_bytes_per_child'])),flush=True)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-index-pin',required=True)
    parser.add_argument('--selection',required=True)
    parser.add_argument('--output-prefix',required=True)
    parser.add_argument('--scratch',required=True)
    parser.add_argument('--code-sha','--code-sha256',dest='code_sha256',required=True)
    parser.add_argument('--image-id',required=True)
    parser.add_argument('--project',default=GCP_PROJECT)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--shard-samples',type=int,default=32)
    parser.add_argument('--min-train-rgb-bytes',type=int,default=10000000000)
    parser.add_argument('--target-train-rgb-bytes',type=int,default=15000000000)
    return run(parser.parse_args())


if __name__=='__main__': main()

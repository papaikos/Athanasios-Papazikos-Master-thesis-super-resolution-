"""Pinned canonical RGB screening; final source replay remains mandatory."""
import argparse
import io
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.maritime_cloud_inventory import digest_bytes, json_bytes, verify_output
from scripts.maritime_earthview_quality import bounded_ordered_map
from src.maritime_uint8 import quantize_rgb


def canonical_arrays(raw):
    """Validate stored canonical crop before hashing frozen uint8 CHW bytes."""
    with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
        hr, context = arrays['hr'], arrays['hr_context']
    if (hr.shape != (3, 512, 512) or context.shape != (3, 544, 544)
            or hr.dtype != np.float32 or context.dtype != np.float32
            or not np.isfinite(context).all() or context.min() < 0 or context.max() > 1
            or not np.array_equal(hr, context[:, 16:528, 16:528])):
        raise ValueError('Finite float32 canonical context544 and exact HR512 crop in [0,1] required')
    hr = np.ascontiguousarray(hr)
    rgb = quantize_rgb(hr)
    return hr, rgb, dict(hr_float32_sha256=digest_bytes(hr.tobytes()),
                         hr_uint8_chw_sha256=digest_bytes(rgb.tobytes()))


def historical_water_fraction(source):
    bounds = source['physical_core_bounds']
    if (len(bounds) != 4 or any(isinstance(x, bool) or not isinstance(x, (int, float))
                               or not math.isfinite(x) for x in bounds)
            or not math.isclose(bounds[2]-bounds[0], 1280, abs_tol=1e-6)
            or not math.isclose(bounds[3]-bounds[1], 1280, abs_tol=1e-6)):
        raise ValueError('Physical 1280m square core required')
    area = source['marine_screening']['marine_area_m2']
    if isinstance(area, bool) or not isinstance(area, (int, float)) or not math.isfinite(area) or not 0 <= area <= 1280**2:
        raise ValueError('Finite historical marine area within the core required')
    return area / 1280**2


def cloud_input(rgb):
    if rgb.shape != (3, 512, 512) or rgb.dtype != np.uint8:
        raise ValueError('Canonical uint8 CHW RGB required')
    resized = np.asarray(Image.fromarray(np.moveaxis(rgb, 0, -1)).resize((256, 256), Image.Resampling.BILINEAR))
    low, high = float(resized.min()), float(resized.max())
    if high == low:
        return None
    return np.ascontiguousarray(((resized.astype(np.float32)-low)/(high-low)).transpose(2, 0, 1))


def prepare_candidate(ref, fetch, *, validator=None):
    """Bounded worker: fetch one standalone source JSON and canonical HR NPZ."""
    if validator is None:
        from src.maritime_earthview_source import validate_source
        validator = validate_source
    raw = fetch(ref['candidate_pin'])
    if digest_bytes(raw) != ref['candidate_pin']['sha256']:
        raise ValueError('Standalone candidate bytes SHA mismatch')
    source = json.loads(raw)
    if source.get('sample_id') != ref['sample_id'] or source.get('hr') != ref['hr_pin']:
        raise ValueError('Standalone candidate identity/HR pin mismatch')
    validator(source)
    historical_water_fraction(source)
    _, rgb, hashes = canonical_arrays(fetch(ref['hr_pin']))
    return source, ref, hashes, cloud_input(rgb)


def result_record(source, candidate_pin, hr_pin, policy_pin, hashes, cloud, nonconstant):
    if cloud is not None and (isinstance(cloud, bool) or not isinstance(cloud, (int, float)) or not math.isfinite(cloud) or not 0 <= cloud <= 1):
        raise ValueError('Finite cloud fraction or constant-image defer required')
    if nonconstant != (cloud is not None):
        raise ValueError('Constant image must defer cloud inference')
    return dict(schema='marine-operational-screening-result-v1', sample_id=source['sample_id'],
        source='earthview', source_family='earthview', candidate_sha256=candidate_pin['sha256'],
        **hashes, historical_water_fraction=historical_water_fraction(source),
        estimated_cloud_fraction=cloud, finite_rgb=True, nonconstant_rgb=nonconstant,
        original_support_verified=True,
        original_support_verification_basis='Pinned validated producer contract for original declared nodata RGB support; no independent dataset-mask certification and no original assembly replay in this ranking.',
        original_assembly_replayed=False, deferred_to_final_release=True,
        candidate_record_pin=candidate_pin, hr_pin=hr_pin, policy_pin=policy_pin,
        model_predictions_not_ground_truth=True, final_eligible=False, dataset_ready=False)


def validate_policy(policy):
    if (policy.get('source_family') != 'earthview'
            or policy.get('schema') != 'earthview-canonical-operational-screening-policy-v1'
            or policy.get('final_original_replay_required') is not True
            or not policy.get('preserved_lock_evidence_pins')):
        raise ValueError('Explicit Earth-scoped policy and preserved lock evidence required')
    model = policy['model']
    if (model.get('weight_sha256') != '0c43160542fdf8a06cab3b361bb590c6f610689729cb92a90d96cdd537f429be'
            or not model.get('vendor_file_pins')):
        raise ValueError('Frozen CloudSeg model weights and vendored-code pins required')
    for pin in model['vendor_file_pins']:
        if digest_bytes(Path(pin['path']).read_bytes()) != pin['sha256']:
            raise ValueError('Vendored model code SHA mismatch')
    if digest_bytes(Path(model['vm_weights']).read_bytes()) != model['weight_sha256']:
        raise ValueError('Model weight SHA mismatch')


def validate_resumed(record, ref, policy_pin):
    if (record.get('schema') != 'marine-operational-screening-result-v1'
            or record.get('sample_id') != ref['sample_id']
            or record.get('source') != 'earthview'
            or record.get('candidate_record_pin') != ref['candidate_pin']
            or record.get('candidate_sha256') != ref['candidate_pin']['sha256']
            or record.get('hr_pin') != ref['hr_pin'] or record.get('policy_pin') != policy_pin
            or record.get('final_eligible') is not False or record.get('dataset_ready') is not False
            or record.get('original_assembly_replayed') is not False
            or any(not re.fullmatch('[a-f0-9]{64}', record.get(key, ''))
                   for key in ('hr_float32_sha256', 'hr_uint8_chw_sha256'))):
        raise ValueError('Resumed score identity/provenance mismatch')
    water, cloud = record.get('historical_water_fraction'), record.get('estimated_cloud_fraction')
    def fraction(value):
        return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1
    if (not fraction(water) or (cloud is not None and not fraction(cloud))
            or record.get('nonconstant_rgb') is not (cloud is not None)
            or record.get('finite_rgb') is not True or record.get('original_support_verified') is not True):
        raise ValueError('Resumed measurement/support contract mismatch')


def run(args):
    from scripts.maritime_cloud_inventory import require_cloud_vm
    require_cloud_vm()
    from google.cloud import storage
    from scripts.maritime_ultimate_release import CloudStore
    client = storage.Client()
    policy_pin = json.loads(Path(args.policy_pin).read_bytes())
    policy = json.loads(verify_output(client, policy_pin))
    validate_policy(policy)
    from scripts.maritime_release_build import pinned
    for evidence_pin in policy['preserved_lock_evidence_pins']:
        verify_output(client, pinned(evidence_pin))
    # Import only after checking the frozen vendored code and weights.
    import torch
    import segmentation_models_pytorch as smp
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = smp.Unet(encoder_name='mobilenet_v2', encoder_weights=None, in_channels=3, classes=2)
        def forward(self, x):
            return self.backbone(x)
    model = Model()
    model.load_state_dict(torch.load(policy['model']['vm_weights'], map_location='cpu', weights_only=True), strict=True)
    model.eval()
    store = CloudStore(client, args.output_prefix, args.scratch)
    partition_pin = json.loads(Path(args.partition_pin).read_bytes())
    if partition_pin not in policy['candidate_partition_pins']:
        raise ValueError('Partition must be explicitly bound by frozen policy')
    partition = json.loads(verify_output(client, partition_pin))
    refs = partition['candidates']
    for ref in refs:
        pinned(ref['candidate_pin'])
        pinned(ref['hr_pin'])
    ids = [r['sample_id'] for r in refs]
    if not refs or any(not isinstance(identity, str) or not identity for identity in ids) or len(set(ids)) != len(ids):
        raise ValueError('Unique source candidates required within partition')
    result_pins = []
    def result_name(ref):
        identity = dict(candidate_pin=ref['candidate_pin'], policy_pin=policy_pin)
        return 'results/' + digest_bytes(json_bytes(identity)) + '.json'
    def unfinished():
        for ref in refs:
            name = result_name(ref)
            blob = client.bucket(store.bucket).blob(store.root+'/'+name)
            if not blob.exists():
                yield ref
                continue
            blob.reload()
            raw = blob.download_as_bytes(if_generation_match=int(blob.generation), checksum='crc32c')
            pin = dict(uri=store.prefix+'/'+name, generation=str(blob.generation), bytes=len(raw),
                       crc32c=blob.crc32c, sha256=digest_bytes(raw))
            record = json.loads(verify_output(client, pin))
            validate_resumed(record, ref, policy_pin)
            result_pins.append(dict(sample_id=ref['sample_id'], result_pin=pin))
    def prepare(ref):
        return prepare_candidate(ref, lambda pin: verify_output(client, pin))
    prepared = bounded_ordered_map(prepare, unfinished(), workers=args.workers, max_pending=args.max_pending)
    batch = []
    def flush(batch):
        nonconstant = [item for item in batch if item[3] is not None]
        predictions = {}
        if nonconstant:
            with torch.inference_mode():
                logits = model(torch.from_numpy(np.stack([item[3] for item in nonconstant])))
            if tuple(logits.shape) != (len(nonconstant), 2, 256, 256) or not torch.isfinite(logits).all():
                raise ValueError('Finite cloud model output geometry mismatch')
            fractions = (logits.argmax(1) == 1).float().mean(dim=(1, 2)).tolist()
            predictions = {item[0]['sample_id']: fraction for item, fraction in zip(nonconstant, fractions)}
        for source, ref, hashes, tensor in batch:
            record = result_record(source, ref['candidate_pin'], ref['hr_pin'], policy_pin, hashes,
                                   predictions.get(source['sample_id']), tensor is not None)
            # Stable source/policy identity makes completed inference resumable.
            name = result_name(ref)
            result_pins.append(dict(sample_id=source['sample_id'], result_pin=store.publish(name, json_bytes(record))))
        print(json.dumps(dict(completed=len(result_pins), total=len(refs))), flush=True)
    for item in prepared:
        batch.append(item)
        if len(batch) == args.batch_size:
            flush(batch)
            batch = []
    if batch:
        flush(batch)
    index = dict(schema='earthview-canonical-operational-screening-index-v1', policy_pin=policy_pin,
        candidate_partition_pin=partition_pin, count=len(result_pins), result_pins=sorted(result_pins, key=lambda row: row['sample_id']),
        screened_only=True, quality_accepted_count=0, final_eligible=False, dataset_ready=False)
    index_raw = json_bytes(index)
    print(json.dumps(store.publish('indexes/'+digest_bytes(index_raw)+'.json', index_raw)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy-pin', required=True)
    parser.add_argument('--partition-pin', required=True)
    parser.add_argument('--output-prefix', required=True)
    parser.add_argument('--scratch', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-pending', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= args.max_pending <= 8 or not 1 <= args.batch_size <= 16 or not 1 <= args.threads <= 4:
        parser.error('Bounded workers/pending <=8, batch<=16 and threads<=4 required')
    run(args)


if __name__ == '__main__':
    main()

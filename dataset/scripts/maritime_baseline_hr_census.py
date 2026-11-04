"""Hash actual decoded baseline HR after final uint8 quantization, on GCE only.

This supports preselection deduplication. It does not alter the baseline, grant
quality gates, or count an ungenerated expanded dataset as usable training data.
"""
import argparse
from collections import defaultdict
import io
import json
from pathlib import Path
import tarfile

import numpy as np

from scripts.maritime_cloud_inventory import json_bytes, require_cloud_vm
from scripts.maritime_ultimate_release import CloudStore
from src.maritime_recipes import array_hash
from src.maritime_uint8 import quantize_rgb, uint8_array_hash, QUANTIZATION


def census_shard(path,samples,expected_ids):
    rows=[];seen=set()
    with tarfile.open(path,'r') as archive:
        for member in archive:
            identity=member.name.removesuffix('.npz')
            if (not member.isfile() or member.name!=identity+'.npz'
                    or identity not in expected_ids or identity in seen):
                raise ValueError('Unexpected baseline shard membership')
            seen.add(identity);sample=samples[identity]
            with np.load(io.BytesIO(archive.extractfile(member).read()),allow_pickle=False) as arrays:
                hr=arrays['hr'];context=arrays['context'];mask=arrays['hr_mask']
                if (hr.shape!=(3,512,512) or hr.dtype!=np.float32
                        or context.shape!=(3,544,544) or context.dtype!=np.float32
                        or not np.isfinite(context).all() or (context<0).any() or (context>1).any()
                        or mask.shape!=(512,512) or mask.dtype!=np.bool_ or not mask.all()
                        or not np.array_equal(hr,context[:,16:-16,16:-16])):
                    raise ValueError('Original baseline shape/support mismatch')
                metadata=json.loads(str(arrays['provenance']))
                if (metadata['sample_id']!=identity or metadata['hr_sha256']!=sample['hr_sha256']
                        or array_hash(hr)!=sample['hr_sha256']
                        or array_hash(context)!=sample['context_sha256']):
                    raise ValueError('Original baseline HR/context hash mismatch')
                rows.append(dict(sample_id=identity,split=sample['split'],source_kind=sample['source_kind'],
                    site_id=sample['site_id'],acquisition_id=sample['acquisition_id'],
                    hr_float32_sha256=sample['hr_sha256'],
                    hr_uint8_chw_sha256=uint8_array_hash(quantize_rgb(hr))))
    if seen!=set(expected_ids):raise ValueError('Incomplete baseline shard membership')
    return rows


def main():
    instance=require_cloud_vm()
    from google.cloud import storage
    parser=argparse.ArgumentParser(description=__doc__)
    for argument in ('baseline-index-pin','output-prefix','scratch'):
        parser.add_argument('--'+argument,required=True)
    args=parser.parse_args()
    store=CloudStore(storage.Client(),args.output_prefix,args.scratch)
    index_pin=json.loads(Path(args.baseline_index_pin).read_bytes())
    index=json.loads(store.get(index_pin));receipt=json.loads(store.get(index['release_receipt']))
    if (receipt['state']!='COMPLETE' or receipt.get('training_format_verified') is not True
            or receipt['manifests']!=index['datasets']):
        raise ValueError('Verified completed baseline required')
    mother_pin=index['datasets']['mother'];mother=json.loads(store.get(mother_pin))
    samples={row['sample_id']:row for row in mother['samples']}
    if len(samples)!=mother['sample_count']:raise ValueError('Duplicate baseline population')
    rows=[]
    for number,shard in enumerate(mother['shards']):
        path=store.scratch/'mother-shard.tar'
        # Byte-pinned download is decoded only inside this GCE process.
        path.write_bytes(store.get(shard['object']))
        try:rows.extend(census_shard(path,samples,shard['sample_ids']))
        finally:path.unlink(missing_ok=True)
        print(json.dumps(dict(completed_shards=number+1,shards=len(mother['shards']),samples=len(rows))),flush=True)
    if len(rows)!=len(samples) or {r['sample_id'] for r in rows}!=set(samples):
        raise ValueError('Baseline census completeness mismatch')
    groups=defaultdict(list)
    for row in rows:groups[row['hr_uint8_chw_sha256']].append(row['sample_id'])
    document=dict(schema='maritime-baseline-quantized-hr-census-v1',complete=True,
        baseline_index_pin=index_pin,mother_manifest_pin=mother_pin,vm_instance_id=instance,
        sample_count=len(rows),rows=sorted(rows,key=lambda r:r['sample_id']),
        quantization=QUANTIZATION,quantization_arithmetic='float64 on original float32 values',
        preserved_duplicate_groups={digest:ids for digest,ids in groups.items() if len(ids)>1},
        baseline_changed=False,dataset_ready_granted=False)
    pin=store.publish('census.json',json_bytes(document))
    print(json.dumps(dict(census_pin=pin,sample_count=len(rows),duplicate_groups=len(document['preserved_duplicate_groups']))),flush=True)


if __name__=='__main__':main()

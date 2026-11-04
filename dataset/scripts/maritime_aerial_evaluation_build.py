"""Dedicated GCE aerial evaluation build; training source dispatch stays unchanged."""
import argparse
from contextlib import ExitStack
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import rasterio
from rasterio.io import MemoryFile

from scripts.maritime_aerial_evaluation import allocate_source
from scripts.maritime_aerial_evaluation_release import load_evaluation_inputs
from scripts.maritime_cloud_inventory import json_bytes,require_cloud_vm
from scripts.maritime_pilot import deterministic_npz
from scripts.maritime_release_build import pinned
from scripts.maritime_uint8_release import _encode,_member,verify_uint8_shard,added_batches
from scripts.maritime_ultimate_release import CloudStore,common_sample,pair_payload
from src.maritime_aerial_source import source_adapter
from src.maritime_recipes import RECIPE_VERSIONS,array_hash
from src.maritime_uint8 import STORAGE_VERSION


def process_evaluation_sample(store,row,seed):
    """Replay genuine original NOAA/NAIP observations with true eval metadata."""
    if (row.get('source_kind')!='aerial_rgb' or row.get('split') not in ('validation','test')
            or row.get('sample_id')!=row['source']['sample_id']):
        raise ValueError('Dedicated aerial evaluation identity/split required')
    source=row['source'];adapter=source_adapter(source);adapter.validate_source(source)
    if allocate_source(source['source'],[])['role']!='evaluation':
        raise ValueError('Frozen separate evaluation source reservation required')
    evidence=json.loads(store.get(source['source_evidence_pin']))
    if evidence.get('source')!={k:v for k,v in source['source'].items() if k!='source_evidence_pin'}:
        raise ValueError('Pinned evaluation source evidence differs from original source')
    keys=(('original_item_metadata_object','official_dataset_metadata_object')
          if source['source_type']=='noaa_aerial' else
          ('original_item_metadata_object','original_collection_metadata_object',
           'official_usda_policy_object','official_usgs_metadata_object'))
    for key in keys:store.get(pinned(evidence[key]))
    with ExitStack() as stack:
        memory=stack.enter_context(MemoryFile(store.get(source['native_context'])))
        ds=stack.enter_context(memory.open())
        context,useful,context_hash,core_hash=adapter.canonical_from_native(ds,source,store.get(source['hr']))
        original=stack.enter_context(rasterio.open(store.cached_source(source['source_original_object'])))
        adapter.verify_original_parent(original,source,ds.read([1,2,3]))
    common=common_sample(row,context,useful,context_hash,core_hash)
    common.update(delivered_gsd_m=source['delivered_gsd_m'],original_camera_capture_gsd_m=None,
        native_gsd_semantics=source['explicit_gsd_semantics'],source_platform='aerial',
        source_product=source['source_product'],original_parent_pixels_replayed=True,
        separate_auxiliary_evaluation=True,contributes_to_training_quota=False,
        native_byte_accounting='decoded original delivered aerial orthomosaic RGB owned core; camera GSD unknown')
    if source['source_type']=='naip_aerial':
        common.update(original_mask_flags=source['source']['original_raster_header']['mask_flags'],
            original_void_mask_available=source['original_void_mask_available'],
            original_support_policy=source['original_support_policy'],
            original_void_mask_caveat='All-valid flags do not independently mark voids; RGB000 guard is conservative and can reject genuinely black pixels.',
            stac_datetime=source['stac_datetime'],publication_datetime=None,
            stac_datetime_role=source['source']['stac_datetime_role'])
    shared=deterministic_npz(dict(context=context,hr=np.ascontiguousarray(context[:,16:-16,16:-16]),
        hr_mask=np.ones((512,512),bool),provenance=np.asarray(json.dumps(common,sort_keys=True))))
    children={name:pair_payload(context,common,name,seed)[0] for name in RECIPE_VERSIONS}
    return common,context,shared,children


def build_evaluation(store,document,seed,*,shard_samples=8):
    """Publish four separate formats only after full exposure and pixel checks."""
    if isinstance(shard_samples,bool) or not isinstance(shard_samples,int) or not 1<=shard_samples<=32:
        raise ValueError('Bounded evaluation shard size1 to32 required')
    rows,exposed=load_evaluation_inputs(store,document)
    config=dict(schema='aerial-evaluation-build-config-v1',release_seed=seed,selection=document,
        exposure_count=len(exposed),recipes=RECIPE_VERSIONS,shard_samples=shard_samples,batch_order_version='original-parent-reuse-v1',
        train_sample_count=0,separate_auxiliary_evaluation=True,original_full_parent_replay_required=True)
    config_pin=store.publish('control/config.json',json_bytes(config))
    samples=[];shards={name:[] for name in ['mother',*RECIPE_VERSIONS]};decoded=set()
    # Never mix true validation/test rows within one shard.
    batches=added_batches(rows,shard_samples)
    for index,batch in enumerate(batches):
        processed=[process_evaluation_sample(store,row,seed) for row in batch]
        paths={name:store.scratch/f'aerial-eval-{name}-{index:06d}.tar' for name in shards}
        try:
            with ExitStack() as stack:
                archives={name:stack.enter_context(tarfile.open(path,'w')) for name,path in paths.items()}
                batch_samples=[]
                for common,context,shared,children in processed:
                    if common['decoded_native_core_sha256'] in decoded:raise ValueError('Duplicate original evaluation core pixels')
                    decoded.add(common['decoded_native_core_sha256'])
                    _member(archives['mother'],common['sample_id'],shared)
                    sample=dict(common,uint8_storage={})
                    for recipe,payload in children.items():
                        with np.load(io.BytesIO(payload),allow_pickle=False) as arrays:
                            pair={key:np.array(arrays[key],copy=True) for key in ('hr','lr','hr_mask','lr_mask')}
                            metadata=json.loads(str(arrays['provenance']))
                        encoded,storage=_encode(pair,metadata,common,recipe,seed)
                        _member(archives[recipe],common['sample_id'],encoded);sample['uint8_storage'][recipe]=storage
                    batch_samples.append(sample)
            contexts={common['sample_id']:context for common,context,_,_ in processed}
            reports={}
            for recipe in RECIPE_VERSIONS:
                reports[recipe],info=verify_uint8_shard(paths[recipe],recipe,contexts,seed)
                if any(info[s['sample_id']]!=s['uint8_storage'][recipe] for s in batch_samples):
                    raise ValueError('Evaluation encoded byte accounting mismatch')
            with tarfile.open(paths['mother'],'r') as archive:
                for common,context,_,_ in processed:
                    with np.load(io.BytesIO(archive.extractfile(common['sample_id']+'.npz').read()),allow_pickle=False) as arrays:
                        metadata=json.loads(str(arrays['provenance']))
                        if (metadata!=common or array_hash(arrays['context'])!=common['context_sha256']
                                or array_hash(arrays['hr'])!=common['hr_sha256']
                                or arrays['hr_mask'].dtype!=np.bool_ or arrays['hr_mask'].shape!=(512,512)
                                or not arrays['hr_mask'].all()):raise ValueError('Evaluation Mother serialized provenance/support mismatch')
            ids=[s['sample_id'] for s in batch_samples];split=batch[0]['split']
            for name,path in paths.items():
                pin=store.publish(f'{name}/shards/{split}-{index:06d}.tar',path.read_bytes())
                shards[name].append(dict(split=split,sample_ids=ids,object=pin,verification=reports.get(name)))
            samples.extend(batch_samples)
        finally:
            for path in paths.values():path.unlink(missing_ok=True)
        store.trim_cache()
    manifests={}
    for name in shards:
        manifest=dict(schema='maritime-aerial-auxiliary-evaluation-manifest-v1',dataset=name,config_pin=config_pin,
            exposure_evidence_pin=document['exposure_evidence_pin'],sample_count=len(samples),samples=samples,shards=shards[name],
            train_sample_count=0,separate_auxiliary_evaluation=True,baseline_evaluation_unchanged=True,
            recipe=RECIPE_VERSIONS.get(name),storage_version=STORAGE_VERSION if name!='mother' else 'unchanged-float-context-native-references',
            loader='src.maritime_uint8_dataset.MaritimeUint8Dataset' if name!='mother' else 'src.maritime_native_dataset.NativeMaritimeDataset',
            source_originals_replayed=True,all_recipe_pairs_numerically_recomputed=True,
            evaluation_format_verified=True,training_format_verified=True)
        manifests[name]=store.publish(name+'/manifest.json',json_bytes(manifest))
    receipt=dict(schema='maritime-aerial-auxiliary-evaluation-build-receipt-v1',status='AUXILIARY_EVALUATION_FORMAT_VERIFIED',
        manifests=manifests,config_pin=config_pin,sample_count=len(samples),train_sample_count=0,
        qualification_authority='External final selection and review remain leader-owned; format status asserts pixel/source/recipe checks only.')
    store.publish('control/receipt.json',json_bytes(receipt));return receipt


def main():
    require_cloud_vm()
    from google.cloud import storage
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('selection-pin','output-prefix','scratch'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--release-seed',type=int,default=20261006);parser.add_argument('--shard-samples',type=int,default=8)
    args=parser.parse_args();store=CloudStore(storage.Client(),args.output_prefix,args.scratch)
    selection_pin=pinned(json.loads(Path(args.selection_pin).read_bytes()));document=json.loads(store.get(selection_pin))
    # A separate prefix cannot replace either the pinned selection or exposure objects.
    for pin in (selection_pin,pinned(document['exposure_evidence_pin'])):
        if pin['uri'].startswith(args.output_prefix.rstrip('/')+'/'):raise ValueError('Evaluation output must use a new separate prefix')
    print(json.dumps(build_evaluation(store,document,args.release_seed,shard_samples=args.shard_samples)))


if __name__=='__main__':main()

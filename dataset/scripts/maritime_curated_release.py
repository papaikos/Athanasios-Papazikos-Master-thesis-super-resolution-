"""GCE-only ShipRS qualified 0.3m chips: four quarantined SR supplements.

Original 930-square BMP imagery supplies a real centered 800-square/240m
context. Exact physical area yields 96-square HR context, 64-square HR core,
and three 16-square LR recipes. Native imagery and source XML remain pinned.
Missing source georeferencing forbids any geographic holdout or unique-byte
quota claims. Supplements are opt-in training data, isolated from evaluation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import re
import tarfile
from xml.etree import ElementTree

import numpy as np
from PIL import Image

from scripts.maritime_cloud_inventory import digest_bytes,file_sha256,json_bytes,require_cloud_vm
from scripts.maritime_curated_intake import pixel_identity
from scripts.maritime_planet_intake import area_matrix
from scripts.maritime_pilot import deterministic_npz
from scripts.maritime_release_build import pinned
from scripts.maritime_ultimate_release import CloudStore,tar_member
from src.maritime_curated_dataset import CuratedMaritimeDataset
from src.maritime_recipes import RECIPE_VERSIONS,array_hash,degrade
from src.cloud_config import GCP_PROJECT


def qualified_records(document):
    records=[];seen=set()
    for record in document['records']:
        if record['status']!='QUALIFIED_GSD_RGB_LABELED':continue
        if record.get('native_gsd_m')!=.3:raise ValueError('Qualified 0.3m source GSD required')
        if record.get('width')!=930 or record.get('height')!=930:raise ValueError('Documented 930-square original chips required')
        pinned(record['image_original']);pinned(record['annotation_original'])
        identity=record['decoded_pixel_sha256']
        if not re.fullmatch('[a-f0-9]{64}',identity):raise ValueError('Decoded source pixel identity required')
        if identity in seen:raise ValueError('Duplicate decoded original chip')
        seen.add(identity);records.append(record)
    if not records:raise ValueError('No qualified documented 0.3m RGB chips')
    return sorted(records,key=lambda row:row['decoded_pixel_sha256'])


def prepare_context(native,xml):
    root=ElementTree.fromstring(xml)
    try:gsd=float(root.findtext('Img_Resolution'))
    except (TypeError,ValueError):raise ValueError('Original XML source GSD unknown')
    if gsd!=.3:raise ValueError('Original XML source GSD must be 0.3m')
    dimensions=[root.findtext('size/width'),root.findtext('size/height'),root.findtext('size/depth')]
    if (dimensions!=['930','930','3'] or native.shape!=(930,930,3) or native.dtype!=np.uint8):
        raise ValueError('Original RGB/XML dimensions or uint8 pixels disagree')
    cropped=np.moveaxis(native[65:865,65:865],-1,0)
    weights=area_matrix(0,96,.3,2.5)
    if weights.shape!=(96,800):raise ValueError('Exact original contributors do not fill 240m context')
    context=np.stack([np.asarray((weights@channel.astype(np.float64))@weights.T)/255 for channel in cropped]).astype(np.float32)
    start=65+40/.3;end=65+200/.3
    geometry=dict(native_context_window=[65,65,800,800],physical_context_m=[240.,240.],
        hr_context_shape=[3,96,96],hr_core_shape=[3,64,64],lr_shape=[3,16,16],
        hr_core_source_bounds=[start,start,end,end],real_halo_m=40.,
        coordinate_system='source_image_local_axes_unknown_earth_orientation',
        georeferencing=None,north_up_verified=False)
    annotations=[]
    for ordinal,item in enumerate(root.findall('object')):
        box=item.find('bndbox')
        if box is None:raise ValueError('Official object missing source bounding box')
        try:bounds=[float(box.findtext(key)) for key in ('xmin','ymin','xmax','ymax')]
        except (TypeError,ValueError):raise ValueError('Official object source bounding box invalid')
        if not np.isfinite(bounds).all() or bounds[2]<=bounds[0] or bounds[3]<=bounds[1]:raise ValueError('Official source bounding box invalid')
        cx,cy=(bounds[0]+bounds[2])/2,(bounds[1]+bounds[3])/2
        if not start<=cx<end or not start<=cy<end:continue
        annotations.append(dict(source_object_index=ordinal,class_name=item.findtext('name'),
            source_bbox=bounds,selection_rule='official source bbox center inside HR core',
            fully_inside_hr_core=min(bounds[:2])>=start and max(bounds[2:])<=end,
            original_object_xml=ElementTree.tostring(item,encoding='unicode'),
            coordinate_basis='unchanged upstream source image pixel coordinates; convention not reinterpreted'))
    return context,geometry,annotations


def sample_metadata(record,context,geometry,annotations):
    identity='shiprs-'+record['decoded_pixel_sha256']
    return dict(sample_id=identity,split='train',quarantined_training_supplement=True,
        geographic_holdout_isolation_verified=False,georeferencing=None,
        coordinate_system=geometry['coordinate_system'],native_gsd_m=.3,hr_spacing_m=2.5,lr_spacing_m=10,
        geometry=geometry,source_provenance=record,official_core_annotations=annotations,
        annotated_core_object_count=len(annotations),annotations_exhaustive_absence_not_inferred=True,
        hr_sha256=array_hash(context[:,16:-16,16:-16]),context_sha256=array_hash(context),
        native_context_ref=dict(object=record['image_original'],pixel_window=geometry['native_context_window']),
        unique_native_quota_contribution_bytes=0,
        quarantine_reason='Original constituent locations/overlapping chips unavailable; main benchmark geographic isolation cannot be guaranteed')


def pair_payload(common,context,recipe,seed):
    pair=degrade(context,common['sample_id'],recipe,release_seed=seed)
    metadata=dict(common,recipe=pair['metadata'],lr_origin='synthetic_hr_only')
    return deterministic_npz(dict(hr=np.ascontiguousarray(context[:,16:-16,16:-16]),lr=pair['lr'],
        hr_mask=pair['hr_mask'],lr_mask=pair['lr_mask'],provenance=np.asarray(json.dumps(metadata,sort_keys=True))))


def verify_shard(path,recipe,contexts,seed):
    dataset=CuratedMaritimeDataset([path],recipe);count=0
    for index in range(len(dataset)):
        arrays,meta=dataset.read_arrays(index);identity=meta['sample_id']
        if identity in contexts:
            expected=degrade(contexts[identity],identity,recipe,release_seed=seed)
            if (expected['metadata']!=meta['recipe'] or not np.array_equal(expected['lr'],arrays['lr'])
                or not np.array_equal(contexts[identity][:,16:-16,16:-16],arrays['hr'])):
                raise ValueError('Curated serialized deterministic recomputation mismatch')
            count+=1
    import torch
    from torch.utils.data import DataLoader,Subset
    batch=next(iter(DataLoader(Subset(dataset,range(min(2,len(dataset)))),batch_size=2,num_workers=0)))
    if not torch.isfinite(batch['hr'].mean()) or not torch.isfinite(batch['lr'].mean()):raise ValueError('Curated Torch loader smoke failed')
    return dict(verified_samples=len(dataset),numeric_recomputed_samples=count,torch_version=torch.__version__,
        loader='torch.utils.data.DataLoader',hr_shape=list(batch['hr'].shape),lr_shape=list(batch['lr'].shape))


def prepare_sample(store,record,seed):
    image=store.get(record['image_original']);xml=store.get(record['annotation_original'])
    if pixel_identity(image)!=record['decoded_pixel_sha256']:raise ValueError('Decoded original RGB chip identity changed')
    with Image.open(io.BytesIO(image)) as picture:
        if picture.mode!='RGB':raise ValueError('Original RGB required without colour conversion')
        native=np.array(picture)
    context,geometry,annotations=prepare_context(native,xml)
    common=sample_metadata(record,context,geometry,annotations)
    shared=deterministic_npz(dict(context=context,hr=np.ascontiguousarray(context[:,16:-16,16:-16]),
        hr_mask=np.ones((64,64),bool),provenance=np.asarray(json.dumps(common,sort_keys=True))))
    children={recipe:pair_payload(common,context,recipe,seed) for recipe in RECIPE_VERSIONS}
    return common,context,shared,children


def run(args):
    instance=require_cloud_vm()
    import torch
    torch.set_num_threads(1)
    from google.cloud import storage
    payload=Path(args.manifest).read_bytes();document=json.loads(payload)
    records=qualified_records(document)
    if args.max_samples is not None:records=records[:args.max_samples]
    if not records:raise ValueError('Positive curated sample limit required')
    store=CloudStore(storage.Client(project=GCP_PROJECT),args.output_prefix,args.scratch)
    config=dict(schema='shiprs-curated-sr-supplement-v1',manifest_sha256=digest_bytes(payload),
        worker_sha256=file_sha256(__file__),image_id=args.image_id,code_sha256=args.code_sha256,
        release_seed=args.release_seed,shard_samples=args.shard_samples,maximum_samples=args.max_samples,
        recipes=RECIPE_VERSIONS,split='train',quarantined=True,unique_native_quota_contribution_bytes=0,
        license=document['license'],license_evidence_url=document['license_evidence_url'])
    config_sha=digest_bytes(json_bytes(config))
    store.publish('configuration.json',json_bytes(config));input_ref=store.publish('qualified-source-manifest.json',payload)
    samples=[];receipts=[]
    with ThreadPoolExecutor(max_workers=2) as workers:
        for ordinal,start in enumerate(range(0,len(records),args.shard_samples)):
            batch=records[start:start+args.shard_samples];name=f'control/batches/{ordinal:06d}.json'
            receipt=store.existing(name)
            identities=['shiprs-'+record['decoded_pixel_sha256'] for record in batch]
            if receipt is not None:
                if receipt['configuration_sha256']!=config_sha or receipt['sample_ids']!=identities:raise ValueError('Curated immutable batch resume conflict')
                for ref in receipt['objects'].values():store.get(ref)
            else:
                prepared=list(workers.map(lambda record:prepare_sample(store,record,args.release_seed),batch))
                paths={dataset:store.scratch/f'{dataset}-{ordinal:06d}.tar' for dataset in ['mother',*RECIPE_VERSIONS]}
                contexts={sample[0]['sample_id']:sample[1] for sample in prepared};reports={};objects={}
                try:
                    from contextlib import ExitStack
                    with ExitStack() as stack:
                        archives={dataset:stack.enter_context(tarfile.open(path,'w')) for dataset,path in paths.items()}
                        for common,context,shared,children in prepared:
                            tar_member(archives['mother'],common['sample_id']+'.npz',shared)
                            for recipe,pair in children.items():tar_member(archives[recipe],common['sample_id']+'.npz',pair)
                    for recipe in RECIPE_VERSIONS:reports[recipe]=verify_shard(paths[recipe],recipe,contexts,args.release_seed)
                    with tarfile.open(paths['mother'],'r') as archive:
                        for common,context,_,_ in prepared:
                            with np.load(io.BytesIO(archive.extractfile(common['sample_id']+'.npz').read()),allow_pickle=False) as arrays:
                                if array_hash(arrays['context'])!=common['context_sha256'] or array_hash(arrays['hr'])!=common['hr_sha256']:raise ValueError('Curated shared HR serialization mismatch')
                    for dataset,path in paths.items():objects[dataset]=store.publish(f'{dataset}/shards/{ordinal:06d}.tar',path.read_bytes())
                    receipt=dict(state='COMPLETE',configuration_sha256=config_sha,sample_ids=identities,
                        samples=[sample[0] for sample in prepared],objects=objects,verification=reports)
                    store.publish(name,json_bytes(receipt))
                finally:
                    for path in paths.values():path.unlink(missing_ok=True)
                del prepared,contexts
            samples.extend(receipt['samples']);receipts.append(receipt)
            print(json.dumps(dict(completed_batches=len(receipts),samples=len(samples))),flush=True)
    manifests={}
    for dataset in ['mother',*RECIPE_VERSIONS]:
        result=dict(schema='shiprs-curated-sr-supplement-v1',dataset=dataset,status='TRAINING_FORMAT_VERIFIED_SUPPLEMENT',
            training_format_verified=True,full_60gb_target_passed=False,unique_native_quota_contribution_bytes=0,
            split='train',quarantined_training_supplement=True,geographic_holdout_isolation_verified=False,
            sample_count=len(samples),samples=samples,configuration=config,source_manifest=input_ref,
            shards=[dict(object=receipt['objects'][dataset],sample_ids=receipt['sample_ids'],split='train') for receipt in receipts])
        manifests[dataset]=store.publish(dataset+'/manifest.json',json_bytes(result))
        card=(f'# ShipRS {dataset} curated super-resolution supplement\n\n'
            f'{len(samples)} original chips documented at 0.3 m. Exact-area 240 m context '
            'provides shared 96x96 HR context, 64x64 HR core at 2.5 m, and 16x16 '
            'synthetic LR at 10 m. Original 800x800 source pixel context is centered '
            'inside each 930x930 image; real halo is 40 m per side. Original BMP and '
            'official XML remain generation/SHA/checksum pinned. Official annotation '
            'bbox centers inside the HR core select labels, retained unchanged in '
            'source pixel coordinates with original object XML.\n\n'
            'All samples are quarantined, opt-in training supplements. Original '
            'train/validation/test labels are provenance only. Earth georeferencing '
            'and original scene locations are unknown, and constituent chips may '
            'overlap. Exclude this supplement from claims of clean geographic '
            'benchmark evaluation: overlap with the main geographic holdouts cannot '
            'be ruled out. Unique native 60 GB quota contribution is zero.\n\n'
            'Download pinned child TAR shards and load with '
            'src.maritime_curated_dataset.CuratedMaritimeDataset(shards, recipe). '
            'Keep 64/16 batches separate from the main 512/128 branch. Every stored '
            'child sample is reopened, hash/shape/mask checked and numerically '
            'recomputed; real PyTorch collation is exercised per shard.\n\n'
            'License: '+str(document['license'])+'. Constituent source terms apply. '
            'Resolution evidence comes from official per-image annotations, not '
            'independent sensor calibration.\n')
        store.publish(dataset+'/DATASET.md',card.encode())
    receipt=dict(schema='shiprs-curated-sr-supplement-v1',state='COMPLETE',
        status='TRAINING_FORMAT_VERIFIED_SUPPLEMENT',vm_instance_id=instance,configuration=config,
        configuration_sha256=config_sha,manifests=manifests,sample_count=len(samples),
        unique_native_quota_contribution_bytes=0,geographic_holdout_isolation_verified=False,
        numeric_recomputed_samples_per_recipe=len(samples),independent_audit=False)
    ref=store.publish('release-receipt.json',json_bytes(receipt))
    print(json.dumps(dict(receipt=ref,samples=len(samples))),flush=True)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True);parser.add_argument('--output-prefix',required=True)
    parser.add_argument('--scratch',required=True);parser.add_argument('--image-id',required=True)
    parser.add_argument('--code-sha256',required=True);parser.add_argument('--release-seed',type=int,default=20261006)
    parser.add_argument('--shard-samples',type=int,default=32);parser.add_argument('--max-samples',type=int)
    args=parser.parse_args()
    if not 1<=args.shard_samples<=32:parser.error('shard-samples must be1..32')
    if args.max_samples is not None and args.max_samples<1:parser.error('max-samples must be positive')
    if not re.fullmatch('[a-f0-9]{64}',args.code_sha256):parser.error('code-sha256 must be SHA256')
    run(args)


if __name__=='__main__':main()

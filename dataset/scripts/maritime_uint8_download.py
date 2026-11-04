"""Byte-only SDK downloader for a generation-pinned uint8 release index.

Completed files are checksum-verified and reused. Interrupted objects restart
atomically; this is file-level resume, not byte-range continuation. Native
Mother objects are copied verbatim, including uint16 TOA products.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil
import threading
from src.cloud_config import GCP_PROJECT


FOLDERS = dict(mother='Mother', d1='D1', d2='D2', d3='D3')
SPLITS = ('train', 'validation', 'test')


def validate_pin(reference):
    try:
        uri=reference['uri']; generation=int(reference['generation'])
        if not uri.startswith('gs://') or '/' not in uri[5:] or generation < 1:
            raise ValueError('Invalid GCS object identity')
        if not isinstance(reference['bytes'],int) or isinstance(reference['bytes'],bool) or reference['bytes'] < 1:
            raise ValueError('Invalid pinned object size')
        digest=reference['sha256']
        if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid pinned SHA-256')
        if len(base64.b64decode(reference['crc32c'],validate=True))!=4:
            raise ValueError('Invalid pinned CRC32C')
    except (KeyError,TypeError) as error:
        raise ValueError('Complete immutable object pin required') from error
    return reference


def _checksums(path):
    import google_crc32c
    sha,crc=hashlib.sha256(),google_crc32c.Checksum()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            sha.update(chunk);crc.update(chunk)
    return sha.hexdigest(),base64.b64encode(crc.digest()).decode()


def _verified(path,reference):
    if path.stat().st_size!=reference['bytes']: return False
    sha,crc=_checksums(path)
    return sha==reference['sha256'] and crc==reference['crc32c']


def fetch(client,reference,target):
    """Download exact generation/CRC/SHA bytes; preserve conflicting local files."""
    validate_pin(reference)
    target=Path(target)
    if target.exists():
        if not target.is_file() or not _verified(target,reference):
            raise ValueError('Existing destination failed size/SHA/CRC verification: '+str(target))
        return target
    bucket,name=reference['uri'][5:].split('/',1)
    generation=int(reference['generation'])
    blob=client.bucket(bucket).blob(name,generation=generation)
    blob.reload()
    if str(blob.generation)!=str(generation) or int(blob.size)!=reference['bytes'] or blob.crc32c!=reference['crc32c']:
        raise ValueError('Cloud metadata differs from immutable object pin')
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix(target.suffix+'.partial')
    try:
        blob.download_to_filename(str(temporary),if_generation_match=generation,checksum='crc32c')
        if not _verified(temporary,reference): raise ValueError('Downloaded size/SHA/CRC mismatch')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def validate_manifest(manifest,name):
    from src.maritime_recipes import RECIPE_VERSIONS
    from src.maritime_uint8 import STORAGE_VERSION
    if (manifest.get('dataset')!=name or manifest.get('training_format_verified') is not True
            or manifest.get('sample_count')!=len(manifest.get('samples',[]))):
        raise ValueError('Verified consistent dataset manifest required')
    if name!='mother' and (manifest.get('storage_version')!=STORAGE_VERSION or manifest.get('recipe')!=RECIPE_VERSIONS[name]):
        raise ValueError('Expected versioned uint8 child recipe')
    assignments={sample['sample_id']:sample['split'] for sample in manifest['samples']}
    if len(assignments)!=manifest['sample_count'] or not set(assignments.values())<=set(SPLITS):
        raise ValueError('Duplicate sample IDs or invalid split assignment')
    members={}
    for shard in manifest['shards']:
        if shard['split'] not in SPLITS: raise ValueError('Invalid shard split')
        validate_pin(shard['object'])
        for identity in shard['sample_ids']:
            if identity in members: raise ValueError('Duplicate shard sample ID')
            members[identity]=shard['split']
    if members!=assignments: raise ValueError('Manifest shard membership/split mismatch')
    return assignments


def native_references(sample):
    """Follow explicit native/mask references without interpreting pixel dtype."""
    window=sample.get('native_window_ref')
    if window:
        refs=[window['object']]
        mask=window.get('mask',{}).get('object')
    elif sample.get('native_context'):
        refs=[sample['native_context']]
        mask=sample.get('source_provenance',{}).get('mask',{}).get('object')
    else:
        raise ValueError('Missing explicit Mother native object reference')
    if mask: refs.append(mask)
    return [validate_pin(reference) for reference in refs]


def build_jobs(manifests,recipes=('mother','d1','d2','d3'),splits=SPLITS,include_native=False):
    if not recipes or not set(recipes)<=set(FOLDERS) or not splits or not set(splits)<=set(SPLITS):
        raise ValueError('Supported datasets and splits required')
    jobs,expected={},None
    def add(reference,relative):
        previous=jobs.get(relative)
        if previous and any(previous[key]!=reference[key] for key in ('bytes','crc32c','sha256')):
            raise ValueError('Conflicting content-addressed destination')
        jobs.setdefault(relative,reference)
    for name in recipes:
        manifest=manifests[name]
        assignments=validate_manifest(manifest,name)
        if expected is None: expected=assignments
        elif assignments!=expected: raise ValueError('Dataset sample IDs or splits disagree')
        for shard in manifest['shards']:
            if shard['split'] in splits:
                reference=shard['object']
                add(reference,Path(FOLDERS[name])/shard['split']/(reference['sha256']+'.tar'))
    if include_native:
        if 'mother' not in manifests: raise ValueError('Mother manifest required for native objects')
        validate_manifest(manifests['mother'],'mother')
        for sample in manifests['mother']['samples']:
            if sample['split'] not in splits: continue
            for reference in native_references(sample):
                suffix=Path(reference['uri']).suffix.lower()
                if suffix=='.tiff': suffix='.tif'
                if suffix=='.jpeg': suffix='.jpg'
                if suffix not in ('.tif','.tiff','.png','.jpg','.jpeg','.jp2','.zip','.npz'): suffix='.bin'
                add(reference,Path('Mother')/'native'/'objects'/(reference['sha256']+suffix))
    return [(reference,relative) for relative,reference in sorted(jobs.items(),key=lambda item:str(item[0]))]


def build_auxiliary_jobs(manifests,quality_audit,*,name,recipes=('mother','d1','d2','d3'),
                         splits=SPLITS,include_native=False):
    """Place completely reviewed aerial evaluation outside the main split folders."""
    if not isinstance(name,str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]*',name):
        raise ValueError('Safe auxiliary evaluation name required')
    assignments=None
    for dataset,manifest in manifests.items():
        current=validate_manifest(manifest,dataset)
        if (manifest.get('schema')!='maritime-aerial-auxiliary-evaluation-manifest-v1'
                or manifest.get('train_sample_count')!=0
                or manifest.get('evaluation_format_verified') is not True
                or manifest.get('separate_auxiliary_evaluation') is not True
                or not current or not set(current.values())<=set(('validation','test'))):
            raise ValueError('Separate verified zero-training auxiliary evaluation required')
        if assignments is None: assignments=current
        elif current!=assignments: raise ValueError('Auxiliary datasets disagree on IDs or splits')
    if (quality_audit.get('schema')!='marine-selected-addition-visual-coverage-v1'
            or quality_audit.get('passed') is not True or quality_audit.get('gaps')
            or quality_audit.get('required_review_fraction')!=1
            or set(quality_audit.get('reviewed_added_ids',[]))!=set(assignments or {})):
        raise ValueError('Exact complete auxiliary actual-image review required')
    wanted=[split for split in splits if split in ('validation','test')]
    if not wanted:return []
    jobs=build_jobs(manifests,recipes,wanted,include_native)
    return [(reference,Path(relative.parts[0])/'evaluation'/name/Path(*relative.parts[1:]))
            for reference,relative in jobs]


def sdk_client(project):
    from google.auth.exceptions import DefaultCredentialsError
    from google.cloud import storage
    try: return storage.Client(project=project)
    except DefaultCredentialsError:
        from scripts.maritime_build_control import client_from_gcloud
        return client_from_gcloud()


def main(argv=None,client_factory=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index-pin',required=True,help='JSON containing complete generation/CRC/SHA index pin')
    parser.add_argument('--output',required=True,help='A separate versioned destination, preserving old downloads')
    parser.add_argument('--datasets',nargs='+',choices=list(FOLDERS),default=list(FOLDERS))
    parser.add_argument('--splits',nargs='+',choices=list(SPLITS),default=list(SPLITS))
    parser.add_argument('--include-mother-native',action='store_true')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--project',default=GCP_PROJECT)
    parser.add_argument('--plan-only',action='store_true',help='Verify metadata and list bytes; download no image shards')
    parser.add_argument('--allow-format-only',action='store_true',help='Explicit engineering use of a format-verified release pending quality/size gates')
    args=parser.parse_args(argv)
    if not 1<=args.workers<=32: parser.error('Workers must be 1..32')
    factory=client_factory or (lambda:sdk_client(args.project))
    client=factory();root=Path(args.output).expanduser().resolve()
    root.mkdir(parents=True,exist_ok=True);control=root/'control'
    pin=json.loads(Path(args.index_pin).read_text(encoding='utf-8-sig'))
    index=json.loads(fetch(client,pin,control/'training-index.json').read_text(encoding='utf-8'))
    if index.get('schema')!='maritime-super-resolution-training-index-uint8-v1' or set(index.get('datasets',{}))!=set(FOLDERS):
        raise ValueError('Complete versioned uint8 training index required')
    if index.get('state')!='READY' and not args.allow_format_only:
        raise ValueError('Release is pending readiness gates; engineering format-only use must be explicit')
    receipt=json.loads(fetch(client,index['release_receipt'],control/'release-receipt.json').read_text(encoding='utf-8'))
    if receipt.get('state')!='COMPLETE' or receipt.get('training_format_verified') is not True or receipt['manifests']!=index['datasets']:
        raise ValueError('Index and verified completed release receipt disagree')
    selected=list(dict.fromkeys(args.datasets))
    if args.include_mother_native and 'mother' not in selected: selected.append('mother')
    manifests={}
    for name in selected:
        path=fetch(client,index['datasets'][name],root/FOLDERS[name]/'manifest.json')
        manifests[name]=json.loads(path.read_text(encoding='utf-8'))
    jobs=build_jobs(manifests,args.datasets,args.splits,args.include_mother_native)
    auxiliary_info={}
    for name,references in sorted(index.get('auxiliary_evaluations',{}).items()):
        # Validate names before constructing any local metadata destination.
        if not isinstance(name,str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]*',name):
            raise ValueError('Safe auxiliary evaluation name required')
        auxiliary_receipt=json.loads(fetch(client,references['receipt'],
            control/'evaluation'/name/'receipt.json').read_bytes())
        if (auxiliary_receipt.get('schema')!='maritime-aerial-auxiliary-evaluation-build-receipt-v1'
                or auxiliary_receipt.get('status')!='AUXILIARY_EVALUATION_FORMAT_VERIFIED'
                or auxiliary_receipt.get('train_sample_count')!=0
                or set(auxiliary_receipt.get('manifests',{}))!=set(FOLDERS)):
            raise ValueError('Complete verified separate auxiliary receipt required')
        quality=json.loads(fetch(client,references['quality_audit'],
            control/'evaluation'/name/'quality-audit.json').read_bytes())
        auxiliary={dataset:json.loads(fetch(client,auxiliary_receipt['manifests'][dataset],
            root/FOLDERS[dataset]/'evaluation'/name/'manifest.json').read_bytes())
            for dataset in selected}
        jobs.extend(build_auxiliary_jobs(auxiliary,quality,name=name,recipes=args.datasets,
            splits=args.splits,include_native=args.include_mother_native))
        if any(m['sample_count']!=auxiliary_receipt['sample_count'] for m in auxiliary.values()):
            raise ValueError('Auxiliary receipt and manifest population sizes disagree')
        auxiliary_info[name]=dict(receipt=references['receipt'],quality_audit=references['quality_audit'],
            sample_count=auxiliary_receipt['sample_count'],train_sample_count=0,
            directory='Mother/D1/D2/D3 evaluation/'+name)
    for _,relative in jobs:
        if not (root/relative).resolve().is_relative_to(root): raise ValueError('Destination escaped versioned dataset root')
    total=sum(reference['bytes'] for reference,_ in jobs)
    missing=sum(reference['bytes'] for reference,relative in jobs if not (root/relative).exists())
    plan=dict(index_pin=pin,datasets=args.datasets,splits=args.splits,include_mother_native=args.include_mother_native,
        auxiliary_evaluations=auxiliary_info,
        objects=len(jobs),download_bytes=total,missing_bytes=missing,output=str(root),
        native_handling='opaque original bytes; dtype and display mapping retained in manifest source provenance')
    print(json.dumps(plan),flush=True)
    if args.plan_only: return plan
    if shutil.disk_usage(root).free<missing+1000000000: raise ValueError('Insufficient destination disk space')
    state=threading.local()
    def download(job):
        if not hasattr(state,'client'): state.client=factory()
        reference,relative=job
        return fetch(state.client,reference,root/relative)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index,_ in enumerate(pool.map(download,jobs),start=1):
            if index==1 or index%25==0 or index==len(jobs): print(json.dumps(dict(verified_objects=index,total_objects=len(jobs))),flush=True)
    control.mkdir(parents=True,exist_ok=True)
    completed=dict(plan,state='COMPLETE',all_selected_objects_verified=True)
    temporary=control/'download-receipt.json.partial'
    temporary.write_text(json.dumps(completed,indent=2),encoding='utf-8')
    temporary.replace(control/'download-receipt.json')
    return completed


if __name__=='__main__': main()

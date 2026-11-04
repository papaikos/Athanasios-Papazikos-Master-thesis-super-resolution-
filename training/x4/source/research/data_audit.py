"""Pinned D1/D2/D3 metadata audit with optional opaque shard verification; sealed test payloads stay closed."""
from __future__ import annotations
import argparse
import base64
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from .experiment import KEY, NAME, RECIPE

INDEX_PIN = {'bytes':8075,'crc32c':'+K9b5w==','generation':'1791488399208942',
             'sha256':'0888d9772668b6cfe3b36f952dbffdaede4f887bf59dfff561b5b8903829c6bc'}
MAIN_COUNTS = {'train':17952,'validation':115,'test':127}
AUX_COUNTS = {'validation':15,'test':48}
GROUP_FIELDS = ('sample_id','site_id','geographic_group_id','acquisition_id','source_scene_id',
                'hr_sha256','uint8_hr_sha256','decoded_native_core_sha256')


def file_identity(path):
    """Read opaque bytes only; CRC32C uses the existing staging dependency."""
    import google_crc32c
    sha = hashlib.sha256(); crc = google_crc32c.Checksum(); size = 0
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            size += len(chunk); sha.update(chunk); crc.update(chunk)
    return {'bytes':size,'sha256':sha.hexdigest(),'crc32c':base64.b64encode(crc.digest()).decode()}


def verify_file(path, pin):
    actual=file_identity(path)
    if any(actual[k]!=pin[k] for k in ('bytes','sha256','crc32c')):
        raise ValueError('Pinned file identity mismatch: '+str(path))
    return actual


class _JSONStream:
    def __init__(self,path,chunk_size):
        self.file=Path(path).open(encoding='utf-8-sig'); self.buffer='';self.pos=0
        self.chunk_size=chunk_size;self.decoder=json.JSONDecoder()
    def fill(self):
        text=self.file.read(self.chunk_size)
        self.buffer=self.buffer[self.pos:]+text;self.pos=0
        return bool(text)
    def peek(self):
        while True:
            while self.pos<len(self.buffer) and self.buffer[self.pos].isspace():self.pos+=1
            if self.pos<len(self.buffer):return self.buffer[self.pos]
            if not self.fill():raise ValueError('Truncated manifest JSON')
    def token(self,token):
        if self.peek()!=token:raise ValueError('Malformed manifest JSON: expected '+token)
        self.pos+=1
    def value(self):
        self.peek()
        while True:
            try:
                value,end=self.decoder.raw_decode(self.buffer,self.pos)
                # A scalar number may continue into the next chunk.
                if end==len(self.buffer):
                    if self.fill():continue
                    self.pos=len(self.buffer);return value
                self.pos=end;return value
            except json.JSONDecodeError:
                if not self.fill():raise ValueError('Truncated or invalid manifest value')


def manifest_items(path,chunk_size=1024*1024):
    """Yield top-level fields; samples/shards yield one object at a time."""
    stream=_JSONStream(path,chunk_size)
    try:
        stream.token('{')
        first=True
        while stream.peek()!='}':
            if not first:stream.token(',')
            key=stream.value();stream.token(':');first=False
            if key in ('samples','shards'):
                stream.token('['); first_item=True
                while stream.peek()!=']':
                    if not first_item:stream.token(',')
                    yield key,stream.value();first_item=False
                stream.token(']')
            else:yield key,stream.value()
        stream.token('}')
        if stream.buffer[stream.pos:].strip() or stream.file.read().strip():
            raise ValueError('Trailing manifest JSON data')
    finally:stream.file.close()


def compact_sample(sample,scope):
    provenance=sample.get('source_provenance') or {};source=provenance.get('source') or {}
    row={k:sample.get(k) for k in GROUP_FIELDS}
    row.update(split=sample.get('split'),scope=scope,source_kind=sample.get('source_kind'),
               crs=sample.get('crs'),core_bounds=sample.get('core_bounds'),
               source_scene_id=source.get('source_id') or source.get('item_id') or provenance.get('source_scene_id') or provenance.get('asset_id')
                               or [item['item']['id'] for item in provenance.get('source_items',[]) if item.get('item',{}).get('id')] or None,
               uint8_hr_sha256=(sample.get('uint8_storage') or {}).get(KEY,{}).get('hr_sha256'))
    return row


def audit_rows(rows,expected_counts=None):
    counts=dict(Counter(r.get('split') for r in rows));failures=[];leakage=[]
    if expected_counts is not None and counts!={k:v for k,v in expected_counts.items() if v}:
        failures.append({'kind':'split_counts','expected':expected_counts,'actual':counts})
    unknown={k:sum(not r.get(k) for r in rows) for k in GROUP_FIELDS+('crs','core_bounds','source_kind')}
    # A single repeated group within a split is expected; crossing a split is leakage.
    for field in GROUP_FIELDS:
        groups=defaultdict(list)
        for r in rows:
            if r.get(field):
                values=r[field] if isinstance(r[field],list) else [r[field]]
                for value in values:groups[str(value)].append(r)
        for value,group in groups.items():
            splits=sorted({r['split'] for r in group})
            if len(splits)>1:
                leakage.append({'field':field,'value':value,'splits':splits,
                                'sample_ids':sorted({r['sample_id'] for r in group})})
    # Only compare actual projected core rectangles with the identical recorded CRS.
    # Cross-CRS geography remains explicitly unverified rather than numerically guessed.
    footprints={'same_crs_pairs_compared':0,'cross_crs_pairs_not_compared':0,'missing_pairs_not_compared':0,
                'overlaps':[],'policy':'strict positive-area core overlap; touching edges allowed; no buffer or reprojection'}
    held=[r for r in rows if r.get('split')!='train'];training=[r for r in rows if r.get('split')=='train']
    pairs=((a,b) for a in training for b in held)
    # Validation/test separation is audited too.
    pairs2=((a,b) for i,a in enumerate(held) for b in held[i+1:] if a.get('split')!=b.get('split'))
    import itertools
    for a,b in itertools.chain(pairs,pairs2):
        if not a.get('crs') or not b.get('crs') or not a.get('core_bounds') or not b.get('core_bounds'):
            footprints['missing_pairs_not_compared']+=1;continue
        if a['crs']!=b['crs']:
            footprints['cross_crs_pairs_not_compared']+=1;continue
        footprints['same_crs_pairs_compared']+=1
        x,y=a['core_bounds'],b['core_bounds']
        if min(x[2],y[2])>max(x[0],y[0]) and min(x[3],y[3])>max(x[1],y[1]):
            footprints['overlaps'].append({'sample_ids':[a['sample_id'],b['sample_id']], 'splits':[a['split'],b['split']], 'crs':a['crs']})
    if leakage:failures.append({'kind':'cross_split_metadata_leakage','count':len(leakage)})
    if footprints['overlaps']:failures.append({'kind':'cross_split_core_overlap','count':len(footprints['overlaps'])})
    return {'status':'FAIL' if failures else 'PASS','counts':counts,'failures':failures,
            'unknown_counts':unknown,'leakage':leakage,'footprints':footprints,
            'source_counts_by_split':{split:dict(Counter(r.get('source_kind') or 'unknown' for r in rows if r['split']==split)) for split in counts}}


def read_manifest(path,scope):
    rows=[];shards=[];metadata={}
    for key,value in manifest_items(path):
        if key=='samples':rows.append(compact_sample(value,scope))
        elif key=='shards':shards.append(value)
        elif key!='configuration':metadata[key]=value
        else:
            metadata['release_seed']=value.get('release_seed');metadata['recipe_versions']=value.get('recipes')
    return rows,shards,metadata


def membership(rows,shards,base):
    assignments={r['sample_id']:r for r in rows}
    if len(assignments)!=len(rows):raise ValueError('Duplicate sample identity within manifest')
    found=set();selected=defaultdict(list)
    for shard in shards:
        split=shard['split'];pin=shard['object']
        path=(Path(base)/split/(pin['sha256']+'.tar')).as_posix()
        for identity in shard['sample_ids']:
            if identity not in assignments or identity in found:raise ValueError('Shard membership not bijective')
            found.add(identity);row=assignments[identity]
            if row['split']!=split:raise ValueError('Shard/sample split disagreement')
            if split=='test':continue
            selected[split].append(dict(row,shard=path,member=identity+'.npz'))
    if found!=set(assignments):raise ValueError('Manifest identities missing from shards')
    return dict(selected)


def validate_metadata(metadata, *, auxiliary):
    if metadata.get('dataset')!=KEY or metadata.get('recipe')!=RECIPE:
        raise ValueError(NAME+' dataset and recipe required')
    if auxiliary:
        if metadata.get('train_sample_count')!=0 or metadata.get('separate_auxiliary_evaluation') is not True:
            raise ValueError(NAME+' auxiliary evaluation must have zero training and remain separate')
    elif metadata.get('release_seed')!=20261006:
        raise ValueError(NAME+' release seed mismatch')


def audit(data_root, *, verify_shards=False):
    root=Path(data_root);index_path=root/'control/training-index.json'
    index_id=verify_file(index_path,INDEX_PIN);index=json.loads(index_path.read_text(encoding='utf-8-sig'))
    if index.get('state')!='READY' or index['release_summary']['split_counts']!=MAIN_COUNTS:
        raise ValueError('READY index with exact '+NAME+' split counts required')
    receipt_path=root/'control/release-receipt.json';verify_file(receipt_path,index['release_receipt'])
    receipt=json.loads(receipt_path.read_text(encoding='utf-8-sig'))
    if receipt.get('state')!='COMPLETE' or receipt.get('split_counts')!=MAIN_COUNTS or receipt['manifests'][KEY]!=index['datasets'][KEY]:
        raise ValueError('COMPLETE release receipt and '+NAME+' index disagree')
    local_pins=json.loads((Path(__file__).parent/f'pins/{KEY}-local-inputs.json').read_text(encoding='utf8'))
    download_path=root/'control/download-receipt.json'
    verify_file(download_path,local_pins['download_receipt'])
    download=json.loads(download_path.read_text(encoding='utf-8-sig'))
    if download.get('state')!='COMPLETE' or download.get('all_selected_objects_verified') is not True:
        raise ValueError('Complete verified staging receipt required')
    if any(download['index_pin'].get(k)!=v for k,v in INDEX_PIN.items()):raise ValueError('Staging receipt index pin mismatch')
    main_path=root/NAME/'manifest.json';main_id=verify_file(main_path,index['datasets'][KEY])
    main,main_shards,main_meta=read_manifest(main_path,'main')
    aux_root=root/'control/evaluation/aerial-positive63-v1';aux_receipt_path=aux_root/'receipt.json'
    verify_file(aux_receipt_path,index['auxiliary_evaluations']['aerial-positive63-v1']['receipt'])
    aux_receipt=json.loads(aux_receipt_path.read_text(encoding='utf-8-sig'))
    aux_path=root/NAME/'evaluation/aerial-positive63-v1/manifest.json'
    aux_id=verify_file(aux_path,aux_receipt['manifests'][KEY])
    aux,aux_shards,aux_meta=read_manifest(aux_path,'aerial')
    quality_path=aux_root/'quality-audit.json'
    quality_id=verify_file(quality_path,index['auxiliary_evaluations']['aerial-positive63-v1']['quality_audit'])
    quality=json.loads(quality_path.read_text(encoding='utf-8-sig'))
    if (quality.get('passed') is not True or quality.get('gaps') or quality.get('required_review_fraction')!=1
            or set(quality.get('reviewed_added_ids',[]))!={r['sample_id'] for r in aux}):
        raise ValueError('Auxiliary complete recorded actual-image review must match exact identities')
    if aux_meta.get('train_sample_count')!=0 or aux_meta.get('separate_auxiliary_evaluation') is not True:
        raise ValueError('Auxiliary evaluation must remain separate with zero training')
    validate_metadata(main_meta,auxiliary=False)
    validate_metadata(aux_meta,auxiliary=True)
    main_result=audit_rows(main,MAIN_COUNTS);aux_result=audit_rows(aux,AUX_COUNTS)
    combined=audit_rows(main+aux)
    main_members=membership(main,main_shards,NAME);aux_members=membership(aux,aux_shards,NAME+'/evaluation/aerial-positive63-v1')
    members={'train':main_members.get('train',[]),'validation':main_members.get('validation',[]),
             'aerial_validation':aux_members.get('validation',[])}
    failures=main_result['failures']+aux_result['failures']+combined['failures']
    shared={r['sample_id'] for r in main}&{r['sample_id'] for r in aux}
    if shared:failures.append({'kind':'main_aerial_identity_overlap','sample_ids':sorted(shared)})
    if main_result['source_counts_by_split'].get('validation')!={'maxar':115}:
        failures.append({'kind':'main_validation_source_scope','actual':main_result['source_counts_by_split'].get('validation')})
    if main_meta.get('release_seed')!=20261006 or main_meta.get('recipe')!=RECIPE:
        failures.append({'kind':'release_recipe_or_seed','actual':main_meta})
    shard_pins={(Path(NAME)/s['split']/(s['object']['sha256']+'.tar')).as_posix():s['object'] for s in main_shards if s['split']!='test'}
    aux_pins={(Path(NAME+'/evaluation/aerial-positive63-v1')/s['split']/(s['object']['sha256']+'.tar')).as_posix():s['object'] for s in aux_shards if s['split']=='validation'}
    staged={'status':'NOT_RUN','files':0,'bytes':0,'policy':'selected TRAIN/main-validation/aerial-validation opaque bytes only'}
    if verify_shards:
        for relative,pin in dict(shard_pins,**aux_pins).items():
            verify_file(root/relative,pin);staged['files']+=1;staged['bytes']+=pin['bytes']
        staged['status']='PASS'
    return {'schema':KEY+'-metadata-audit-v1','status':'FAIL' if failures else 'PASS','failures':failures,
            'staged_shard_integrity':staged,
            'input_identities':{'index':dict(INDEX_PIN,**index_id),'main_manifest':main_id,'aux_manifest':aux_id,'aerial_quality_audit':quality_id},
            'main':main_result,'aerial':aux_result,'combined':combined,'membership':members,
            'shard_pins':shard_pins,
            'auxiliary_shard_pins':aux_pins,
            'release_seed':main_meta.get('release_seed'),'recipe':main_meta.get('recipe'),
            'scope':'metadata leakage audit; optional opaque TRAIN/validation shard checksums; no images decoded; test shards never opened',
            'limitations':['Missing source/scene/capture metadata is unknown, not guessed.',
                           'Footprint comparisons use identical recorded CRS only; cross-CRS overlap and 2km buffered separation require a separate geospatial audit.',
                           'This metadata audit does not independently decode or verify payload support masks; frozen loader verifies decoded arrays.',
                           'Local generation is provenance from the immutable pin; only remote readback can prove cloud generation.']}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--source-map')
    parser.add_argument('--verify-shards',action='store_true',help='Hash all selected allowed shards as opaque bytes, never test shards')
    args=parser.parse_args(argv)
    input_root=Path(args.data_root).expanduser().resolve()
    for destination in (args.output,args.source_map):
        if destination and Path(destination).expanduser().resolve().is_relative_to(input_root):
            parser.error('Audit outputs must be outside the read-only data root')
    if args.source_map and Path(args.output).resolve()==Path(args.source_map).resolve():
        parser.error('Audit JSON and source map require separate destinations')
    try:report=audit(args.data_root,verify_shards=args.verify_shards)
    except (ValueError,KeyError,OSError) as error:
        report={'schema':KEY+'-metadata-audit-v1','status':'FAIL','failures':[{'kind':'input_integrity','error':str(error)}]}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    if args.source_map and 'membership' in report:
        target=Path(args.source_map);target.parent.mkdir(parents=True,exist_ok=True)
        columns=['selection','sample_id','split','scope','source_kind','site_id','geographic_group_id','acquisition_id','source_scene_id','shard','member']
        with target.open('w',newline='',encoding='utf8') as stream:
            writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore');writer.writeheader()
            for selection,rows in report['membership'].items():
                for row in rows:
                    exported=dict(row,selection=selection)
                    if isinstance(exported.get('source_scene_id'),list):exported['source_scene_id']=json.dumps(exported['source_scene_id'])
                    writer.writerow(exported)
    print(json.dumps({'status':report['status'],'failures':report['failures'],'output':str(output)}))
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())

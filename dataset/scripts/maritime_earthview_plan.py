"""GCE metadata-only EarthView training expansion planning; never release approval."""
import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time
from urllib.request import Request, urlopen
import zipfile
import requests
from requests.adapters import HTTPAdapter

from rasterio.warp import transform as project_xy
from shapely.geometry import box
from shapely.errors import GEOSException
from shapely.ops import transform
from scripts.maritime_cloud_inventory import create_only, json_bytes, parse_gs, require_cloud_vm, verify_output
from scripts.maritime_earthview_intake import ROOT, official_url, validate_mosaic
from scripts.maritime_marine_geography import CoastIndex, read_land, SHA, SIZE, URL


def sha(raw):return hashlib.sha256(raw).hexdigest()


def physical_key(row):return (int(row['epsg']),*map(float,row['core_bounds']))


def geometry(row):
 source=row.get('source',{});bounds=row.get('core_bounds',row.get('physical_core_bounds',source.get('core_bounds',source.get('physical_core_bounds'))))
 crs=row.get('crs',row.get('output_crs',source.get('crs',source.get('output_crs'))))
 if crs is None and source.get('epsg'):crs='EPSG:'+str(source['epsg'])
 if bounds is None or crs is None:raise ValueError('Missing protected physical geometry')
 polygon=box(*bounds)
 if not polygon.is_valid or polygon.area<=0:raise ValueError('Invalid protected physical geometry')
 return str(crs),polygon


class ProtectedIndex:
 def __init__(self,protected,rejected=()):
  self.protected=[geometry(r) for r in protected];self.rejected=[geometry(r) for r in rejected]
  self.acquisitions={str(r['acquisition_id']) for r in protected if r.get('acquisition_id')}
  self.sites={str(r.get('site_id','')) for r in protected};self.projected={}
 def reject(self,row):
  if str(row['acquisition_filename']) in self.acquisitions:return 'protected_acquisition'
  if any(prefix+str(row['port']['id']) in self.sites for prefix in ('earthview-port-','maxar-port-')):return 'protected_port_site'
  epsg=int(row['epsg']);crs='EPSG:'+str(epsg)
  if epsg not in self.projected:
   def project(values):
    result=[]
    for source,poly in values:
     xs,ys=project_xy(source,'EPSG:4326',[poly.centroid.x],[poly.centroid.y]);lon,lat=xs[0],ys[0]
     central=(epsg%100)*6-183
     if abs((lon-central+180)%360-180)>15:continue  # Far outside this UTM zone; 4km cannot bridge15degrees.
     def projector(x,y,z=None):return project_xy(source,crs,list(x),list(y))
     projected=poly if source==crs else transform(projector,poly)
     if not projected.is_valid:raise ValueError('Invalid nearby protected reprojection')
     result.append(projected)
    return result
   self.projected[epsg]=(project(self.protected),project(self.rejected))
  core=box(*row['core_bounds']);held,rejected=self.projected[epsg]
  if any(core.distance(poly)<=4000 for poly in held):return 'protected_core_within_4km'
  if any(core.intersection(poly).area>1 for poly in rejected):return 'prior_quality_rejected_core_overlap'
  return None


def site_cap_reached(count,cap):
 """An explicit None permits source inventory traversal without a site quota."""
 return cap is not None and count>=cap


def diverse_order(rows,unique_keys):
 """Unique physical locations first; round-robin country and port/date coverage."""
 first=[];other_new=[];repeat=[];seen=set()
 for row in sorted(rows,key=lambda r:(0 if physical_key(r) in unique_keys else 1,{'L':0,'M':1}.get(r['port']['size'],2),r['port_center_distance_m'] if 'port_center_distance_m' in r else 0,r['acquisition_filename'])):
  key=physical_key(row)
  (repeat if key in seen else first if key in unique_keys else other_new).append(row);seen.add(key)
 def balanced(values):
  countries=defaultdict(lambda:defaultdict(list))
  for row in values:countries[row['port']['country']][str(row['port']['id'])].append(row)
  queues={}
  for country,ports in countries.items():
   portqueues=[]
   for port,rs in sorted(ports.items(),key=lambda p:({'L':0,'M':1}.get(p[1][0]['port']['size'],2),p[0])):
    dates=defaultdict(deque)
    for row in rs:dates[row['acquisition_filename']].append(row)
    datequeues=[dates[k] for k in sorted(dates)];ordered=[]
    while datequeues:
     for q in datequeues:ordered.append(q.popleft())
     datequeues=[q for q in datequeues if q]
    portqueues.append(deque(ordered))
   ordered=[]
   while portqueues:
    for q in portqueues:ordered.append(q.popleft())
    portqueues=[q for q in portqueues if q]
   queues[country]=deque(ordered)
  result=[]
  while queues:
   for country in sorted(queues):result.append(queues[country].popleft())
   queues={c:q for c,q in queues.items() if q}
  return result
 return balanced(first)+balanced(other_new)+balanced(repeat)


def screen_marine(coast,row):
 try:
  return coast.classify(dict(source=dict(epsg=row['epsg']),physical_core_bounds=row['core_bounds']),margin_m=1000)
 except GEOSException as exc:
  return dict(status='UNKNOWN_INVALID_GEOMETRY',marine_area_m2=None,shore_distance_m=None,geometry_error=str(exc))


def marine_allowed(result):
 return result['status']=='MARINE_INTERSECTION' or (result['status']=='COAST_MARGIN_UNCERTAIN' and result['shore_distance_m']<=1000)


def cached_metadata(directory,pin):
 official_url(pin['url'],'.json');raw=(Path(directory)/(pin['sha256']+'.json')).read_bytes()
 if len(raw)!=pin['bytes'] or sha(raw)!=pin['sha256'] or not pin.get('etag'):raise ValueError('Immutable metadata pin mismatch')
 return json.loads(raw),raw


def verify_plan(row,items,evidence):
 verified=validate_mosaic(items,evidence,product='TOA')
 for key,value in [('mosaic_bounds',verified['mosaic_bounds']),('core_bounds',verified['physical_core_bounds']),('context_bounds',verified['physical_context_bounds'])]:
  if row[key]!=value:raise ValueError('Frozen index geometry differs from verified original chip grids')
 if int(row['epsg'])!=int(verified['acquisition_identity'][2]):raise ValueError('Frozen index geometry EPSG mismatch')
 return dict(row,acquisition_identity=verified['acquisition_identity'],input_product='TOA',toa_mapping='robust',planned_split='train',final_eligible=False,metadata_16_chip_exact_acquisition_and_alignment_pass=True)


def validate_resume(document,sourcepins):
 if document.get('schema')!='earthview-source-mosaics-v1':raise ValueError('Versioned parent plan required')
 for key in ('input_sha256','distinct_sha256','locks_sha256','gshhg'):
  if document['split_exclusion_evidence'].get(key)!=sourcepins[key]:raise ValueError('Parent scientific input pin changed')
 rows=document['mosaics']
 if any(r.get('planned_split')!='train' or r.get('input_product')!='TOA' or r.get('final_eligible') is not False for r in rows):raise ValueError('Parent plan must remain provisional training only')
 if len({tuple(sorted(r['json_paths'])) for r in rows})!=len(rows):raise ValueError('Duplicate parent mosaic')
 return rows


class MetadataStore:
 def __init__(self,directory,publish=None):
  self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True);self.receipts=self.directory/'receipts';self.receipts.mkdir(exist_ok=True)
  self.publish=publish;self.locks=defaultdict(threading.Lock);self.guard=threading.Lock()
  self.session=requests.Session();self.session.headers.update({'User-Agent':'Academic-EarthView-marine-plan/1','Accept-Encoding':'identity'})
  self.session.mount('https://',HTTPAdapter(pool_connections=32,pool_maxsize=32,pool_block=True))
 def get(self,key):
  url=key if key.startswith('https://') else ROOT+key;official_url(url,'.json')
  with self.guard:lock=self.locks[url]
  with lock:
   path=self.receipts/(sha(url.encode())+'.json')
   if path.exists():
    pin=json.loads(path.read_bytes());assert pin['url']==url
    item,raw=cached_metadata(self.directory,pin)
   else:
    for attempt in range(3):
     try:
      with self.session.get(url,stream=True,timeout=(10,30)) as response:
       response.raise_for_status();official_url(response.url,'.json')
       raw=response.raw.read(2*1024*1024+1,decode_content=False)
       if response.status_code!=200 or len(raw)>2*1024*1024 or not response.headers.get('ETag'):raise ValueError('Bounded JSON with ETag required')
       pin=dict(url=url,etag=response.headers['ETag'],bytes=len(raw),sha256=sha(raw),retrieved_utc=datetime.now(timezone.utc).isoformat())
      item=json.loads(raw);break
     except Exception:
      if attempt==2:raise
      time.sleep(attempt+1)
    cache=self.directory/(pin['sha256']+'.json')
    if cache.exists() and cache.read_bytes()!=raw:raise ValueError('Immutable metadata SHA conflict')
    if not cache.exists():cache.write_bytes(raw)
    pin['local_cache_path']=str(cache)
   if self.publish and 'object' not in pin:pin['object']=self.publish('metadata/'+pin['sha256']+'.json',raw)
   path.write_bytes(json_bytes(pin))
   return key,item,pin


def run(args):
 require_cloud_vm()
 archive=Path(args.archive)
 if archive.stat().st_size!=SIZE or hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()!=SHA:raise ValueError('Pinned GSHHG 2.3.7 archive mismatch')
 with zipfile.ZipFile(archive) as z:
  members=[n for n in z.namelist() if n.endswith('gshhs_f.b')]
  if len(members)!=1:raise ValueError('One full-resolution GSHHG L1 source required')
  with z.open(members[0]) as stream:coast=CoastIndex(read_land(stream))
 protected=json.loads(Path(args.locks).read_bytes());index=ProtectedIndex(protected['protected'],protected.get('quality_rejected',[]))
 evidence=json.loads(Path(args.evidence).read_bytes())['source_evidence']
 rows=[json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
 unique={physical_key(json.loads(line)) for line in Path(args.distinct).read_text().splitlines() if line.strip()}
 rows=diverse_order(rows,unique);output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
 from google.cloud import storage
 client=storage.Client();client._http.mount('https://',HTTPAdapter(pool_connections=32,pool_maxsize=32,pool_block=True));bucketname,prefix=parse_gs(args.output_prefix);bucket=client.bucket(bucketname)
 def publish(name,raw):return dict(uri=f'gs://{bucketname}/{prefix}/{name}',**create_only(bucket.blob(prefix+'/'+name),raw))
 store=MetadataStore(args.metadata_cache_dir or output/'metadata',publish)
 if args.seed_directory:
  seed=Path(args.seed_directory)
  for line in (seed/'metadata-receipts.jsonl').read_text().splitlines():
   record=json.loads(line)
   if 'receipt' not in record:continue
   pin=dict(record['receipt']);item,raw=cached_metadata(seed/'item-metadata',pin)
   destination=store.directory/(pin['sha256']+'.json')
   if destination.exists() and destination.read_bytes()!=raw:raise ValueError('Conflicting immutable seed metadata')
   if not destination.exists():destination.write_bytes(raw)
   pin['local_cache_path']=str(destination);receipt=store.receipts/(sha(pin['url'].encode())+'.json')
   if not receipt.exists():receipt.write_bytes(json_bytes(pin))
 rejects=Counter();errors=[];accepted=[];pins={};marine_cache={};sites=Counter();attempted=0;stage=0;marine_passed=0;resumed_count=0;resume_keys=set();parent_pin=None
 sourcepins=dict(input_sha256=sha(Path(args.input).read_bytes()),distinct_sha256=sha(Path(args.distinct).read_bytes()),locks_sha256=sha(Path(args.locks).read_bytes()),gshhg=dict(url=URL,sha256=SHA,bytes=SIZE,version='2.3.7'),planner_sha256=sha(Path(__file__).read_bytes()))
 if args.resume_pin:
  parent_pin=json.loads(Path(args.resume_pin).read_bytes());parent=json.loads(verify_output(client,parent_pin))
  accepted=validate_resume(parent,sourcepins);resumed_count=len(accepted);pins=dict(parent['metadata_pins'])
  resume_keys={tuple(sorted(r['json_paths'])) for r in accepted};sites=Counter(str(r['port']['id']) for r in accepted)
  sourcepins['parent_closed_stage']=parent_pin
 def close_stage(exhausted=False):
  nonlocal stage
  stage+=1
  document=dict(schema='earthview-source-mosaics-v1',status='VERIFIED_METADATA_TRAINING_PLAN_NOT_READY',source_evidence=evidence,mosaics=list(accepted),metadata_pins=dict(pins),input_product='TOA',toa_mapping='robust',source_census_complete=exhausted and not errors and not rejects['site_scheduling_cap'],metadata_traversal_exhausted=exhausted,final_dataset_ready=False,split_exclusion_evidence=sourcepins,counts=dict(input_candidates=len(rows),attempted=attempted,remaining_index_candidates=len(rows)-attempted,marine_geometry_passed_current_run=marine_passed,resumed_verified_plans=resumed_count,verified_plans=len(accepted),countries=dict(Counter(r['port']['country'] for r in accepted)),port_sizes=dict(Counter(r['port']['size'] for r in accepted)),full_acquisitions=len({tuple(r['acquisition_identity']) for r in accepted}),site_cap_reached_sites=sorted(site for site,count in sites.items() if site_cap_reached(count,args.site_cap)),sites=len(sites),site_counts=dict(sites),rejections=dict(rejects),metadata_errors=len(errors),errors=errors),capacity_projection=dict(hr_lr_uint8_rgb_decimal_gb_upper=len(accepted)*835584/1e9,approved_training_bytes=0,basis='one512HR+128LRpairperplan; all native-mask/quality/semantic/duplicate gates pending'),policy=dict(training_only=True,protected_core_buffer_m=4000,inland_margin_m=1000,site_cap=args.site_cap,raw_raster_bytes_read=0,vessel_presence='unconfirmed',coast_topology_caveat='2017 GSHHG prior; no contemporary semantic proof'))
  payload=json_bytes(document);name=f'stage-{stage:03d}-{len(accepted):05d}.json';local=output/name
  if local.exists():raise ValueError('Preserve immutable closed stage')
  local.write_bytes(payload);ref=publish('stages/'+name,payload);(output/(name+'.pin.json')).write_bytes(json_bytes(ref));print(json.dumps(dict(stage=stage,verified_plans=len(accepted),attempted=attempted,rejections=dict(rejects),metadata_errors=len(errors),source_census_complete=document['source_census_complete'],object=ref)),flush=True)
 last_closed=0
 def prepare():
  nonlocal attempted,marine_passed
  for row in rows:
   attempted+=1
   if tuple(sorted(row['json_paths'])) in resume_keys:continue
   site=str(row['port']['id'])
   if site_cap_reached(sites[site],args.site_cap):rejects['site_scheduling_cap']+=1;continue
   reason=index.reject(row)
   if reason:rejects[reason]+=1;continue
   key=physical_key(row)
   if key not in marine_cache:marine_cache[key]=screen_marine(coast,row)
   marine=marine_cache[key]
   if not marine_allowed(marine):rejects['marine_geography_'+marine['status']]+=1;continue
   marine_passed+=1
   yield row,marine
 iterator=iter(prepare())
 with ThreadPoolExecutor(max_workers=args.workers) as fetchers,ThreadPoolExecutor(max_workers=8) as mosaics:
  def verify(pair):
   row,marine=pair
   try:
    fetched=list(fetchers.map(store.get,row['json_paths']));verified=verify_plan(row,[item for key,item,pin in fetched],evidence)
    verified['marine_screening']=dict(marine,gshhg_sha256=SHA,marine_prior_only=True);verified['split_exclusion_evidence']=dict(locks_sha256=sourcepins['locks_sha256'],buffer_m=4000,protected_geometry_pass=True,final_selector_required=True)
    verified['site_id']='earthview-port-'+str(row['port']['id'])
    return verified,fetched,None,None
   except (ValueError,KeyError,TypeError) as exc:return None,None,'invalid_exact_source_metadata_'+str(exc),None
   except Exception as exc:return None,None,None,dict(json_paths=row['json_paths'],error=str(exc))
  exhausted=False
  while len(accepted)<args.target:
   batch=[]
   for _ in range(min(32,args.target-len(accepted))):
    try:batch.append(next(iterator))
    except StopIteration:exhausted=True;break
   if not batch:break
   for verified,fetched,rejection,error in mosaics.map(verify,batch):
    if rejection:rejects[rejection]+=1
    if error:errors.append(error)
    if verified:
     site=str(verified['port']['id'])
     if site_cap_reached(sites[site],args.site_cap):rejects['site_scheduling_cap']+=1;continue
     accepted.append(verified);sites[site]+=1
     for key,item,pin in fetched:pins[key]=pin
     if len(accepted)%args.stage_size==0 or len(accepted) in args.checkpoints:close_stage();last_closed=len(accepted)
   print(json.dumps(dict(attempted=attempted,verified_plans=len(accepted),rejects=dict(rejects),errors=len(errors))),flush=True)
  if len(accepted)!=last_closed or exhausted or not accepted:close_stage(exhausted)
 return len(accepted)


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('input','distinct','archive','locks','evidence','output-dir','output-prefix'):p.add_argument('--'+name,required=True)
 p.add_argument('--seed-directory')
 p.add_argument('--metadata-cache-dir')
 p.add_argument('--resume-pin')
 p.add_argument('--checkpoints',type=lambda value:[int(n) for n in value.split(',')],default=[2500,5000,12000,20000]);p.add_argument('--target',type=int,default=12000);p.add_argument('--stage-size',type=int,default=1000);p.add_argument('--workers',type=int,default=16)
 cap=p.add_mutually_exclusive_group()
 cap.add_argument('--site-cap',type=int,default=1000)
 cap.add_argument('--uncapped-sites',dest='site_cap',action='store_const',const=None,help='Explicitly remove the metadata planning site quota; final dataset site limits are unchanged')
 args=p.parse_args()
 if min(args.target,args.stage_size,args.workers)<1 or args.workers>32 or (args.site_cap is not None and args.site_cap<1):raise ValueError('Positive bounded planning controls required; workers <=32')
 run(args)

"""Uncapped official STAC coastal census; JSON and visual HEAD only, never rasters.

Candidates require later source-mask, local quality, semantics, duplicate and
frozen-split verification. Port size is a scheduling prior, never a vessel label.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen

from pyproj import CRS, Transformer
from shapely.geometry import Point, shape

CATALOG='https://maxar-opendata.s3.amazonaws.com/events/catalog.json'
SIZE_ORDER={'L':0,'M':1,'S':2,'V':3}


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def official_json(url):
    p=urlsplit(url)
    if p.scheme!='https' or p.hostname!='maxar-opendata.s3.amazonaws.com' or not p.path.startswith('/events/') or not p.path.endswith('.json'):
        raise ValueError('Only official Open Data event JSON permitted')
    return url


def item_identity(url):
    """Official path zones are padded; item IDs use their numeric UTM zone."""
    official_json(url)
    parts=urlsplit(url).path.split('/')
    zone=int(parts[-4])
    if not 1<=zone<=60:raise ValueError('Invalid item UTM zone')
    return '/'.join((str(zone),parts[-3],parts[-1].removesuffix('.json')))


def normalize_ports(document,receipt):
    result=[]
    for feature in document['features']:
        a=feature['attributes'];kind=str(a.get('harbortype') or '').strip().upper()
        if not (kind.startswith('C') or kind=='OR'): continue
        g=feature.get('geometry') or {}
        lon=g.get('x',a.get('longitude'));lat=g.get('y',a.get('latitude'))
        try: lon,lat=float(lon),float(lat)
        except (TypeError,ValueError): continue
        if not math.isfinite(lon+lat) or not -180<=lon<=180 or not -90<=lat<=90: continue
        result.append(dict(port_id=str(a['index_no']),name=a.get('port_name'),country_code_as_published=a.get('country'),
            longitude=lon,latitude=lat,harbor_type=kind,harbor_size=str(a.get('harborsize') or '').strip().upper(),source_query=receipt))
    return sorted(result,key=lambda p:(SIZE_ORDER.get(p['harbor_size'],4),p['port_id']))


def bbox_ports(bounds,ports):
    """Conservative geographic envelope only; exact projected check follows.

    9 km with 100 km/degree latitude and the smallest cosine over the envelope
    avoids excluding 8 km candidates. Dateline/high-latitude ambiguity bypasses
    this optional prefilter.
    """
    if not bounds or len(bounds)!=4 or not all(math.isfinite(float(v)) for v in bounds): return ports
    left,bottom,right,top=bounds
    if left>right or abs(top-bottom)>80 or max(abs(bottom),abs(top))>80: return ports
    dy=.09;dx=dy/max(.05,math.cos(math.radians(max(abs(bottom),abs(top))+dy)))
    return [p for p in ports if left-dx<=p['longitude']<=right+dx and bottom-dy<=p['latitude']<=top+dy]


def collection_ports(document,ports):
    bounds=document.get('extent',{}).get('spatial',{}).get('bbox')
    if not bounds: return ports
    found={p['port_id']:p for bbox in bounds for p in bbox_ports(bbox,ports)}
    return list(found.values())


def port_matches(item,ports):
    props=item['properties'];crs=CRS.from_epsg(props['proj:epsg'])
    if not crs.is_projected or not crs.axis_info or crs.axis_info[0].unit_name!='metre':
        raise ValueError('Projected metre footprint required')
    geometry=props['proj:geometry']
    if geometry.get('type') not in ('Polygon','MultiPolygon'): raise ValueError('Valid projected polygon required')
    polygon=shape(geometry)
    if polygon.geom_type not in ('Polygon','MultiPolygon') or polygon.is_empty or not polygon.is_valid or not all(math.isfinite(v) for v in polygon.bounds):
        raise ValueError('Valid projected polygon required')
    transformer=Transformer.from_crs(4326,crs,always_xy=True)
    result=[]
    for port in bbox_ports(item.get('bbox'),ports):
        x,y=transformer.transform(port['longitude'],port['latitude'])
        if not math.isfinite(x+y): continue
        distance=polygon.distance(Point(x,y))
        if distance<=8000+1e-6:
            result.append(dict(port,distance_from_valid_item_footprint_m=round(distance,2)))
    return sorted(result,key=lambda p:(SIZE_ORDER.get(p['harbor_size'],4),p['distance_from_valid_item_footprint_m'],p['port_id']))


def source_from_item(item,item_ref,event,event_ref,acquisition,acquisition_ref,ports,*,max_tile_cloud_percent=10):
    if isinstance(max_tile_cloud_percent,bool) or not isinstance(max_tile_cloud_percent,(int,float)) or not math.isfinite(max_tile_cloud_percent) or not 0<=max_tile_cloud_percent<=100:
        raise ValueError('Invalid tile cloud ceiling')
    if event.get('license')!='CC-BY-NC-4.0': raise ValueError('event_license_not_academic_free')
    p=item['properties'];gsd=p.get('gsd');cloud=p.get('tile:clouds_percent')
    if not isinstance(gsd,(int,float)) or isinstance(gsd,bool) or not math.isfinite(gsd) or not 0<gsd<=1:
        raise ValueError('native_gsd_not_verified_le1')
    if not isinstance(cloud,(int,float)) or isinstance(cloud,bool) or not math.isfinite(cloud) or not 0<=cloud<=max_tile_cloud_percent:
        raise ValueError(f'cloud_not_verified_le{max_tile_cloud_percent:g}')
    if p.get('platform')=='WV01': raise ValueError('panchromatic_platform')
    if not p.get('platform') or not p.get('catalog_id'): raise ValueError('missing_sensor_capture_identity')
    visual=item['assets']['visual'];bands=[b.get('common_name') for b in visual.get('eo:bands',[])]
    if bands!=['red','green','blue']: raise ValueError('visual_not_declared_rgb')
    transform=visual['proj:transform'];dimensions=visual['proj:shape']
    if len(transform)<6 or not all(math.isfinite(float(v)) for v in transform) or not 0<transform[0]<=1 or abs(transform[4]+transform[0])>1e-10 or transform[1]!=0 or transform[3]!=0:
        raise ValueError('visual_grid_not_supported')
    if len(dimensions)!=2 or not all(isinstance(n,int) and n>0 for n in dimensions): raise ValueError('invalid_visual_shape')
    href=urljoin(item_ref['url'],visual['href']);url=urlsplit(href)
    if url.scheme!='https' or url.hostname!='maxar-opendata.s3.amazonaws.com' or not url.path.endswith('.tif'):
        raise ValueError('visual_not_official_COG')
    matches=port_matches(item,ports)
    if not matches: raise ValueError('outside_8km_port_AOI')
    licenses={'event':event.get('license'),'acquisition':acquisition.get('license'),
              'item':item.get('license',p.get('license'))}
    conflict=any(v is not None and v!='CC-BY-NC-4.0' for v in licenses.values())
    masks={key:dict(asset,absolute_href=urljoin(item_ref['url'],asset['href'])) for key,asset in item.get('assets',{}).items()
           if 'mask' in key and asset.get('href')}
    return dict(source_id=p['catalog_id']+'_'+p.get('grid:code',item['id'].replace('/','-')),
        item_id=item['id'],item_metadata=item_ref,event_id=event['id'],event_metadata=event_ref,
        acquisition_metadata=acquisition_ref,event_license=licenses['event'],acquisition_license=licenses['acquisition'],
        item_license=licenses['item'],license_declarations=licenses,license_conflict=conflict,
        license_status='event_academic_grant_with_unresolved_nested_conflict' if conflict else 'event_academic_grant_documented',
        native_gsd_m=gsd,native_gsd_evidence='STAC properties.gsd; not visual output grid',
        visual_pixel_size_m=[transform[0],-transform[4]],rgb_band_names=bands,visual_href=href,visual_asset=visual,
        masks=masks,platform=p['platform'],datetime=p.get('datetime'),catalog_id=p['catalog_id'],gsd_pass=True,
        tile_cloud_percent=cloud,off_nadir_deg=p.get('view:off_nadir'),epsg=p['proj:epsg'],
        visual_shape=dimensions,visual_transform=transform,visual_bbox=visual.get('proj:bbox',p.get('proj:bbox')),
        valid_item_geometry_wgs84=item.get('geometry'),valid_item_bbox_wgs84=item.get('bbox'),
        valid_data_area_km2_metadata=p.get('tile:data_area'),port_AOI_matches=matches,
        country_codes_as_published=sorted({q['country_code_as_published'] for q in matches if q['country_code_as_published']}),
        maximum_complete_1280m_core_grid_count_before_masks=max(0,math.floor((dimensions[1]*transform[0]-80)/1280))*max(0,math.floor((dimensions[0]*transform[0]-80)/1280)),
        field_provenance={'native_gsd_m':'item.properties.gsd','tile_cloud_percent':'item.properties.tile:clouds_percent',
            'port_AOI_matches':'pinned OCHA WorldPortIndex points to item.properties.proj:geometry in item.properties.proj:epsg',
            'rgb_band_names':'item.assets.visual.eo:bands.common_name','event_license':'event.license'},
        semantic_status='port-proximity candidate only; maritime contents not confirmed',
        vessel_presence='unconfirmed',quality_status='metadata screened; native mask, local cloud and quality checks pending',final_eligible=False)


def schedule_sources(rows,max_sources=None):
    def rank(row):
        return (min(SIZE_ORDER.get(p['harbor_size'],4) for p in row['port_AOI_matches']),
                row['tile_cloud_percent'],row['source_id'])
    rows=sorted(rows,key=rank);selected=rows if max_sources is None else rows[:max_sources]
    return dict(sources=selected,available_source_count=len(rows),selected_source_count=len(selected),
                selection_cap_reached=len(selected)<len(rows),unscheduled_source_count=len(rows)-len(selected))


class MetadataCache:
    def __init__(self,directory,legacy,parents):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.receipts=self.directory/'url-receipts';self.receipts.mkdir(exist_ok=True)
        self.known={};self.ambiguous_urls=set();self.identities=defaultdict(set);self.lock=threading.Lock()
        self.counts=Counter()
        for path in Path(legacy).glob('*.json'):
            try:
                data=path.read_bytes();d=json.loads(data)
                if path.stem==sha(data) and d.get('type') and d.get('id'):
                    self.identities[(d['type'],d['id'])].add(path)
            except (ValueError,OSError): pass
        def scan(value):
            if isinstance(value,dict):
                if value.get('url') and value.get('sha256') and value.get('bytes'):
                    old=Path(legacy)/(value['sha256']+'.json')
                    if old.exists():
                        url=value['url']
                        if url in self.known and self.known[url][1]['sha256']!=value['sha256']:
                            self.ambiguous_urls.add(url)
                        if url not in self.ambiguous_urls:self.known[url]=(old,value)
                for v in value.values(): scan(v)
            elif isinstance(value,list):
                for v in value: scan(v)
        for parent in parents: scan(parent)
        for url in self.ambiguous_urls:self.known.pop(url,None)

    def get(self,url,expected_type=None,expected_id=None,unique_context=False):
        official_json(url);receipt_path=self.receipts/(sha(url.encode())+'.json')
        if receipt_path.exists():
            receipt=json.loads(receipt_path.read_bytes());path=self.directory/(receipt['sha256']+'.json')
            if receipt['url']!=url: raise ValueError('Cache URL identity conflict')
            data=path.read_bytes()
        else:
            old=None;basis=None
            if url in self.known: old=self.known[url][0];basis='exact prior pinned URL receipt'
            elif url not in self.ambiguous_urls and unique_context and expected_type and expected_id:
                matches=self.identities[(expected_type,expected_id)]
                if len(matches)==1: old=next(iter(matches));basis='unique immutable STAC type/id and unique official catalog URL context'
            if old:
                data=old.read_bytes();headers={}
                if sha(data)!=old.stem: raise ValueError('Prior immutable metadata cache changed')
                if url in self.known:
                    pin=self.known[url][1]
                    if len(data)!=pin['bytes'] or sha(data)!=pin['sha256']: raise ValueError('Prior metadata pin changed')
                with self.lock:self.counts['cached']=self.counts['cached']+1
            else:
                data,headers=self.fetch(url)
                with self.lock:self.counts['fetched']=self.counts['fetched']+1
            document=json.loads(data)
            if expected_type and document.get('type')!=expected_type: raise ValueError('STAC type identity mismatch')
            if expected_id and document.get('id')!=expected_id: raise ValueError('STAC ID identity mismatch')
            path=self.directory/(sha(data)+'.json')
            if not path.exists(): path.write_bytes(data)
            receipt=dict(url=url,sha256=sha(data),bytes=len(data),local_path=str(path),
                etag=headers.get('ETag'),last_modified=headers.get('Last-Modified'),
                retrieved_at=datetime.now(timezone.utc).isoformat(),cached_metadata_response=bool(old),cache_identity_check=basis)
            receipt_path.write_bytes(encoded(receipt))
        if len(data)!=receipt['bytes'] or sha(data)!=receipt['sha256']: raise ValueError('Immutable metadata cache bytes changed')
        document=json.loads(data)
        if expected_type and document.get('type')!=expected_type: raise ValueError('Cached STAC type mismatch')
        if expected_id and document.get('id')!=expected_id: raise ValueError('Cached STAC ID mismatch')
        return document,receipt

    @staticmethod
    def fetch(url):
        for attempt in range(3):
            try:
                with urlopen(Request(url,headers={'User-Agent':'Academic-maritime-metadata-census/2'}),timeout=60) as response:
                    data=response.read(16*1024*1024+1)
                    if len(data)>16*1024*1024: raise ValueError('Metadata response exceeds bound')
                    return data,dict(response.headers)
            except OSError:
                if attempt==2: raise
                time.sleep(attempt+1)


def visual_identity(headers,status,prior=None):
    size=int(headers.get('Content-Length','0'));etag=headers.get('ETag')
    if status!=200 or not etag or size<=0: raise ValueError('Visual HEAD identity incomplete')
    if prior and (prior.get('etag')!=etag or prior.get('content_length',prior.get('bytes'))!=size):
        raise ValueError('Visual object changed from parent inventory HEAD pin')
    return dict(status=status,content_length=size,bytes=size,etag=etag,
        last_modified=headers.get('Last-Modified'),content_type=headers.get('Content-Type'),accept_ranges=headers.get('Accept-Ranges'),
        checked_at=datetime.now(timezone.utc).isoformat())


def head_visual(row,prior=None):
    with urlopen(Request(row['visual_href'],method='HEAD'),timeout=60) as response:
        return dict(row,visual_http_head=visual_identity(response.headers,response.status,prior))


def cached_source_rows(directory,ports,parent_ids,*,max_tile_cloud_percent=10):
    """Read a partial snapshot of verified archive receipts without any HTTP."""
    directory=Path(directory);errors=[];counts=Counter();rows={}
    def read(url):
        receipt=json.loads((directory/'url-receipts'/(sha(url.encode())+'.json')).read_bytes())
        data=(directory/(receipt['sha256']+'.json')).read_bytes()
        if receipt['url']!=url or len(data)!=receipt['bytes'] or sha(data)!=receipt['sha256']:
            raise ValueError('Partial cached metadata pin mismatch')
        return json.loads(data),receipt
    for path in sorted((directory/'url-receipts').glob('*.json')):
        url=str(path)
        try:
            ref=json.loads(path.read_bytes());url=ref['url'];item,item_ref=read(url)
            if item.get('type')!='Feature':continue
            counts['cached_item_metadata_examined']+=1
            official_json(url);parts=urlsplit(url).path.split('/');event_id=parts[2]
            event_url=f'https://maxar-opendata.s3.amazonaws.com/events/{event_id}/collection.json'
            capture=item['properties']['catalog_id']
            acq_url=f'https://maxar-opendata.s3.amazonaws.com/events/{event_id}/ard/acquisition_collections/{capture}_collection.json'
            event,event_ref=read(event_url);acquisition,acq_ref=read(acq_url)
            if event.get('id')!=event_id or acquisition.get('id')!=capture:
                raise ValueError('Partial cached context identity mismatch')
            row=source_from_item(item,item_ref,event,event_ref,acquisition,acq_ref,ports,max_tile_cloud_percent=max_tile_cloud_percent)
            row['novel_source_id_vs_parents']=row['source_id'] not in parent_ids
            # Preserve duplicate event contexts, while choosing a single source.
            previous=rows.get(row['source_id'])
            aliases=[] if previous is None else previous['event_source_aliases']
            aliases.append(dict(event_id=row['event_id'],item_metadata=row['item_metadata'],
                event_metadata=row['event_metadata'],acquisition_metadata=row['acquisition_metadata'],
                license_declarations=row['license_declarations'],license_conflict=row['license_conflict']))
            if previous is None or (row['license_conflict'],row['event_id'])<(previous['license_conflict'],previous['event_id']):
                rows[row['source_id']]=row
            rows[row['source_id']]['event_source_aliases']=aliases
        except (KeyError,ValueError,TypeError) as exc:
            counts['metadata_rejected']+=1
        except Exception as exc:errors.append(dict(url=url,error=str(exc)))
    return list(rows.values()),errors,dict(counts)


def run_partial(args):
    """HEAD-pin only a bounded novel chunk; full census continues independently."""
    output=Path(args.output)
    if output.exists():raise ValueError('Preserve prior inventories: output already exists')
    parents=[json.loads(Path(p).read_text(encoding='utf-8-sig')) for p in args.parent_inventory]
    authority=parents[0]['port_authority'];ports=[]
    for pin in authority['query_pins']:
        data=(Path(args.legacy_cache_dir)/(pin['sha256']+'.json')).read_bytes()
        if sha(data)!=pin['sha256'] or len(data)!=pin['bytes']:raise ValueError('Pinned port query changed')
        ports.extend(normalize_ports(json.loads(data),pin))
    ids={row['source_id'] for parent in parents for row in parent.get('sources',[])}
    rows,errors,counts=cached_source_rows(args.cache_dir,ports,ids,max_tile_cloud_percent=args.max_tile_cloud_percent)
    novel=[row for row in rows if row['novel_source_id_vs_parents']]
    scheduled=schedule_sources(novel,args.max_sources or 128);checked=[];head_errors=[]
    def check(row):
        try:return head_visual(row),None
        except Exception as exc:return None,dict(url=row['visual_href'],error=str(exc))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row,error in pool.map(check,scheduled['sources']):
            if row:checked.append(row)
            if error:head_errors.append(error)
    result=dict(schema_version='maxar-coastal-metadata-inventory-v2' if args.max_tile_cloud_percent==10 else 'maxar-coastal-metadata-inventory-v3',created_at=datetime.now(timezone.utc).isoformat(),
        status='PARTIAL_HEAD_PINNED_METADATA_CANDIDATES_NOT_TRAINING_READY',source_census_complete=False,
        catalog_traversal_exhausted=False,head_checks_complete=not head_errors,port_authority=authority,
        method=dict(raw_raster_pixels_downloaded_locally=0,snapshot_basis='already archived exact URL JSON receipts only',
            source_gsd_max_m=1,tile_cloud_percent_max=args.max_tile_cloud_percent,max_sources=args.max_sources or 128,
            selection_priority='novel source IDs; large/medium coastal harbors; low cloud; no inferred vessel labels'),
        counts=dict(counts,unique_qualified_cached_sources=len(rows),parent_sources_already_present=len(rows)-len(novel),
            novel_qualified_cached_sources=len(novel),available_COGs=len(checked),
            source_asset_bytes=sum(row['visual_http_head']['bytes'] for row in checked)),
        available_source_count=len(novel),selection_cap_reached=scheduled['selection_cap_reached'],
        unscheduled_source_count=scheduled['unscheduled_source_count'],sources=checked,
        failed_head_checks=head_errors,metadata_fetch_errors=errors,
        coverage_warning='Explicitly partial. Native mask, quality, semantic, physical duplicate and frozen split checks required before release.')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(encoded(result))
    print(json.dumps(dict(output=str(output),counts=result['counts'],source_census_complete=False)),flush=True)
    return result


def run(args):
    output=Path(args.output)
    if output.exists(): raise ValueError('Preserve prior inventories: output already exists')
    parents=[json.loads(Path(path).read_text(encoding='utf-8-sig')) for path in args.parent_inventory]
    if not parents: raise ValueError('Pinned parent inventory with port query receipts required')
    authority=parents[0]['port_authority'];ports=[]
    prior_heads={}
    for parent in parents:
        for row in parent.get('sources',[]):
            if row.get('visual_http_head'):
                head=row['visual_http_head'];previous=prior_heads.get(row['visual_href'])
                if previous and (previous['etag']!=head['etag'] or previous['content_length']!=head['content_length']):
                    raise ValueError('Conflicting parent visual HTTP identities')
                prior_heads[row['visual_href']]=head
    for receipt in authority['query_pins']:
        data=(Path(args.legacy_cache_dir)/(receipt['sha256']+'.json')).read_bytes()
        if sha(data)!=receipt['sha256'] or len(data)!=receipt['bytes']: raise ValueError('Pinned port query changed')
        ports.extend(normalize_ports(json.loads(data),receipt))
    ports=sorted({p['port_id']:p for p in ports}.values(),key=lambda p:(SIZE_ORDER.get(p['harbor_size'],4),p['port_id']))
    cache=MetadataCache(args.cache_dir,args.legacy_cache_dir,parents)
    root,root_ref=cache.get(CATALOG,'Catalog','maxar-opendata',True)
    errors=[];rejects=Counter();counts=Counter();acquisition_urls=defaultdict(list)
    def links(document,url,rel):
        return [urljoin(url,link['href']) for link in document.get('links',[]) if link.get('rel')==rel]
    for url in links(root,CATALOG,'child'):
        try:
            event_id=urlsplit(url).path.split('/')[-2]
            event,ref=cache.get(url,'Collection',event_id,True);counts['events_examined']+=1
            if event.get('license')!='CC-BY-NC-4.0': rejects['event_license_not_academic_free']+=1;continue
            for acq_url in links(event,url,'child'):
                acquisition_urls[acq_url].append((event,ref))
        except Exception as exc:errors.append(dict(url=url,error=str(exc)))
    ids=Counter(urlsplit(url).path.split('/')[-1].removesuffix('_collection.json') for url in acquisition_urls)
    item_jobs=[]
    def acquisition_job(pair):
        url,contexts=pair
        try:
            identifier=urlsplit(url).path.split('/')[-1].removesuffix('_collection.json')
            document,ref=cache.get(url,'Collection',identifier,ids[identifier]==1)
            relevant=collection_ports(document,ports)
            return url,document,ref,contexts,relevant,None
        except Exception as exc:return url,None,None,contexts,[],str(exc)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for url,document,ref,contexts,relevant,error in pool.map(acquisition_job,acquisition_urls.items()):
            counts['acquisitions_examined']+=1
            if error:errors.append(dict(url=url,error=error));continue
            if not relevant:continue
            counts['acquisitions_port_relevant']+=1
            event,event_ref=contexts[0]
            for item_url in links(document,url,'item'):
                parts=urlsplit(item_url).path.split('/')
                identifier=item_identity(item_url)
                item_jobs.append((item_url,identifier,ids[document['id']]==1,event,event_ref,document,ref,relevant))
        print(json.dumps(dict(phase='acquisitions',counts=dict(counts),items_to_examine=len(item_jobs))),flush=True)
        def item_job(job):
            url,identifier,unique,event,event_ref,document,ref,relevant=job
            try:
                item,pin=cache.get(url,'Feature',identifier,unique)
            except Exception as exc:return None,None,dict(url=url,error=str(exc))
            try:
                row=source_from_item(item,pin,event,event_ref,document,ref,relevant,max_tile_cloud_percent=args.max_tile_cloud_percent)
                return row,None,None
            except (ValueError,KeyError,TypeError) as exc:return None,str(exc),None
            except Exception as exc:return None,None,dict(url=url,error=str(exc))
        rows=[]
        for row,rejection,error in pool.map(item_job,item_jobs):
            counts['item_metadata_examined']+=1
            if row:rows.append(row)
            if rejection:rejects[rejection]+=1
            if error:errors.append(error)
            if counts['item_metadata_examined']%500==0:
                print(json.dumps(dict(phase='items',examined=counts['item_metadata_examined'],eligible=len(rows))),flush=True)
        # Keep the same capture/tile only once, with all event declarations retained.
        grouped=defaultdict(list)
        for row in rows:grouped[row['source_id']].append(row)
        unique_rows=[]
        for aliases in grouped.values():
            preferred=sorted(aliases,key=lambda r:(r['license_conflict'],r['event_id']))[0]
            preferred['event_source_aliases']=[dict(event_id=r['event_id'],item_metadata=r['item_metadata'],
                event_metadata=r['event_metadata'],acquisition_metadata=r['acquisition_metadata'],
                license_declarations=r['license_declarations'],license_conflict=r['license_conflict']) for r in aliases]
            unique_rows.append(preferred)
        scheduled=schedule_sources(unique_rows,args.max_sources)
        checked=[];head_errors=[]
        def head_job(row):
            try:return head_visual(row,prior_heads.get(row['visual_href'])),None
            except Exception as exc:return None,dict(url=row['visual_href'],error=str(exc))
        for row,error in pool.map(head_job,scheduled['sources']):
            if row:checked.append(row)
            if error:head_errors.append(error)
    result=dict(schema_version='maxar-coastal-metadata-inventory-v2' if args.max_tile_cloud_percent==10 else 'maxar-coastal-metadata-inventory-v3',created_at=datetime.now(timezone.utc).isoformat(),
        status='METADATA_SCREENED_CANDIDATES_NOT_TRAINING_READY',catalog=root_ref,port_authority=authority,
        method=dict(raw_raster_pixels_downloaded_locally=0,source_gsd_max_m=1,tile_cloud_percent_max=args.max_tile_cloud_percent,
            max_acquisitions_per_event=None,max_acquisitions_per_port=None,max_sources=args.max_sources,
            selection_priority='large then medium coastal harbors; lower verified tile cloud; stable source identity',
            spatial_method='conservative collection/item geographic envelopes then exact projected valid item footprint distance <=8000m',
            grid_reference='https://developers.maxar.com/docs/specifications/ard-user-guide/about-ard'),
        source_census_complete=not errors,catalog_traversal_exhausted=True,head_checks_complete=not head_errors,
        available_source_count=scheduled['available_source_count'],selection_cap_reached=scheduled['selection_cap_reached'],
        unscheduled_source_count=scheduled['unscheduled_source_count'],sources=checked,
        counts=dict(counts,coastal_ports_indexed=len(ports),unique_matching_rows=len(unique_rows),available_COGs=len(checked),
            rejection_counts=dict(rejects),metadata_cache=dict(cache.counts),
            ports=len({p['port_id'] for r in checked for p in r['port_AOI_matches']}),
            countries=dict(Counter(c for r in checked for c in r['country_codes_as_published'])),
            source_asset_bytes=sum(r['visual_http_head']['content_length'] for r in checked)),
        failed_head_checks=head_errors,metadata_fetch_errors=errors,
        coverage_warning='Port size/proximity are priors; ships, shoreline quality, complete native masks and release eligibility remain unconfirmed.')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_bytes(encoded(result));print(json.dumps(dict(output=str(output),counts=result['counts'],source_census_complete=result['source_census_complete'])),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir',required=True);parser.add_argument('--legacy-cache-dir',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--parent-inventory',action='append',required=True)
    parser.add_argument('--max-sources',type=int);parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--max-tile-cloud-percent',type=float,default=10,help='Explicit tile-wide cloud ceiling; original values retained, local window quality still required')
    parser.add_argument('--partial-from-cache',action='store_true',help='No new JSON requests; HEAD-pin a bounded novel cached chunk with incomplete census flag')
    args=parser.parse_args()
    if not 1<=args.workers<=16 or (args.max_sources is not None and args.max_sources<1): parser.error('workers1..16; positive optional max-sources')
    if not math.isfinite(args.max_tile_cloud_percent) or not 0<=args.max_tile_cloud_percent<=100: parser.error('tile cloud ceiling must be finite0..100')
    run_partial(args) if args.partial_from_cache else run(args)

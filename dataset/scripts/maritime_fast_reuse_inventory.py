"""Cloud-only parallel mask census; reference trusted existing native windows.

No raster duplication, inferred vessel labels, or training-ready claims. Original
objects remain generation pinned; native windows are checked again by release.
"""
import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import ExitStack
import json
from pathlib import Path
import re
import tempfile

import rasterio

from scripts.maritime_cloud_inventory import create_only, digest_bytes, json_bytes, parse_gs, require_cloud_vm, verify_output
from scripts.maritime_reuse_candidates import approved_asset_ids, download_pinned, source_rows, supported_windows

DEVELOPMENT={'08735b3a43aba3cf','8429fa5d1a1bf33d','8d28ae817826c699',
             '99e5a9ee6fa4ca5f','bf4e2020679c809e','fb6db363ca99da4e'}


def usable_sources(rows, approved, development):
    return [row for row in rows if row['asset_id'] in approved and row['site_id'] not in development]


def virtual_row(source, core, sha, mask_sha=None):
    if not re.fullmatch('[0-9a-f]{64}',sha):
        raise ValueError('Actual downloaded original SHA required')
    pin={key:source[key] for key in ('uri','generation','bytes','crc32c')}
    pin['sha256']=sha
    identity=digest_bytes(json_bytes(core['global_core_identity']))
    mask=dict(provenance=core['mask_provenance'])
    if source.get('producer_mask'):
        if not mask_sha or not re.fullmatch('[0-9a-f]{64}',mask_sha):
            raise ValueError('Actual original mask SHA required')
        mask['object']=dict(source['producer_mask'],sha256=mask_sha)
    original=dict(core,source_object=pin,mask=mask,native_gsd_m=.8,core_bytes=7680000,
                  asset_id=source['asset_id'],source_metadata=source,license='CC-BY-4.0',
                  attribution='Planet Labs PBC',source_grid_gsd_m=.8)
    country=source.get('country')
    if isinstance(country,list): country=country[0] if len(country)==1 else None
    return dict(sample_id=identity,source_kind='skysat_raw_window',source=original,
                site_id=source['site_id'],geographic_group_id=source['site_id'],
                acquisition_id=source['catalog_id'],crs=core['crs'],core_bounds=core['core_bounds'],
                country=country,vessel_presence='unconfirmed',small_craft_presence='unconfirmed',
                trust_basis='user_inspected_existing_collection_and_documented_coastal_asset_screen',
                requires_stratified_native_audit=True)


def unique_rows(rows, site_cap):
    result=[]; seen=set(); cells=set(); count=Counter()
    for row in rows:
        key=(row['acquisition_id'],row['crs'],tuple(round(x,5) for x in row['core_bounds']))
        if row['sample_id'] in seen or key in cells or count[row['site_id']]>=site_cap:
            continue
        seen.add(row['sample_id']);cells.add(key);count[row['site_id']]+=1;result.append(row)
    return result


def run(args):
    instance=require_cloud_vm()
    from google.cloud import storage
    approval=json.loads(Path(args.approved_assets).read_text(encoding='utf-8-sig'))
    sources=usable_sources(source_rows(json.loads(Path(args.sources).read_text(encoding='utf-8-sig')),approval),
                           approved_asset_ids(approval),DEVELOPMENT)
    if not sources or not 1<=args.workers<=8 or args.site_cap<1:
        raise ValueError('Nonempty sources, workers1..8 and positive site cap required')
    scratch=Path(args.scratch);scratch.mkdir(parents=True,exist_ok=True)
    client=storage.Client(); bn,prefix=parse_gs(args.output_prefix.rstrip('/'))
    config=dict(schema='trusted-skysat-virtual-windows-v1',sources_sha256=digest_bytes(json_bytes(sources)),
                approval_sha256=digest_bytes(json_bytes(approval)),site_cap=args.site_cap,
                development_excluded=sorted(DEVELOPMENT),workers=args.workers,
                review_policy='trusted existing imagery; individual semantics remain unknown absent reviewed ledger')
    def put(name,data):
        return dict(uri=f'gs://{bn}/{prefix}/{name}',**create_only(client.bucket(bn).blob(prefix+'/'+name),data))
    put('configuration.json',json_bytes(config))
    def process(source):
        key=digest_bytes(json_bytes(source));name='source-reports/'+key+'.json'
        blob=client.bucket(bn).blob(prefix+'/'+name)
        if blob.exists():
            blob.reload(); data=blob.download_as_bytes(if_generation_match=int(blob.generation),checksum='crc32c')
            report=json.loads(data)
            if report['configuration']!=config: raise ValueError('Source census configuration conflict')
            return report
        rows=[];stats={};error=None
        with tempfile.TemporaryDirectory(dir=scratch,prefix='census-') as directory:
            path=Path(directory)/'original.tif'
            try:
                sha=download_pinned(client,source,path);mask_sha=None
                with ExitStack() as stack:
                    ds=stack.enter_context(rasterio.open(path));ms=None
                    if source.get('producer_mask'):
                        mp=Path(directory)/'mask.tif';mask_sha=download_pinned(client,source['producer_mask'],mp)
                        ms=stack.enter_context(rasterio.open(mp))
                    for core in supported_windows(ds,source,ms,statistics=stats):
                        rows.append(virtual_row(source,core,sha,mask_sha))
            except (ValueError,rasterio.errors.RasterioError) as exc:
                error=str(exc)
        report=dict(configuration=config,source=source,rows=rows,statistics=stats,error=error)
        put(name,json_bytes(report))
        return report
    queues=defaultdict(deque)
    for source in sources: queues[source['site_id']].append(source)
    # One active source per site prevents unnecessary downloads after site cap.
    ready=deque(sorted(queues));active={};reports=[];rows=[];counts=Counter()
    def census_progress():
        remaining={site:len(queue) for site,queue in sorted(queues.items()) if queue}
        return dict(source_census_complete=not active and not remaining,
                    remaining_queued_source_count=sum(remaining.values()),
                    remaining_queued_source_counts_by_site=remaining,
                    site_cap_reached_sites=sorted(site for site,count in counts.items() if count>=args.site_cap))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while ready or active:
            while ready and len(active)<args.workers:
                site=ready.popleft()
                if counts[site]>=args.site_cap or not queues[site]: continue
                active[pool.submit(process,queues[site].popleft())]=site
            if not active: break
            done,_=wait(active,return_when=FIRST_COMPLETED)
            for future in done:
                site=active.pop(future); report=future.result();reports.append(report)
                combined=unique_rows(rows+report['rows'],args.site_cap);rows=combined
                counts=Counter(row['site_id'] for row in rows)
                if queues[site] and counts[site]<args.site_cap: ready.append(site)
                snapshot=dict(schema='trusted-skysat-virtual-windows-v1',configuration=config,instance_id=instance,
                    candidates=sorted(rows,key=lambda r:r['sample_id']),candidate_count=len(rows),
                    useful_native_candidate_bytes=len(rows)*7680000,site_counts=dict(counts),
                    completed_source_reports=len(reports),**census_progress(),
                    source_errors=[dict(uri=r['source']['uri'],error=r['error']) for r in reports if r['error']],
                    status='TRUSTED_REUSE_WINDOWS_PENDING_RELEASE_VERIFICATION')
                temporary=scratch/'checkpoint.tmp';temporary.write_bytes(json_bytes(snapshot));temporary.replace(scratch/'checkpoint.json')
                print(json.dumps(dict(sources=len(reports),cores=len(rows),site=site,site_cores=counts[site])),flush=True)
    snapshot.update(census_progress())
    temporary=scratch/'checkpoint.tmp';temporary.write_bytes(json_bytes(snapshot));temporary.replace(scratch/'checkpoint.json')
    ref=put('candidate-manifest.json',json_bytes(snapshot));print(json.dumps(dict(manifest=ref,cores=len(rows))),flush=True)
    return snapshot


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',required=True);p.add_argument('--approved-assets',required=True)
    p.add_argument('--output-prefix',required=True);p.add_argument('--scratch',required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--site-cap',type=int,default=781)
    run(p.parse_args())

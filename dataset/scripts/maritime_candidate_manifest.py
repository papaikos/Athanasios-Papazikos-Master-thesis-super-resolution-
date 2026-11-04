"""Metadata-only GCS SDK inventory of provisional native review candidates.

Read-only locally by default. --publish requires GCE and writes immutable JSON.
--prefix can repeat for reuse, Planet, disaster, and pilot output roots.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from rasterio.crs import CRS

from scripts.maritime_cloud_inventory import create_only, digest_bytes, json_bytes, parse_gs, require_cloud_vm


def pinned(ref, *, sha_required=False):
    if not isinstance(ref, dict):
        raise ValueError('Missing generation-pinned object provenance')
    parse_gs(ref.get('uri', ''))
    if not str(ref.get('generation', '')).isdigit() or int(ref['generation']) <= 0:
        raise ValueError('Missing positive object generation')
    if not isinstance(ref.get('bytes'), int) or ref['bytes'] <= 0 or not ref.get('crc32c'):
        raise ValueError('Missing object byte size or CRC32C')
    if sha_required and (len(ref.get('sha256', '')) != 64):
        raise ValueError('Missing artifact SHA256')
    return {key: ref[key] for key in ('uri','generation','bytes','crc32c','sha256') if key in ref}


def catalog_index(document):
    if isinstance(document,dict): return document
    if not isinstance(document,list): raise ValueError('Catalog metadata must be rows or an asset map')
    index={}
    catalogs={}
    ambiguous=set()
    for row in document:
        identity=row.get('id')
        if identity: index[identity]=row
        catalog=row.get('properties',{}).get('catalogID')
        if catalog:
            if catalog in catalogs: ambiguous.add(catalog)
            else: catalogs[catalog]=row
    index.update({key:value for key,value in catalogs.items() if key not in ambiguous})
    return index


def normalize_candidate(candidate, report, report_ref, site_metadata=None, catalog_metadata=None):
    source = candidate.get('source_object') or candidate.get('source') or report.get('source')
    if not isinstance(source, dict):
        raise ValueError('Missing source provenance')
    catalog_record=(catalog_metadata or {}).get(source.get('asset_id') or source.get('id'),{})
    if not catalog_record:
        catalog_record=(catalog_metadata or {}).get(source.get('catalog_id'),{})
    properties={**catalog_record.get('properties',{}),**source.get('stac',{}).get('properties',{}),
        **source.get('source_stac',{}).get('properties',{}),**source.get('properties',{})}
    catalog = source.get('catalog_id') or properties.get('catalogID')
    strip = source.get('strip_id') or source.get('stac', {}).get('properties', {}).get('pl:strip_id')
    # Full EE asset IDs and short catalog IDs route to the same acquisition.
    acquisition = 'ee_catalog:' + str(catalog).rsplit('/',1)[-1] if catalog else 'planet_strip:' + str(strip) if strip else None
    if acquisition is None:
        raise ValueError('Missing explicit original acquisition catalog/strip identity')
    crs = candidate.get('crs') or report.get('crs')
    parsed_crs = CRS.from_user_input(crs)
    if not parsed_crs.is_projected or parsed_crs.linear_units != 'metre':
        raise ValueError('Projected metre CRS required for core geometry')
    crs = parsed_crs.to_string()
    transform = candidate.get('core_transform') or candidate.get('mother_core_transform')
    bounds = candidate.get('core_bounds')
    if transform is None:
        if bounds is None or len(bounds) != 4:
            raise ValueError('Missing original core geometry')
        transform = [.8,0,bounds[0],0,-.8,bounds[3]]
    if len(transform) != 6 or not all(math.isfinite(v) for v in transform) or abs(transform[0]-.8)>1e-6 or abs(transform[4]+.8)>1e-6 or transform[1] or transform[3]:
        raise ValueError('Native 0.8 m north-up core geometry required')
    calculated = [transform[2], transform[5]+1600*transform[4], transform[2]+1600*transform[0], transform[5]]
    if bounds is not None and (len(bounds)!=4 or max(abs(a-b) for a,b in zip(bounds,calculated))>.002):
        raise ValueError('Core transform and bounds disagree')
    bounds = calculated
    context_transform = [transform[0],0,transform[2]-50*transform[0],0,transform[4],transform[5]-50*transform[4]]
    context_bounds = [bounds[0]-40,bounds[1]-40,bounds[2]+40,bounds[3]+40]
    artifacts = candidate.get('artifacts', {})
    mother = pinned(artifacts.get('rgb') or candidate.get('mother'), sha_required=True)
    preview = pinned(artifacts.get('native_preview') or candidate.get('preview') or candidate.get('native_review_preview'), sha_required=True)
    mask_ref = artifacts.get('mask') or candidate.get('mother_mask')
    provenance = candidate.get('mask_provenance') or candidate.get('original_mask_provenance') or candidate.get('source_support') or report.get('mask_provenance')
    allowed_masks={'legacy_nodata_zero_conservative','documented_unmask_zero_nodata_zero_conservative',
        'original_explicit_dataset_mask','original_embedded_alpha_mask','original_embedded_dataset_mask',
        'original_udm_all_unusable_bits_zero','original_udm2_all_unusable_bits_zero',
        'producer_original_min_rgb_mask'}
    if provenance not in allowed_masks:
        raise ValueError('Missing original mask provenance')
    conservative = 'conservative' in provenance
    if conservative:
        recipe = source.get('export_recipe', {})
        if recipe.get('unmask') != 0 or recipe.get('noData') != 0:
            raise ValueError('Conservative original mask has no documented export provenance')
    original_mask_ref=report.get('mask_mirror')
    if provenance=='producer_original_min_rgb_mask':
        original_mask_ref=dict(pinned(source.get('producer_mask')))
        producer_sha=candidate.get('producer_mask_sha256') or report.get('producer_mask_sha256') or original_mask_ref.get('sha256')
        if not producer_sha or len(producer_sha)!=64:
            raise ValueError('Missing original producer mask SHA256 provenance')
        original_mask_ref['sha256']=producer_sha
    mask = dict(kind='separate_validity' if mask_ref else 'embedded_original_support',
                object=pinned(mask_ref,sha_required=True) if mask_ref else mother, provenance=provenance,
                halo_pixels=50, context_size=[1700,1700])
    originals = [pinned(report['original_mirror'])] if report.get('original_mirror') else [pinned(source)]
    source_sha = candidate.get('source_sha256') or report.get('source_sha256') or report.get('original_http',{}).get('sha256')
    if not source_sha or len(source_sha)!=64:
        raise ValueError('Missing complete original SHA256 provenance')
    site = candidate.get('site_id') or source.get('site_id')
    if not site:
        raise ValueError('Missing source site provenance')
    metadata = (site_metadata or {}).get(site, {})
    country = source.get('country') or source.get('site_country') or candidate.get('country') or metadata.get('country')
    country_candidates=[]
    if isinstance(country,list):
        if any(not isinstance(value,str) or not value.strip() for value in country):
            raise ValueError('Invalid documented country list')
        country_candidates=sorted({value.strip() for value in country})
        country=country_candidates[0] if len(country_candidates)==1 else None
    elif country is not None:
        if not isinstance(country,str) or not country.strip():
            raise ValueError('Invalid documented country')
        country=country.strip()
        country_candidates=[country]
    for options in (source.get('country_candidates',[]),metadata.get('country_candidates',[])):
        if not isinstance(options,list) or any(not isinstance(value,str) or not value.strip() for value in options):
            raise ValueError('Invalid country candidate metadata')
        country_candidates=sorted(set(country_candidates+[value.strip() for value in options]))
    license_value = candidate.get('license') or source.get('license') or ('CC-BY-4.0' if catalog else None)
    if license_value not in ('CC-BY-4.0','CC-BY-NC-4.0'):
        raise ValueError('Verified free/academic imagery license required')
    overview_ref = artifacts.get('overview') or report.get('source_overview')
    identity = dict(acquisition_id=acquisition,crs=crs,core_bounds_mm=[round(value*1000) for value in bounds])
    flags = []
    if candidate.get('near_stereo_review_pending') or site == 'san_francisco': flags.append('near_stereo_review_pending')
    if candidate.get('requires_quality_and_near_duplicate_review'): flags.append('near_duplicate_review_pending')
    if conservative: flags.append('conservative_original_mask')
    core_window=candidate.get('core_window')
    context_window=candidate.get('context_window')
    read_window=candidate.get('source_pixel_window')
    fractional_window=candidate.get('source_fractional_window') or candidate.get('physical_source_fractional_window')
    if read_window is not None:
        window_status='integer_and_fractional_area_window_recorded' if fractional_window is not None else 'integer_read_window_recorded_fractional_support_unresolved'
    elif core_window is not None:
        window_status='original_native_windows_recorded' if context_window is not None else 'core_recorded_context_unresolved'
    else:
        window_status='fractional_area_window_recorded_integer_read_unresolved' if fractional_window is not None else 'unresolved_source_windows'
    observed_header=(candidate.get('native_header') or candidate.get('source_native_header')
                     or report.get('native_header') or source.get('native_header') or {})
    raster_transform=report.get('original_transform') or observed_header.get('affine') or source.get('original_transform')
    source_grid=source.get('grid') or source.get('native_plan')
    grid_scope='source_manifest_grid' if source.get('grid') else 'native_export_plan' if source.get('native_plan') else 'unresolved'
    if not source_grid and (raster_transform or observed_header):
        source_grid=dict(observed_header or dict(crs=report.get('crs'),original_transform=raster_transform))
        grid_scope='observed_raster_header'
    if not source_grid and catalog_record.get('bands'):
        source_grid=catalog_record['bands'][0].get('grid')
        grid_scope='catalog_original_asset_grid'
    acquisition_utc=(properties.get('collectionStartTime') if catalog else properties.get('datetime')) or (
        candidate.get('acquisition_utc') or source.get('acquisition_utc') or source.get('datetime')
        or source.get('native_plan',{}).get('acquisition_utc') or catalog_record.get('startTime')
        or source.get('startTime') or source.get('timestamp'))
    if acquisition_utc is None and isinstance(properties.get('system:time_start'),(int,float)):
        acquisition_utc=datetime.fromtimestamp(properties['system:time_start']/1000,timezone.utc).isoformat()
    if acquisition_utc is not None and not isinstance(acquisition_utc,str): acquisition_utc=None
    processing=candidate.get('source_processing') or report.get('resampling') or source.get('source_processing')
    if processing is None:
        processing='native_pixels_only_no_warp_no_resampling' if catalog else 'unresolved_intake_processing_record'
    return dict(candidate_id=candidate['sample_id'],canonical_core_id=digest_bytes(json_bytes(identity)),
        acquisition_id=acquisition,catalog_id=catalog,asset_id=source.get('asset_id') or source.get('id'),site_id=site,
        geographic_group_id=metadata.get('geographic_group_id') or candidate.get('geographic_group_id') or site,
        country=country,country_status='documented' if country else 'unresolved',country_candidates=country_candidates,crs=crs,
        core_bounds=bounds,core_transform=transform,context_bounds=context_bounds,context_transform=context_transform,
        world_grid_cell=candidate.get('world_grid_cell'),mother=mother,mask=mask,native_preview=preview,
        core_window=core_window,context_window=context_window,source_pixel_window=read_window,
        source_fractional_window=fractional_window,source_window_provenance_status=window_status,
        source_grid=source_grid,source_grid_scope=grid_scope,source_raster_transform=raster_transform,
        source_native_header=observed_header or None,
        source_raster_transform_status='recorded_original_raster_transform' if raster_transform else 'unresolved_tile_header_transform',
        source_properties=properties,source_view_angles={key:value for key,value in properties.items()
            if key.startswith('view:') or any(token in key.lower() for token in ('azimuth','elevationangle','off_nadir','offnadir'))},
        catalog_metadata_asset_id=catalog_record.get('id'),
        catalog_metadata_sha256=digest_bytes(json_bytes(catalog_record)) if catalog_record else None,
        overview=pinned(overview_ref,sha_required=True) if overview_ref else None,
        license=license_value,attribution=source.get('attribution') or 'Planet Labs PBC',
        license_url=source.get('license_url') or source.get('license_source') or (
            'https://creativecommons.org/licenses/by/4.0/' if license_value=='CC-BY-4.0'
            else 'https://creativecommons.org/licenses/by-nc/4.0/'),
        acquisition_utc=acquisition_utc,acquisition_time_status='documented' if acquisition_utc else 'unresolved',
        native_gsd_m=source.get('native_gsd_m',.8),native_gsd_status='source_metadata' if source.get('native_gsd_m') else 'catalog_nominal_0.8m',
        source_processing=processing,source_processing_history=source.get('processing_history') or report.get('processing_history'),
        source_objects=originals,source_sha256=source_sha,
        original_mask_object=pinned(original_mask_ref) if original_mask_ref else None,
        report_object=pinned(report_ref,sha_required=True),conservative_mask=conservative,
        semantic_status='pending',quality_status='pending',vessel_presence='unconfirmed',small_craft_presence='unconfirmed',
        final_eligible=False,flags=flags,alternate_candidate_ids=[],alternate_report_objects=[],core_bytes=7680000)


def overlaps(a,b):
    if (a['acquisition_id'],a['crs'])!=(b['acquisition_id'],b['crs']): return False
    x,y,z,t=a['core_bounds'];u,v,w,q=b['core_bounds']
    return min(z,w)-max(x,u)>1e-6 and min(t,q)-max(y,v)>1e-6


def consolidate(records):
    unique = {}
    for record in records:
        key=record['canonical_core_id']
        if key in unique:
            prior=unique[key]
            if record['candidate_id']!=prior['candidate_id'] and record['candidate_id'] not in prior['alternate_candidate_ids']:
                prior['alternate_candidate_ids'].append(record['candidate_id'])
            if record['report_object']!=prior['report_object'] and record['report_object'] not in prior['alternate_report_objects']:
                prior['alternate_report_objects'].append(record['report_object'])
            prior['flags']=sorted(set(prior['flags']+record['flags']))
        else:
            unique[key]=dict(record,flags=list(record['flags']),alternate_candidate_ids=[],alternate_report_objects=[])
    disjoint=[]
    candidates=list(unique.values())
    for candidate in candidates:
        conflicts=[previous['canonical_core_id'] for previous in disjoint if overlaps(candidate,previous)]
        candidate['same_acquisition_overlap_with']=conflicts
        if conflicts:
            candidate['flags']=sorted(set(candidate['flags']+['same_acquisition_overlap']))
            candidate['included_in_disjoint_capacity_upper_bound']=False
        else:
            disjoint.append(candidate)
            candidate['included_in_disjoint_capacity_upper_bound']=True
    return dict(schema_version=1,status='native_review_candidates_no_semantic_acceptance',candidates=candidates,
        summary=dict(input_candidate_count=len(records),distinct_candidate_count=len(candidates),
        exact_duplicate_count=len(records)-len(candidates),candidate_core_bytes=sum(c['core_bytes'] for c in candidates),
        disjoint_candidate_core_bytes_upper_bound=sum(c['core_bytes'] for c in disjoint),
        semantic_accepted_count=0,semantic_accepted_core_bytes=0,
        sites=len({c['site_id'] for c in candidates}),countries=sorted({c['country'] for c in candidates if c['country']}),
        unresolved_country_count=sum(c['country'] is None for c in candidates),
        licenses=dict(Counter(c['license'] for c in candidates)),
        capacity_limitations='Same-acquisition overlaps excluded; cross-acquisition near duplicates and semantic/quality rejection remain unresolved.'))


def read_json_pinned(blob,bucket_name,max_bytes=32*1024*1024):
    if blob.size is None or blob.size>max_bytes:
        raise ValueError('JSON metadata exceeds bounded read size')
    payload=blob.download_as_bytes(if_generation_match=int(blob.generation),checksum='crc32c')
    if len(payload)!=blob.size:
        raise ValueError('JSON metadata size changed during pinned download')
    return json.loads(payload),dict(uri=f'gs://{bucket_name}/{blob.name}',generation=str(blob.generation),
        bytes=len(payload),crc32c=blob.crc32c,sha256=digest_bytes(payload))


def run(args):
    instance=require_cloud_vm() if args.publish else None
    if args.publish:
        from google.cloud import storage
        client=storage.Client(project=args.project)
    else:
        from scripts.maritime_build_control import client_from_gcloud
        client=client_from_gcloud()
    site_metadata=json.loads(Path(args.site_metadata).read_text(encoding='utf-8-sig')) if args.site_metadata else {}
    catalog_metadata=catalog_index(json.loads(Path(args.catalog_metadata).read_text(encoding='utf-8-sig'))) if args.catalog_metadata else {}
    records,rejected,inputs=[],[],[]
    for uri in args.prefix:
        bucket,prefix=parse_gs(uri)
        objects=sorted(client.list_blobs(bucket,prefix=prefix.rstrip('/')+'/'),key=lambda blob:blob.name)
        manifest_name=prefix.rstrip('/')+'/candidate-manifest.json'
        shared_name=prefix.rstrip('/')+'/shared/common-samples.json'
        available={blob.name for blob in objects}
        selected=[blob for blob in objects if blob.name==manifest_name] if manifest_name in available else (
            [blob for blob in objects if blob.name==shared_name] if shared_name in available else
            [blob for blob in objects if blob.name.endswith('.json') and blob.name.startswith((
                prefix.rstrip('/')+'/sources/',prefix.rstrip('/')+'/source-reports/'))])
        if not selected:
            rejected.append(dict(prefix=uri,reason='No completed candidate metadata found'))
        for blob in selected:
            document,ref=read_json_pinned(blob,bucket)
            inputs.append(ref)
            candidates=document if isinstance(document,list) else document.get('candidates',[])
            report={} if isinstance(document,list) else document
            for candidate in candidates:
                try: records.append(normalize_candidate(candidate,report,ref,site_metadata,catalog_metadata))
                except (ValueError,KeyError,TypeError) as exc:
                    rejected.append(dict(candidate_id=candidate.get('sample_id'),report=ref,reason=str(exc)))
    result=consolidate(records)
    result.update(inputs=inputs,rejected_metadata=rejected,configuration=dict(prefixes=args.prefix,instance=instance,
        site_metadata_sha256=digest_bytes(json_bytes(site_metadata)),catalog_metadata_sha256=digest_bytes(json_bytes(catalog_metadata))))
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    payload=json_bytes(result)
    path.write_bytes(payload)
    if args.publish:
        bucket,prefix=parse_gs(args.publish)
        ref=create_only(client.bucket(bucket).blob(prefix.rstrip('/')+'/candidate-inventory.json'),payload)
        print(json.dumps(dict(uri=f'gs://{bucket}/{prefix.rstrip("/")}/candidate-inventory.json',**ref)),flush=True)
    print(json.dumps(result['summary']),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix',action='append',required=True)
    parser.add_argument('--project',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--site-metadata')
    parser.add_argument('--catalog-metadata',help='Explicit original EE catalog inventory JSON to join source properties/grids')
    parser.add_argument('--publish',help='GCE-only immutable output prefix')
    run(parser.parse_args())


if __name__=='__main__': main()

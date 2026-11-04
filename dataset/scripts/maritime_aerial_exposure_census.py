"""Metadata-only conservative exposure census; completeness requires enumeration."""
import hashlib
import json
from scripts.maritime_release_select import polygon_wgs84


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def normalize_row(row,profile):
    parent={};aliases=[];urls=[]
    if profile=='common':
        source=row.get('source') or row.get('source_provenance') or {}
        bounds=source.get('physical_context_bounds') or source.get('context_bounds') or row.get('context_bounds') or row.get('core_bounds')
        crs=row.get('crs') or source.get('source_crs') or source.get('crs')
        out={k:row.get(k) for k in ('sample_id','site_id','acquisition_id','source_kind','split')}
        for key in ('source_original_object','source_object','mother'):
            if isinstance(source.get(key),dict):parent[key]=source[key]
        original=source.get('source',{})
        urls=[v for v in (source.get('source_href'),original.get('href'),original.get('visual_href')) if v]
    elif profile in ('aerial_parent','maxar_parent'):
        aerial=profile=='aerial_parent';bounds=row.get('projected_bounds') if aerial else row.get('visual_bbox')
        crs='EPSG:'+str(row['epsg'])
        ports=[str(p['port_id']) for p in row.get('port_AOI_matches',[])]
        aliases=[('aerial-port-' if aerial else 'maxar-port-')+p for p in sorted(set(ports))]
        if not aliases:aliases=['original-parent-'+row['source_id']]
        out=dict(sample_id='exposure-parent-'+identity([profile,row['source_id']]),site_id=aliases[0],acquisition_id=row.get('acquisition_id') or row.get('catalog_id') or row.get('source_id'),source_kind='aerial_rgb' if aerial else 'maxar',split='exposure-only')
        urls=[row['href'] if aerial else row['visual_href']]
        if row.get('source_original_object'):parent['source_original_object']=row['source_original_object']
    elif profile=='earthview_plan':
        acq=row['acquisition_identity'];acq=acq[0] if isinstance(acq[0],list) else acq
        crs='EPSG:'+str(row.get('epsg',acq[2]));bounds=row.get('mosaic_bounds') or row.get('context_bounds')
        out=dict(sample_id=('earthview-'+identity([row.get('input_product','TOA'),sorted(row['json_paths'])])[:32] if row.get('json_paths') else 'exposure-earthview-'+identity([acq,bounds])),site_id=row.get('site_id') or 'earthview-port-'+str(row['port']['id']),acquisition_id=identity(acq),source_kind='earthview',split='exposure-only')
        urls=[p['url'] for p in row.get('source_pins',[])];aliases=['aerial-port-'+str(row['port']['id'])]
    elif profile=='explicit':
        return normalize_row(row,'common')
    else:raise ValueError('Unknown exposure normalization profile')
    if any(not out.get(k) for k in ('sample_id','site_id','acquisition_id','source_kind')):
        raise ValueError('Exposure identity/site/capture required')
    geometry=dict(crs=crs,core_bounds=bounds);polygon_wgs84(geometry)
    out.update(geometry,source=parent,site_aliases=aliases,source_parent_urls=sorted(set(urls)),geometry_semantics='conservative_observed_context_or_entire_original_parent',normalization_profile=profile)
    return out


def build_population(scope,inputs):
    if scope not in ('preserved_baseline','expanded_training','source_development'):raise ValueError('Unknown exposure scope')
    rows={};evidence=[]
    for pin,normalized in inputs:
        evidence.append(dict(pin=pin,enumerated_rows=len(normalized)))
        for row in normalized:
            key=identity(row);rows[key]=row
    return dict(schema='aerial-evaluation-exposure-population-v1',scope=scope,complete=False,
                completeness_semantics='Not asserted until all named exposure universes and review/model populations are enumerated and covered.',
                rows=[rows[k] for k in sorted(rows)],input_evidence=evidence)


def validate_coverage(population,checks):
    ids={r['sample_id'] for r in population['rows']};reports=[]
    for check in checks:
        missing=set(check['sample_ids'])-ids
        if missing:raise ValueError('Exposure coverage missing '+check['name']+': '+str(sorted(missing)[:5]))
        reports.append(dict(name=check['name'],count=len(check['sample_ids']),covered=True))
    return dict(covered=True,checks=reports)


def cover_rows(population_rows,observed_rows):
    """Prove exact IDs/geometry or original-URL plus enclosing parent geometry."""
    byid={};byurl={};shapes={};coverage=[]
    for index,row in enumerate(population_rows):
        byid.setdefault(row['sample_id'],[]).append(index)
        for url in row.get('source_parent_urls',[]):byurl.setdefault(url,[]).append(index)
    for observed in observed_rows:
        candidates=set(byid.get(observed['sample_id'],[]))
        for url in observed.get('source_parent_urls',[]):candidates.update(byurl.get(url,[]))
        shape=polygon_wgs84(observed);matched=None
        for index in sorted(candidates):
            if index not in shapes:shapes[index]=polygon_wgs84(population_rows[index])
            if shapes[index].buffer(1e-10).covers(shape):matched=index;break
        if matched is None:raise ValueError('Exposure census uncovered '+observed['sample_id'])
        coverage.append(dict(sample_id=observed['sample_id'],census_sample_id=population_rows[matched]['sample_id'],observed_row_sha256=identity(observed),census_row_sha256=identity(population_rows[matched])))
    return dict(covered_count=len(coverage),rows=coverage)

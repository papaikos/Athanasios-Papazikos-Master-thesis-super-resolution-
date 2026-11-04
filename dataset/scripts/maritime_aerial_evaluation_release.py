"""Separate aerial evaluation gates; never contributes to training quotas."""
from scripts.maritime_aerial_evaluation import allocate_source
from scripts.maritime_release_select import near,polygon_wgs84


def _geometry(row):
    source=row.get('source') or row.get('source_provenance') or {}
    if source.get('physical_context_bounds') and source.get('output_crs'):
        return dict(crs=source['output_crs'],core_bounds=source['physical_context_bounds'])
    if row.get('crs') and row.get('core_bounds'):return row
    from scripts.maritime_ultimate_release import source_geometry
    crs,bounds=source_geometry(row)
    return dict(crs=crs,core_bounds=bounds)


def _parent_hash(row):
    source=row.get('source',{})
    pin=source.get('source_original_object') or source.get('source_object') or source.get('mother')
    return pin.get('sha256') if pin else None


def _site_ids(row):
    source=row.get('source') or row.get('source_provenance') or {}
    original=source.get('source',{})
    values=[row['site_id'],*row.get('site_aliases',[])]
    values.extend(str(p['port_id']) for p in original.get('port_AOI_matches',[]))
    result=set()
    for value in values:
        for prefix in ('aerial-port-','maxar-port-','earthview-port-'):
            if value.startswith(prefix):value=value[len(prefix):];break
        result.add('port:'+value if value.isdigit() else value)
    return result


def _parent_urls(row):
    source=row.get('source') or row.get('source_provenance') or {}
    original=source.get('source',{})
    return {v for v in [*row.get('source_parent_urls',[]),source.get('source_href'),original.get('href'),original.get('visual_href')] if v}


def _near_exposure(shape,shapes,index):
    """Index the exact latitude pruning already used by authoritative near().

    Longitude remains unrestricted, including dateline and polar footprints.
    No distance or footprint approximation replaces the existing AEQD check.
    """
    from shapely.geometry import box
    _,bottom,_,top=shape.bounds
    # polygon_wgs84 unwraps valid longitude around a reference in [-180,180].
    candidates=index.query(box(-720,bottom-1,720,top+1))
    return any(near(shape,shapes[int(i)],4000) for i in sorted(candidates))


def validate_evaluation_selection(evaluation_rows,exposed_rows):
    """Exclude every supplied training/development/legacy-evaluation exposure.

    Callers must supply the complete preserved baseline and expanded selection,
    plus source-development exposures. Final pixel/source replay is separate.
    """
    if not evaluation_rows or not exposed_rows:raise ValueError('Nonempty evaluation and complete exposure populations required')
    seen=set();sites={site for r in exposed_rows for site in _site_ids(r)};captures={r['acquisition_id'] for r in exposed_rows}
    exposed_ids={r['sample_id'] for r in exposed_rows}
    parents={value for r in exposed_rows if (value:=_parent_hash(r))}
    parent_urls={url for r in exposed_rows for url in _parent_urls(r)}
    shapes=[polygon_wgs84(_geometry(r)) for r in exposed_rows]
    from shapely.strtree import STRtree
    spatial_index=STRtree(shapes)
    for row in evaluation_rows:
        if row['sample_id'] in seen:raise ValueError('Duplicate aerial evaluation identity')
        seen.add(row['sample_id'])
        if row['source_kind']!='aerial_rgb' or row['split'] not in ('validation','test'):
            raise ValueError('Separate aerial validation/test population required')
        if allocate_source(row['source']['source'],[])['role']!='evaluation':
            raise ValueError('Evaluation candidate violates frozen aerial reservation')
        if row['sample_id'] in exposed_ids or _site_ids(row).intersection(sites):
            raise ValueError('Evaluation identity/site already exposed')
        if row['acquisition_id'] in captures:raise ValueError('Evaluation capture already exposed')
        if _parent_hash(row) in parents or _parent_urls(row).intersection(parent_urls):raise ValueError('Evaluation original parent already exposed')
        shape=polygon_wgs84(_geometry(row))
        if _near_exposure(shape,shapes,spatial_index):
            raise ValueError('Separate aerial evaluation geography within4km of prior exposure')
    return evaluation_rows


def load_evaluation_inputs(store,document):
    """Load complete, immutable exposure populations before any native reads.

    Normalized population objects retain geometry, parent identity, site and
    capture for preserved baseline (including legacy val/test), expanded train,
    and every source-development exposure. Completeness is an explicit caller
    attestation; object pins prove bytes, not omissions in the supplied census.
    """
    import json
    from scripts.maritime_release_build import pinned
    if document.get('schema')!='aerial-evaluation-selection-v1':
        raise ValueError('Versioned separate aerial evaluation selection required')
    try: evidence_pin=pinned(document['exposure_evidence_pin'])
    except (KeyError,TypeError):raise ValueError('Full pinned exposure evidence required')
    evidence=json.loads(store.get(evidence_pin))
    required={'preserved_baseline','expanded_training','source_development'}
    populations=evidence.get('populations',[])
    if (evidence.get('schema')!='aerial-evaluation-exposure-evidence-v1'
            or evidence.get('complete') is not True
            or len(populations)!=len(required) or {p.get('scope') for p in populations}!=required):
        raise ValueError('Three complete pinned exposure populations required')
    exposed=[]
    for entry in populations:
        population=json.loads(store.get(pinned(entry['pin'])))
        if (population.get('schema')!='aerial-evaluation-exposure-population-v1'
                or population.get('scope')!=entry['scope'] or population.get('complete') is not True
                or not isinstance(population.get('rows'),list)):
            raise ValueError('Incomplete or mismatched exposure population')
        for row in population['rows']:
            if any(not row.get(k) for k in ('sample_id','source_kind','site_id','acquisition_id')):
                raise ValueError('Exposure identity/site/capture required')
            polygon_wgs84(_geometry(row))
        exposed.extend(population['rows'])
    rows=validate_evaluation_selection(document.get('selection',[]),exposed)
    groups={}
    for index,row in enumerate(rows):
        group=row.get('geographic_group_id')
        if not isinstance(group,str) or not group or groups.setdefault(group,row['split'])!=row['split']:
            raise ValueError('Separate evaluation geographic split assignment required')
        # Validation/test also remain disjoint from each other.
        for prior in rows[:index]:
            if prior['split']==row['split']:continue
            if (_site_ids(prior).intersection(_site_ids(row)) or prior['acquisition_id']==row['acquisition_id']
                    or _parent_urls(prior).intersection(_parent_urls(row))
                    or (_parent_hash(prior) is not None and _parent_hash(prior)==_parent_hash(row))
                    or near(polygon_wgs84(_geometry(prior)),polygon_wgs84(_geometry(row)),4000)):
                raise ValueError('Separate evaluation validation/test exposure overlap')
    return rows,exposed

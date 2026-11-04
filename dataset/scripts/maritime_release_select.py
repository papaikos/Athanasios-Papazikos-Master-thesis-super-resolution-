"""Metadata release selection and strict gates; never creates READY markers.

Reviews and audits are append-only JSONL. Latest explicit decisions win, with
resolved disagreement references required after an independent disagreement.
Geographic groups use 2 km buffers on both footprints (4 km edge separation).
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from rasterio.crs import CRS
from rasterio.errors import CRSError
from rasterio.warp import transform

from scripts.maritime_cloud_inventory import create_only, digest_bytes, json_bytes, parse_gs, require_cloud_vm

CATEGORIES={'marina_small_craft':.35,'commercial_fishing_port':.25,'coastline':.25,'open_water':.15}
TARGET_CORES=7813
CORE_BYTES=7680000


def polygon_wgs84(row):
    from shapely.geometry import Polygon
    bounds=row.get('core_bounds') or row.get('bounds')
    if not bounds or len(bounds)!=4 or not all(math.isfinite(v) for v in bounds):
        raise ValueError('Missing finite geographic footprint')
    left,bottom,right,top=bounds
    if right<=left or top<=bottom: raise ValueError('Invalid footprint bounds')
    # Densify projected edges before crossing CRS boundaries.
    points=[]
    corners=[(left,bottom),(right,bottom),(right,top),(left,top),(left,bottom)]
    for a,b in zip(corners,corners[1:]):
        for index in range(9):
            f=index/9
            points.append((a[0]+(b[0]-a[0])*f,a[1]+(b[1]-a[1])*f))
    x,y=transform(row['crs'],'EPSG:4326',[p[0] for p in points],[p[1] for p in points])
    if not all(math.isfinite(value) for value in x+y): raise ValueError('Footprint CRS transform is unresolved')
    # Avoid a tiny dateline-crossing core becoming a globe-spanning polygon.
    reference=x[0]
    x=[reference+(value-reference+180)%360-180 for value in x]
    polygon=Polygon(zip(x,y))
    if not polygon.is_valid or polygon.is_empty: raise ValueError('Invalid transformed geographic footprint')
    return polygon


def local_pair(a,b):
    from shapely.ops import transform as shape_transform
    longitude=(a.centroid.x+b.centroid.x)/2
    latitude=(a.centroid.y+b.centroid.y)/2
    if abs(a.centroid.x-b.centroid.x)>180:
        longitude=((a.centroid.x+b.centroid.x+360)/2+180)%360-180
    crs=CRS.from_dict(proj='aeqd',lat_0=latitude,lon_0=longitude,datum='WGS84',units='m')
    def project(x,y,z=None):
        xx,yy=transform('EPSG:4326',crs,list(x),list(y))
        return xx,yy
    return shape_transform(project,a),shape_transform(project,b)


def near(a,b,edge_distance=4000):
    # Cheap latitude pruning; longitude is intentionally not used near poles.
    if a.bounds[1]>b.bounds[3]+1 or b.bounds[1]>a.bounds[3]+1: return False
    latitude=max(abs(a.bounds[1]),abs(a.bounds[3]),abs(b.bounds[1]),abs(b.bounds[3]))
    if latitude<85 and a.bounds[2]-a.bounds[0]<180 and b.bounds[2]-b.bounds[0]<180:
        centres=abs((a.centroid.x-b.centroid.x+180)%360-180)
        gap=max(0,centres-(a.bounds[2]-a.bounds[0]+b.bounds[2]-b.bounds[0])/2)
        if gap>edge_distance/(110000*math.cos(math.radians(latitude)))+.02: return False
    aa,bb=local_pair(a,b)
    return aa.distance(bb)<=edge_distance


def geographic_groups(candidates,prior_training,development_sites,buffer_m=2000):
    from shapely.ops import unary_union
    sites=defaultdict(list)
    acquisition_sites=defaultdict(set)
    prior_sites=set()
    for candidate in candidates:
        sites[candidate['site_id']].append(polygon_wgs84(candidate))
        acquisition_sites[candidate['acquisition_id']].add(candidate['site_id'])
    for prior in prior_training:
        site=prior['site_id']
        prior_sites.add(site)
        prior_acquisition=prior.get('acquisition_id') or (
            'ee_catalog:'+str(prior['catalog_id']).rsplit('/',1)[-1] if prior.get('catalog_id') else None)
        if prior_acquisition: acquisition_sites[prior_acquisition].add(site)
        if prior.get('bounds') or prior.get('core_bounds'):
            sites[site].append(polygon_wgs84(prior))
        elif site not in sites:
            raise ValueError('Prior training site absent from candidates requires footprint')
    names=sorted(sites)
    parent={name:name for name in names}
    def find(name):
        while parent[name]!=name:
            parent[name]=parent[parent[name]]
            name=parent[name]
        return name
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[max(a,b)]=min(a,b)
    for identities in acquisition_sites.values():
        identities=sorted(identities)
        for site in identities[1:]: union(identities[0],site)
    shapes={site:unary_union(sites[site]) for site in names}
    for index,a in enumerate(names):
        for b in names[index+1:]:
            if find(a)!=find(b) and near(shapes[a],shapes[b],2*buffer_m): union(a,b)
    components=defaultdict(list)
    for name in names: components[find(name)].append(name)
    groups=[]
    for members in sorted(components.values()):
        groups.append(dict(geographic_group_id=digest_bytes(json_bytes(members))[:24],site_ids=members,
            prior_training=bool(set(members)&prior_sites),development=bool(set(members)&set(development_sites)),
            inspected_for_development=bool(set(members)&set(development_sites)),
            buffer_m_each=buffer_m,maximum_edge_connection_m=2*buffer_m))
    return groups


def resolve_rows(rows,aliases):
    latest={}
    history=defaultdict(list)
    unresolved=[]
    for row in rows:
        value=row.get('canonical_core_id') or row.get('candidate_id') or row.get('sample_id')
        key=aliases.get(value)
        if key is None:
            unresolved.append(dict(id=value,reason='Unknown review candidate identity'))
            continue
        latest[key]=row
        history[key].append(row)
    return latest,history,unresolved


def presence(value):
    if value in ('present','confirmed'): return 'confirmed'
    if value in ('absent','unconfirmed'): return value
    raise ValueError('Explicit presence must be confirmed, absent, or unconfirmed')


def accepted_review(candidate,row):
    if row is None or row.get('decision')!='accept': return 'No explicit accept decision'
    if not isinstance(row.get('reviewer'),str) or not row['reviewer'].strip() or row.get('native_scale_reviewed') is not True: return 'Missing attributable native review'
    for field,required in [('quality_status','pass'),('duplicate_status','clear'),('support_status','verified')]:
        if row.get(field)!=required: return 'Unresolved '+field
    if not isinstance(candidate.get('country'),str) or not candidate['country'].strip(): return 'Undocumented source country'
    if not candidate.get('mask',{}).get('provenance') or not candidate.get('source_sha256') or not candidate.get('native_preview'):
        return 'Missing support or pinned review provenance'
    if row.get('primary_category') not in CATEGORIES: return 'Missing primary maritime category'
    try: vessel,small=presence(row.get('vessel_presence')),presence(row.get('small_craft_presence'))
    except ValueError as exc: return str(exc)
    if small=='confirmed' and vessel!='confirmed': return 'Small craft and vessel presence disagree'
    if not isinstance(row.get('hard_negative'),bool): return 'Explicit hard-negative annotation required'
    if row['hard_negative'] and (vessel!='absent' or small!='absent'): return 'Hard negative has unresolved or confirmed vessels'
    return None


def assign_splits(groups,accepted,frozen_splits=None):
    counts=Counter(c['geographic_group_id'] for c in accepted)
    frozen_splits=frozen_splits or {}
    site_locks={}
    if 'assignment' in frozen_splits:
        for previous in frozen_splits.get('groups',[]):
            for site in previous['site_ids']:
                site_locks[site]=previous['split']
        frozen_splits=frozen_splits['assignment']
    if any(value not in ('train','validation','test','development') for value in frozen_splits.values()):
        raise ValueError('Unknown frozen split assignment')
    mappings={}
    for group in groups:
        gid=group['geographic_group_id']
        fixed=frozen_splits.get(gid)
        inherited={site_locks[site] for site in group['site_ids'] if site in site_locks}
        if len(inherited)>1: raise ValueError('Geographic bridge joins conflicting frozen splits')
        if inherited:
            inherited_split=next(iter(inherited))
            if fixed and fixed!=inherited_split: raise ValueError('Frozen group and site assignment disagree')
            fixed=inherited_split
        forced='development' if group['development'] else 'train' if group['prior_training'] else None
        if fixed and forced and fixed!=forced: raise ValueError('Frozen holdout violates prior training/development isolation')
        mappings[gid]=forced or fixed
    eligible=[g for g in groups if counts[g['geographic_group_id']] and mappings[g['geographic_group_id']] is None]
    eligible.sort(key=lambda g:(counts[g['geographic_group_id']],g['geographic_group_id']))
    total=sum(value for gid,value in counts.items() if mappings[gid]!='development')
    for split,minimum in [('test',10),('validation',5)]:
        existing=[gid for gid,s in mappings.items() if s==split and counts[gid]]
        current=sum(counts[gid] for gid in existing)
        while eligible and (len(existing)<minimum or current<.1*total):
            group=eligible.pop(0)
            gid=group['geographic_group_id']
            mappings[gid]=split
            existing.append(gid)
            current+=counts[gid]
    for group in groups:
        gid=group['geographic_group_id']
        mappings[gid]=mappings[gid] or 'train'
        group.update(split=mappings[gid],selected_cores=counts[gid],selected_core_bytes=counts[gid]*CORE_BYTES)
    return dict(groups=groups,assignment=mappings,
        policy='Frozen connected site groups; 2 km buffer per footprint; prior training/development never held out; holdout diversity precedes byte ratios')


def audit_pass(selected,reviews,audits,audit_history):
    strata=defaultdict(list)
    valid=set()
    failures=[]
    for candidate in selected:
        key=candidate['canonical_core_id']
        row=reviews[key]
        stratum=(row['primary_category'],presence(row['vessel_presence']),presence(row['small_craft_presence']))
        strata[stratum].append(key)
        audit=audits.get(key)
        if audit is None: continue
        fields=['primary_category','hard_negative']
        agrees=(audit.get('decision')=='agree' and audit.get('independent') is True
            and audit.get('native_scale_reviewed') is True and audit.get('reviewer')
            and audit['reviewer']!=row['reviewer'] and all(audit.get(f)==row.get(f) for f in fields))
        try:
            agrees=agrees and all(presence(audit.get(f))==presence(row.get(f)) for f in ['vessel_presence','small_craft_presence'])
        except ValueError: agrees=False
        if any(prior.get('decision')=='disagree' for prior in audit_history.get(key,[])[:-1]) and not audit.get('resolution_reference'):
            agrees=False
        if agrees: valid.add(key)
        else: failures.append(key)
    coverage=[]
    for stratum,members in sorted(strata.items()):
        actual=sum(key in valid for key in members)
        required=math.ceil(len(members)*.1)
        coverage.append(dict(stratum=list(stratum),selected=len(members),required=required,independently_agreed=actual))
    passed=bool(selected) and not failures and all(s['independently_agreed']>=s['required'] for s in coverage)
    return passed,dict(coverage=coverage,unresolved_disagreements=failures,independent_agreements=len(valid),required_fraction=.1)


def select_release(inventory,review_rows,audit_rows,*,prior_training=None,development_sites=None,frozen_splits=None):
    candidates=[]
    geometry_rejections=[]
    for candidate in inventory['candidates']:
        try:
            if not candidate.get('site_id') or not candidate.get('acquisition_id'): raise ValueError('Missing geographic/acquisition identity')
            polygon_wgs84(candidate)
            candidates.append(candidate)
        except (ValueError,KeyError,TypeError,CRSError) as exc:
            geometry_rejections.append(dict(canonical_core_id=candidate.get('canonical_core_id'),reason='Unresolved geography: '+str(exc)))
    groups=geographic_groups(candidates,prior_training or [],development_sites or set())
    by_site={site:g['geographic_group_id'] for g in groups for site in g['site_ids']}
    aliases={}
    for candidate in candidates:
        for alias in [candidate['candidate_id'],candidate['canonical_core_id']]+candidate.get('alternate_candidate_ids',[]):
            if alias in aliases and aliases[alias]!=candidate['canonical_core_id']: raise ValueError('Ambiguous review alias')
            aliases[alias]=candidate['canonical_core_id']
    reviews,_,unresolved_reviews=resolve_rows(review_rows,aliases)
    audits,audit_history,unresolved_audits=resolve_rows(audit_rows,aliases)
    eligible,rejected=[],list(geometry_rejections)
    acquisition_members=defaultdict(list)
    for original in candidates:
        candidate=dict(original,geographic_group_id=by_site[original['site_id']])
        key=candidate['canonical_core_id']
        review=reviews.get(key)
        reason=accepted_review(candidate,review)
        if reason:
            rejected.append(dict(canonical_core_id=key,reason=reason))
            continue
        near_flags={'near_duplicate_review_pending','near_stereo_review_pending','near_identical_observation'}
        if set(candidate.get('flags',[]))&near_flags and not review.get('repeat_observation_reason'):
            rejected.append(dict(canonical_core_id=key,reason='Flagged repeated observation lacks explicit change/diversity reason'))
            continue
        # Same-acquisition pixels are unique regardless of manifest IDs/hashes.
        duplicate=False
        for previous in acquisition_members[candidate['acquisition_id']]:
            if candidate['crs']==previous['crs']:
                x,y,z,t=candidate['core_bounds'];u,v,w,q=previous['core_bounds']
                if min(z,w)-max(x,u)>1e-6 and min(t,q)-max(y,v)>1e-6: duplicate=True;break
            else:
                a,b=local_pair(polygon_wgs84(candidate),polygon_wgs84(previous))
                if a.intersection(b).area>1e-4: duplicate=True;break
        if duplicate:
            rejected.append(dict(canonical_core_id=key,reason='Overlapping same-acquisition original pixels'))
            continue
        candidate.update(primary_category=review['primary_category'],vessel_presence=presence(review['vessel_presence']),
            small_craft_presence=presence(review['small_craft_presence']),hard_negative=review['hard_negative'],
            review_status='accepted_native_review',review_decision=review,final_eligible=False)
        eligible.append(candidate)
        acquisition_members[candidate['acquisition_id']].append(candidate)
    # Freeze group assignment before selecting release cores. Literal
    # development data is retained separately and never fills release quotas.
    preliminary_splits=assign_splits(groups,eligible,frozen_splits)
    inspected_groups={group['geographic_group_id'] for group in groups if group['inspected_for_development']}
    development_selection=[]
    release_eligible=[]
    for candidate in eligible:
        gid=candidate['geographic_group_id']
        candidate['inspected_for_development']=gid in inspected_groups
        candidate['split']=preliminary_splits['assignment'][gid]
        if candidate['split']=='development':
            development_selection.append(candidate)
        else:
            release_eligible.append(candidate)
    # Deterministic balanced interleave, bounded by full-release group cap.
    queues=defaultdict(list)
    for candidate in sorted(release_eligible,key=lambda c:c['canonical_core_id']): queues[candidate['primary_category']].append(candidate)
    selected=[]
    category_counts=Counter()
    group_counts=Counter()
    country_counts=Counter()
    vessel_selected=small_selected=negative_selected=0
    while len(selected)<TARGET_CORES:
        available=[category for category in CATEGORIES if queues[category]]
        if not available: break
        category=min(available,key=lambda key:(category_counts[key]/CATEGORIES[key],key))
        next_count=len(selected)+1
        def priority(candidate):
            score=(3*(negative_selected/next_count<.05 and candidate['hard_negative'])
                +2*(small_selected/next_count<.15 and candidate['small_craft_presence']=='confirmed')
                +(vessel_selected/next_count<.3 and candidate['vessel_presence']=='confirmed'))
            return (-score,group_counts[candidate['geographic_group_id']],country_counts[candidate['country']],candidate['canonical_core_id'])
        index=min(range(len(queues[category])),key=lambda i:priority(queues[category][i]))
        candidate=queues[category].pop(index)
        gid=candidate['geographic_group_id']
        if group_counts[gid]>=math.floor(TARGET_CORES*.1):
            rejected.append(dict(canonical_core_id=candidate['canonical_core_id'],reason='Full-release site concentration cap'))
            continue
        selected.append(candidate)
        category_counts[category]+=1
        group_counts[gid]+=1
        country_counts[candidate['country']]+=1
        vessel_selected+=candidate['vessel_presence']=='confirmed'
        small_selected+=candidate['small_craft_presence']=='confirmed'
        negative_selected+=candidate['hard_negative']
    splits=assign_splits(groups,selected,preliminary_splits['assignment'])
    development_counts=Counter(candidate['geographic_group_id'] for candidate in development_selection)
    for group in groups:
        group['development_cores']=development_counts[group['geographic_group_id']]
        group['development_core_bytes']=group['development_cores']*CORE_BYTES
    for candidate in selected: candidate['split']=splits['assignment'][candidate['geographic_group_id']]
    count=len(selected)
    size=count*CORE_BYTES
    category_shares={category:category_counts[category]/count if count else 0 for category in CATEGORIES}
    vessels=sum(c['vessel_presence']=='confirmed' for c in selected)
    small=sum(c['small_craft_presence']=='confirmed' for c in selected)
    negatives=sum(c['hard_negative'] for c in selected)
    active_groups={c['geographic_group_id'] for c in selected}
    countries={c['country'] for c in selected}
    split_counts=Counter(c['split'] for c in selected)
    split_groups={split:sum(g['split']==split and g['selected_cores']>0 for g in groups) for split in ['train','validation','test','development']}
    semantic_audit,audit_report=audit_pass(selected,reviews,audits,audit_history)
    _,development_audit_report=audit_pass(development_selection,reviews,audits,audit_history)
    requirements=dict(size=60000000000<=size<=60100000000,core_count=count==TARGET_CORES,
        category_balance=bool(count) and all(abs(category_shares[key]-target)<=.05+1e-12 for key,target in CATEGORIES.items()),
        confirmed_vessels=bool(count) and vessels/count>=.3,confirmed_small_craft=bool(count) and small/count>=.15,
        hard_negatives=bool(count) and .03<=negatives/count<=.07,
        independent_sites=len(active_groups)>=30,countries=len(countries)>=3,
        validation_sites=split_groups['validation']>=5,test_sites=split_groups['test']>=10,
        site_concentration=bool(count) and max(group_counts.values())/count<=.1+1e-12,
        split_category_coverage=all({c['primary_category'] for c in selected if c['split']==split}==set(CATEGORIES) for split in ['train','validation','test']),
        semantic_audit=semantic_audit,final_holdout_isolation=not any(g['split'] in ('validation','test') and (g['prior_training'] or g['development']) for g in groups),
        prior_split_history_verified=prior_training is not None and development_sites is not None,
        geographic_split_integrity=not geometry_rejections,all_source_quality_support_reviewed=bool(count))
    passed=all(requirements.values())
    # This selector cannot attest stored imagery, masks, or final source-window
    # byte audits. These downstream numerical gates remain mandatory.
    requirements['independent_pixel_size_audit']=False
    requirements['published_mother_integrity_audit']=False
    return dict(schema_version=1,status='PROVISIONAL_SELECTION' if selected or development_selection else 'NO_REVIEWED_SELECTION',
        selection=selected,development_selection=development_selection,splits=splits,gates=dict(requirements=requirements,metadata_selection_passed=passed,
        full_release_passed=False,ready_marker_authorized=False,missing=[key for key,value in requirements.items() if not value]),
        size_audit=dict(accepted_candidate_count=count,accepted_unique_core_bytes=size,
            development_candidate_count=len(development_selection),development_unique_core_bytes=len(development_selection)*CORE_BYTES,
            development_counted_in_release_quotas=False,
            target_core_count=TARGET_CORES,target_min_bytes=60000000000,target_max_bytes=60100000000,
            target_shortfall_bytes=max(0,60000000000-size),category_counts=dict(category_counts),category_shares=category_shares,
            confirmed_vessel_count=vessels,confirmed_small_craft_count=small,hard_negative_count=negatives,
            connected_groups_with_selected_cores=len(active_groups),all_connected_group_count=len(groups),
            countries=sorted(countries),split_core_counts=dict(split_counts),split_group_counts=split_groups,
            max_group_share=max(group_counts.values())/count if count else 0,
            hard_negative_target_fraction=.05,hard_negative_tolerance_fraction=.02,
            native_review_accepted_before_balancing=len(eligible),release_eligible_before_balancing=len(release_eligible),
            byte_basis='Train/validation/test only: 1600*1600*3 native uint8 core; development/halo/duplicates/child arrays excluded'),
        audit_report=audit_report,development_audit_report=development_audit_report,rejections=rejected,unknown_review_references=unresolved_reviews,
        unknown_audit_references=unresolved_audits,review_log_sha256=digest_bytes(json_bytes(review_rows)),
        audit_log_sha256=digest_bytes(json_bytes(audit_rows)),inventory_sha256=digest_bytes(json_bytes(inventory)))


def load_rows(path):
    text=Path(path).read_text(encoding='utf-8-sig')
    return json.loads(text) if text.lstrip().startswith('[') else [json.loads(line) for line in text.splitlines() if line.strip()]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('inventory','reviews','audits','output'): parser.add_argument('--'+name,required=True)
    for name in ('prior-training','development-sites','frozen-splits','publish','project'): parser.add_argument('--'+name)
    args=parser.parse_args()
    inventory=json.loads(Path(args.inventory).read_text(encoding='utf-8-sig'))
    prior=json.loads(Path(args.prior_training).read_text(encoding='utf-8-sig')) if args.prior_training else None
    development=set(json.loads(Path(args.development_sites).read_text(encoding='utf-8-sig'))) if args.development_sites else None
    frozen=json.loads(Path(args.frozen_splits).read_text(encoding='utf-8-sig')) if args.frozen_splits else None
    result=select_release(inventory,load_rows(args.reviews),load_rows(args.audits),prior_training=prior,
        development_sites=development,frozen_splits=frozen)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    artifacts={'selection.json':result,'development-selection.json':result['development_selection'],'splits.json':result['splits'],'size-audit.json':result['size_audit'],
        'rejections.json':result['rejections'],'gate-report.json':result['gates']}
    if args.publish:
        require_cloud_vm()
        from google.cloud import storage
        client=storage.Client(project=args.project)
        bucket,prefix=parse_gs(args.publish)
    for name,value in artifacts.items():
        payload=json_bytes(value)
        (out/name).write_bytes(payload)
        if args.publish: create_only(client.bucket(bucket).blob(prefix.rstrip('/')+'/'+name),payload)
    print(json.dumps(dict(status=result['status'],size_audit=result['size_audit'],gates=result['gates'])),flush=True)


if __name__=='__main__': main()

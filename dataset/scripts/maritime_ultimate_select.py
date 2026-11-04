"""Metadata-only mixed-source selection, honest byte targets and frozen geography."""
import argparse
from collections import Counter,defaultdict,deque
import json
from pathlib import Path
from scripts.maritime_cloud_inventory import digest_bytes,json_bytes
from scripts.maritime_release_select import geographic_groups,assign_splits
from scripts.maritime_fast_reuse_inventory import DEVELOPMENT


def native_bytes(row):
    return row['source']['useful_native_rgb_bytes'] if row['source_kind']=='maxar' else 7680000


def count_bytes(rows):
    return sum(native_bytes(row) for row in rows)


def byte_balanced(rows,minimum=60000000000,maximum=60100000000,site_fraction=.1):
    queues=defaultdict(deque);seen=set()
    for row in sorted(rows,key=lambda r:r['sample_id']):
        if row['sample_id'] in seen: continue
        seen.add(row['sample_id']);queues[row['site_id']].append(row)
    counts=Counter();result=[];total=0
    while any(queues.values()) and total<minimum:
        for site in sorted(queues,key=lambda s:(counts[s],s)):
            if not queues[site]:continue
            row=queues[site].popleft();size=native_bytes(row)
            if counts[site]+size>minimum*site_fraction or total+size>maximum:continue
            result.append(row);counts[site]+=size;total+=size
            if total>=minimum:break
    return result


def physical_key(row):
    acq=row['acquisition_id'].removeprefix('ee_catalog:')
    left,bottom,_,_=row['core_bounds']
    # World-cell identity survives the permitted half-native-pixel snap.
    return (acq,row['crs'],round(left/1280),round(bottom/1280))


def disjoint_native_rows(rows):
    accepted=[];seen=set();capture_crs={};cells={}
    for row in rows:
        key=physical_key(row)
        if key in seen:continue
        capture,crs,cx,cy=key
        # A single original CRS per capture also prevents zone-edge duplicates.
        if capture in capture_crs and capture_crs[capture]!=crs:continue
        bounds=row['core_bounds'];a,b,c,d=bounds;overlap=False
        for x in range(cx-1,cx+2):
            for y in range(cy-1,cy+2):
                old=cells.get((capture,crs,x,y))
                if old:
                    e,f,g,h=old
                    # 0.1mm tolerates float32 encoding of the nominal0.8m grid;
                    # shifted native export phases with real overlap are rejected.
                    if min(c,g)-max(a,e)>1e-4 and min(d,h)-max(b,f)>1e-4:overlap=True
        if overlap:continue
        capture_crs[capture]=crs;seen.add(key);cells[key]=bounds;accepted.append(row)
    return accepted


def assign_mixed_splits(groups,rows,frozen=None):
    # Delegate all historical lock/conflict validation to the established rule.
    # Its count ratios assume equal-sized native cores; mixed sources use bytes.
    result=assign_splits(groups,[],frozen)
    frozen=frozen or {}
    fixed_groups=frozen.get('assignment',frozen)
    fixed_sites={site for group in frozen.get('groups',[]) for site in group['site_ids']}
    sizes=Counter();counts=Counter()
    for row in rows:
        gid=row['geographic_group_id'];sizes[gid]+=native_bytes(row);counts[gid]+=1
    assignment=result['assignment'];eligible=[]
    for group in groups:
        gid=group['geographic_group_id']
        fixed=(group['prior_training'] or group['development'] or gid in fixed_groups
            or bool(set(group['site_ids'])&fixed_sites))
        if sizes[gid] and not fixed:eligible.append(group)
    eligible.sort(key=lambda group:(sizes[group['geographic_group_id']],group['geographic_group_id']))
    total=sum(size for gid,size in sizes.items() if assignment[gid]!='development')
    free={group['geographic_group_id'] for group in eligible}
    for split,minimum in [('test',10),('validation',5)]:
        held=[gid for gid,s in assignment.items() if s==split and sizes[gid] and gid not in free]
        current=sum(sizes[gid] for gid in held)
        while eligible and (len(held)<minimum or current<total*.1):
            gid=eligible.pop(0)['geographic_group_id']
            assignment[gid]=split;held.append(gid);current+=sizes[gid];free.remove(gid)
    for group in groups:
        gid=group['geographic_group_id']
        group.update(split=assignment[gid],selected_cores=counts[gid],selected_core_bytes=sizes[gid])
    result['policy']+='; mixed native holdout ratios use decoded useful RGB bytes'
    return result


def quality_rejected_keys(rows,documents):
    identities={row['sample_id']:physical_key(row) for row in rows}
    rejected=set()
    for document in documents:
        decisions=document if isinstance(document,list) else document['reviews']
        for decision in decisions:
            if decision['decision']=='reject':
                identity=decision['sample_id']
                if identity not in identities:raise ValueError('Quality rejection identity absent from source inventories')
                rejected.add(identities[identity])
    return rejected


def build(args):
    rows=[];physical=set();inputs=[]
    for path in args.inventory:
        data=Path(path).read_bytes();document=json.loads(data);inputs.append(dict(path=str(path),sha256=digest_bytes(data)))
        for original in document['candidates']:
            if original.get('source_kind')=='skysat_raw_window': row=dict(original)
            elif document.get('schema')=='maxar-mixed-native-v1':
                port=original['nearest_port'];source=original['source']
                site='maxar-port-'+str(port['port_id'])
                row=dict(sample_id=original['sample_id'],source_kind='maxar',source=original,site_id=site,
                    geographic_group_id=site,acquisition_id=source['catalog_id'],crs='EPSG:'+str(source['epsg']),
                    core_bounds=original['physical_core_bounds'],country=port['country_code_as_published'],
                    trust_basis='automated_source_metadata_original_masks_and_coastal_port_prior',
                    vessel_presence='unconfirmed',small_craft_presence='unconfirmed',requires_stratified_native_audit=True)
            else:raise ValueError('Unsupported inventory schema; original source identity required')
            rows.append(row)
    quality_documents=[]
    for path in getattr(args,'quality_exclusions',[]):
        data=Path(path).read_bytes();quality_documents.append(json.loads(data))
        inputs.append(dict(path=str(path),sha256=digest_bytes(data),role='sample_quality_exclusions'))
    quality_keys=quality_rejected_keys(rows,quality_documents)
    rows=disjoint_native_rows(sorted(rows,key=lambda r:r['sample_id']))
    # Source-proven individual rejection overrides collection trust.
    rejected=set()
    if args.review_inventory:
        known=json.loads(Path(args.review_inventory).read_text())['candidates']
        aliases={}
        for c in known:
            key=physical_key(c)
            for identity in [c['canonical_core_id'],c['candidate_id']]+c.get('alternate_candidate_ids',[]): aliases[identity]=key
        latest={}
        for path in args.reviews:
            ledger=json.loads(Path(path).read_text());ledger=ledger if isinstance(ledger,list) else ledger.get('reviews',[])
            for decision in ledger:
                identity=decision.get('canonical_core_id') or decision.get('candidate_id') or decision.get('sample_id')
                if identity in aliases:latest[aliases[identity]]=decision
        rejected={key for key,d in latest.items() if d['decision']=='reject'}
        for row in rows:
            decision=latest.get(physical_key(row))
            if decision and decision.get('decision')=='accept':
                row['review_decision']=decision
                for field in ('primary_category','vessel_presence','small_craft_presence','hard_negative'):
                    if field in decision:row[field]=decision[field]
    rejected.update(quality_keys)
    rows=[row for row in rows if physical_key(row) not in rejected]
    prior=json.loads(Path(args.prior_training).read_text())
    groups=geographic_groups(rows,prior,DEVELOPMENT)
    groupmap={site:g for g in groups for site in g['site_ids']}
    rows=[r for r in rows if not groupmap[r['site_id']]['development']]
    for row in rows:row['geographic_group_id']=groupmap[row['site_id']]['geographic_group_id']
    chosen=byte_balanced(rows,args.minimum_bytes,args.maximum_bytes,args.site_fraction)
    frozen=json.loads(Path(args.frozen_splits).read_text()) if args.frozen_splits else None
    splits=assign_mixed_splits(groups,chosen,frozen)
    for row in chosen:row['split']=splits['assignment'][row['geographic_group_id']]
    # Override old selector's fixed .8 accounting with observed mixed bytes.
    for group in splits['groups']:
        group['selected_core_bytes']=count_bytes([r for r in chosen if r['geographic_group_id']==group['geographic_group_id']])
    total=count_bytes(chosen);sitebytes=Counter()
    for row in chosen:sitebytes[row['site_id']]+=native_bytes(row)
    counts=Counter(r['split'] for r in chosen)
    splitgroups={s:len({r['geographic_group_id'] for r in chosen if r['split']==s}) for s in counts}
    gates=dict(native_size_target=args.minimum_bytes<=total<=args.maximum_bytes,
        site_fraction=bool(total) and max(sitebytes.values(),default=0)<=total*args.site_fraction,
        required_sites=len(sitebytes)>=30,required_countries=len({r['country'] for r in chosen if r.get('country')})>=3,
        validation_groups=splitgroups.get('validation',0)>=5,test_groups=splitgroups.get('test',0)>=10,
        strict_semantic_category_and_vessel_quotas=False,independent_stratified_native_quality_audit=False,
        independent_original_source_window_pixel_audit=False)
    result=dict(schema='ultimate-selection-v1',selector_sha256=digest_bytes(Path(__file__).read_bytes()),selection=chosen,splits=splits,gates=gates,
        review_policy=dict(version='trusted-curated-expansion-v1',authority='user requested curated reuse and confirmed existing imagery quality',
            existing_collection='trust reused with automated masks/resolution/dedup plus sample audit; specific recorded rejects remain excluded',
            maxar='source metadata and port prior only; individual semantics unconfirmed',
            exhaustive_individual_native_review=False,known_vessel_absence_never_inferred=True),
        inputs=inputs,summary=dict(sample_count=len(chosen),useful_native_rgb_bytes=total,site_bytes=dict(sitebytes),
            split_counts=dict(counts),split_group_counts=splitgroups,candidate_unique_bytes=count_bytes(rows),
            source_kind_counts=dict(Counter(r['source_kind'] for r in chosen)),individual_rejections=len(rejected)),
        status='FROZEN_TRAINING_BUILD_SELECTION',strict_full_target_passed=all(value is True for value in gates.values()))
    Path(args.output).write_bytes(json_bytes(result));print(json.dumps(result['summary'],sort_keys=True))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prior-training',required=True);p.add_argument('--frozen-splits')
    p.add_argument('--review-inventory');p.add_argument('--reviews',nargs='*',default=[])
    p.add_argument('--quality-exclusions',nargs='*',default=[])
    p.add_argument('--minimum-bytes',type=int,default=60000000000);p.add_argument('--maximum-bytes',type=int,default=60100000000)
    p.add_argument('--site-fraction',type=float,default=.1)
    build(p.parse_args())

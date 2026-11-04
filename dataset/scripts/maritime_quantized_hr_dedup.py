"""Preselect unique actual uint8 HR; synthetic degradation is not new imagery.

Callers verify the pinned GCE census and scoring witnesses before calling this
metadata-only helper. Final release rechecks every decoded hash independently.
"""
import copy
from datetime import date
import math
import re


def _hash(row):
    digest=row.get('hr_uint8_chw_sha256',row.get('qualification',{}).get('hr_uint8_chw_sha256'))
    if not isinstance(digest,str) or not re.fullmatch('[0-9a-f]{64}',digest):
        raise ValueError('Actual quantized CHW HR SHA-256 required')
    return digest


def deduplicate_quantized_hr(candidates,baseline_census):
    """Preserve all baseline rows; retain the best nominated new image per hash."""
    old=baseline_census.get('rows',[])
    if (baseline_census.get('schema')!='maritime-baseline-quantized-hr-census-v1'
            or baseline_census.get('complete') is not True or not old
            or baseline_census.get('sample_count')!=len(old)
            or len({r['sample_id'] for r in old})!=len(old)):
        raise ValueError('Complete verified baseline quantized HR census required')
    old_hashes={_hash(r) for r in old};old_ids={r['sample_id'] for r in old}
    candidate_hashes={}
    for row in candidates:
        digest=_hash(row);identity=row['sample_id']
        if identity in candidate_hashes and candidate_hashes[identity]!=digest:
            raise ValueError('Conflicting quantized HR hashes for one sample identity')
        candidate_hashes[identity]=digest
    seen_ids=set();seen_hashes={};kept=[];rejected=[]
    def order(row):
        priority=row.get('screening',{}).get('priority')
        if isinstance(priority,bool) or not isinstance(priority,int) or not 0<=priority<=5:
            raise ValueError('Explicit qualified priority 0..5 required')
        return (priority,row.get('qualification',{}).get('tier')!='assistant_actual_image_review',row['sample_id'])
    for row in sorted(candidates,key=order):
        identity=row['sample_id'];digest=_hash(row)
        if row.get('screening',{}).get('eligible') is not True:
            reason='screening_not_qualified'
        elif identity in old_ids or identity in seen_ids:
            reason='duplicate_sample_identity'
        elif digest in old_hashes:
            reason='duplicate_preserved_baseline_uint8_hr'
        elif digest in seen_hashes:
            reason='duplicate_added_uint8_hr'
        else:
            seen_ids.add(identity);seen_hashes[digest]=identity;kept.append(copy.deepcopy(row));continue
        rejected.append(dict(sample_id=identity,reason=reason,hr_uint8_chw_sha256=digest,
            kept_added_id=seen_hashes.get(digest)))
    return kept,rejected


def deduplicate_aerial_day_cells(candidates):
    """Conservatively retain one added aerial core per known calendar day.

    Parent mosaics can overlap. Their differing bytes do not establish new
    acquisition information. Original flight time remains unknown; this key is
    a conservative exclusion, not a certified same-flight assertion. Unknown
    NOAA dates are never combined. Temporal revisits remain distinct.
    """
    def order(row):
        priority=row.get('screening',{}).get('priority')
        if isinstance(priority,bool) or not isinstance(priority,int) or not 0<=priority<=5:
            raise ValueError('Explicit qualified priority 0..5 required')
        return priority,row.get('qualification',{}).get('tier')!='assistant_actual_image_review',row['sample_id']
    kept=[];rejected=[];seen={}
    for row in sorted(candidates,key=order):
        source=row.get('source',{})
        day=source.get('acquisition_day')
        if row.get('source_kind')!='aerial_rgb' or day is None:
            kept.append(copy.deepcopy(row));continue
        try:
            if not isinstance(day,str) or date.fromisoformat(day).isoformat()!=day:
                raise ValueError('Noncanonical date')
        except (TypeError,ValueError) as error:
            raise ValueError('Verified canonical aerial acquisition calendar day required') from error
        bounds=source.get('physical_core_bounds',[]);crs=source.get('output_crs');family=source.get('source_type')
        if (not isinstance(crs,str) or not crs or family not in {'naip_aerial','noaa_aerial'}
                or len(bounds)!=4 or any(isinstance(v,bool) or not isinstance(v,(int,float))
                                         or not math.isfinite(v) for v in bounds)
                or not math.isclose(bounds[2]-bounds[0],1280,abs_tol=1e-4,rel_tol=0)
                or not math.isclose(bounds[3]-bounds[1],1280,abs_tol=1e-4,rel_tol=0)):
            raise ValueError('Original aerial CRS and 1280m physical core required')
        key=(family,day,crs,*bounds)
        if key in seen:
            rejected.append(dict(sample_id=row['sample_id'],reason='same_aerial_calendar_day_core',
                kept_added_id=seen[key],source_family=family,acquisition_day=day,
                original_crs=crs,physical_core_bounds=list(bounds)))
        else:
            seen[key]=row['sample_id'];kept.append(copy.deepcopy(row))
    return kept,rejected

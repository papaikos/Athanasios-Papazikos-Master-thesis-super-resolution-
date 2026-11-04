"""Freeze a reproducible visual-audit sample of an exact selected population.

This is metadata only. It nominates images for actual review and grants no
quality or READY status. Callers verify and pin the selection before rendering.
The deterministic sample is not a calibrated population accuracy estimate.
"""
import hashlib
import json
import math
from collections import defaultdict

from scripts.maritime_expansion_audit import _identity


def selected_audit_plan(additions, *, fraction=.05, strata_fields=()):
    if isinstance(fraction,bool) or not isinstance(fraction,(int,float)) or not 0<fraction<=1:
        raise ValueError('Review fraction must lie in (0,1]')
    fields=sorted(set(strata_fields))
    rows=[];seen=set()
    for item in additions:
        _identity(item)
        if item['sample_id'] in seen:
            raise ValueError('Duplicate selected sample identity')
        seen.add(item['sample_id'])
        row={key:item[key] for key in ('sample_id','source','candidate_sha256','hr_float32_sha256')}
        for key in fields:
            if not isinstance(item.get(key),str) or not item[key]:
                raise ValueError('Requested stratum requires explicit metadata: '+key)
            row[key]=item[key]
        rows.append(row)
    if not rows:
        raise ValueError('Nonempty selected additions required')
    rows.sort(key=lambda r:r['sample_id'])
    raw=json.dumps(rows,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')
    fingerprint=hashlib.sha256(raw).hexdigest()
    populations=defaultdict(list)
    for row in rows:
        populations[json.dumps([row['source'],'source'],separators=(',',':'))].append(row)
        for field in fields:
            populations[json.dumps([row['source'],field,row[field]],separators=(',',':'))].append(row)
    def rank(row):
        # Stable per-image order avoids reshuffling the entire sample when an
        # observed rejection is removed. The separate fingerprint binds the
        # exact population against which the sample must be rechecked.
        witness=json.dumps(['maritime-selected-visual-audit-v1',row['source'],row['sample_id'],
                            row['candidate_sha256'],row['hr_float32_sha256']],
                           separators=(',',':')).encode('utf-8')
        return hashlib.sha256(witness).hexdigest(),row['sample_id']
    groups={};required=set()
    for name,items in sorted(populations.items()):
        count=math.ceil(len(items)*fraction)
        ids=[r['sample_id'] for r in sorted(items,key=rank)[:count]]
        required.update(ids)
        groups[name]=dict(selected_count=len(items),required_review_count=count,required_sample_ids=ids)
    return dict(schema='maritime-selected-visual-audit-plan-v1',selection_sha256=fingerprint,
        selection_count=len(rows),review_fraction=fraction,strata_fields=fields,
        sampling_method='Stable SHA-256 order of source, sample identity and exact candidate/HR hashes; separate selected-population fingerprint',
        groups=groups,required_sample_ids=sorted(required),dataset_ready=False,
        population_accuracy_estimate=False,actual_image_review_still_required=True)

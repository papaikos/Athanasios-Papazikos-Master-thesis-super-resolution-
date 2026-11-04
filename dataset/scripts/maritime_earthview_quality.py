"""Descriptive SR-target diagnostics; review flags never constitute acceptance."""
import argparse
import io
import json
import hashlib
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter,label


def review_policy(metrics, explicitly_observed_exclusion=False):
    flags=list(metrics['flags'])
    if metrics['luminance_high_frequency_spectral_flatness']>.35:
        flags.append('HIGH_SPECTRAL_FLATNESS_REVIEW')
    if explicitly_observed_exclusion:status='EXCLUDE_OBSERVED_QUALITY_DEFECT'
    elif flags:status='EXCLUDE_CONSERVATIVE_DIAGNOSTIC_FLAGS'
    else:status='REQUIRES_FINAL_STRATIFIED_REVIEW'
    return dict(status=status,flags=flags,quality_accepted=False,final_eligible=False,
        caveat='Conservative research selection policy; absence of flags does not establish quality. Assistant audit is not independent ground truth.')


def validate_explicit_revision(audit, prior_exclusions):
    ids=audit.get('explicit_excluded_sample_ids',[])
    if len(ids)!=31 or len(set(ids))!=31 or not set(prior_exclusions).issubset(ids):
        raise ValueError('Revision requires31 unique exclusions preserving every prior exclusion')
    return set(ids)


def revise_quality_result(prior, measurement_pin, config_pin, excluded):
    if prior.get('schema')!='earthview-provisional-quality-result-v1':
        raise ValueError('Original v1 measurement record required')
    revised=dict(prior,schema='earthview-provisional-quality-result-v2',
                 prior_measurement_pin=measurement_pin,prior_config_pin=prior['config_pin'],
                 config_pin=config_pin,policy=review_policy(prior['metrics'],prior['sample_id'] in excluded))
    return revised


def validate_development_revision(audit, prior_exclusions):
    """Validate the pinned second development audit without weakening prior locks."""
    previous=audit.get('previous_excluded_sample_ids',[])
    added=audit.get('newly_excluded_sample_ids',[])
    excluded=audit.get('explicit_excluded_sample_ids',[])
    held=audit.get('marine_content_unresolved_sample_ids',[])
    if (audit.get('schema')!='root-earthview-development-actual-hr-visual-audit-v1'
        or audit.get('quality_accepted_count')!=0 or audit.get('final_eligible') is not False
        or len(previous)!=31 or len(set(previous))!=31 or set(previous)!=set(prior_exclusions)
        or len(added)!=44 or len(set(added))!=44 or set(added)&set(previous)
        or len(excluded)!=75 or len(set(excluded))!=75 or set(excluded)!=set(previous)|set(added)
        or len(held)!=25 or len(set(held))!=25 or set(held)&set(excluded)):
        raise ValueError('Development revision requires exact preserved31 plus44 exclusions and25 disjoint unresolved marine holds')
    return set(excluded),set(held)


def development_review_policy(metrics, sample_id, excluded, held):
    policy=review_policy(metrics,sample_id in excluded)
    if sample_id in held:
        if sample_id in excluded:raise ValueError('Marine holds and locked quality exclusions must be disjoint')
        policy=dict(policy,status='HOLD_MARINE_CONTENT_UNRESOLVED',
                    caveat='Observed land-dominant content requires marine-content adjudication; numeric flags do not establish acceptance.')
    return policy


def bright_neutral_diagnostics(hr):
    a=np.asarray(hr,dtype=np.float64)
    if a.shape!=(3,512,512) or not np.isfinite(a).all() or a.min()<0 or a.max()>1:
        raise ValueError('Finite canonical RGB 512 HR in [0,1] required')
    neutral=(a.min(axis=0)>=.7)&((a.max(axis=0)-a.min(axis=0))<=.1)
    components,count=label(neutral,structure=np.ones((3,3),dtype=np.uint8))
    largest=float(np.bincount(components.ravel())[1:].max()/neutral.size) if count else 0.
    patches=neutral.reshape(16,32,16,32).mean(axis=(1,3))
    patch_fraction=float(np.mean(patches>=.75))
    flags=['EXTENSIVE_BRIGHT_NEUTRAL_REGION_REVIEW'] if largest>=.1 or patch_fraction>=.1 else []
    return dict(bright_neutral_fraction=float(neutral.mean()),largest_bright_neutral_component_fraction=largest,
                bright_neutral_patch_fraction=patch_fraction,flags=flags,quality_accepted=False,
                definition='All RGB channels>=0.7; max-minus-min RGB<=0.1. Eight-connected components;32pixel patches>=75percent neutral.',
                caveat='Descriptive cloud-like brightness only; bright buildings, foam or sediment may also satisfy it. No calibrated cloud labels or automatic acceptance.')


def bounded_ordered_map(function, iterable, *, workers=4, max_pending=8):
    """Bound submitted payloads/futures independently of population size."""
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    if workers<1 or max_pending<workers:raise ValueError('Positive workers and max_pending>=workers required')
    iterator=iter(iterable)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending=deque()
        for _ in range(max_pending):
            try:item=next(iterator)
            except StopIteration:break
            pending.append(pool.submit(function,item))
        while pending:
            yield pending.popleft().result()
            try:item=next(iterator)
            except StopIteration:continue
            pending.append(pool.submit(function,item))


def measurement_function_fingerprint(source):
    """Fingerprint computation AST separately from adjacent selector additions."""
    import ast
    functions=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='quality_diagnostics']
    if len(functions)!=1:raise ValueError('Exactly one quality_diagnostics function required')
    return hashlib.sha256(ast.dump(functions[0],include_attributes=False).encode()).hexdigest()


def quality_candidate_projection(candidate):
    if candidate.get('source_kind')!='earthview' or candidate.get('final_eligible') is not False:
        raise ValueError('Only provisional EarthView candidate metadata supported')
    return {key:candidate[key] for key in ('sample_id','source_kind','final_eligible','hr','port','acquisition_identity')}


def select_development_quality_review(rows, excluded, model_hints=None):
    """Purposeful120+40 development audit; never a final probability sample."""
    from collections import Counter
    model_hints=model_hints or {};by_id={r['sample_id']:r for r in rows}
    if len(by_id)!=len(rows):raise ValueError('Unique quality observations required')
    for identity,hint in model_hints.items():
        if identity not in by_id or hint['candidate_pin']!=by_id[identity]['candidate_pin']:
            raise ValueError('Model hint requires exact candidate pin')
    def risk(row):
        m=row['metrics']
        return max(m['chromatic_highpass_rms']/.02,m['luminance_high_frequency_spectral_flatness']/.35,
                   .15/max(m['structure_variance_fraction_sigma4'],1e-15))
    def tie(row):return hashlib.sha256(('earthview-development-review-v1:'+row['sample_id']).encode()).hexdigest()
    eligible=[r for r in rows if r['sample_id'] not in excluded]
    clear=sorted((r for r in eligible if r['policy']['status']=='REQUIRES_FINAL_STRATIFIED_REVIEW'),key=lambda r:(risk(r),tie(r)))
    flagged=sorted((r for r in eligible if r['policy']['status']=='EXCLUDE_CONSERVATIVE_DIAGNOSTIC_FLAGS'),key=lambda r:(risk(r),tie(r)))
    if len(clear)<120 or len(flagged)<40:raise ValueError('At least120 unflagged and40 eligible flagged observations required')
    countries=Counter();ports=Counter();dates=Counter();selected=[]
    def choose(pool,count,stratum,quantile=None):
        remaining=list(pool)
        for _ in range(count):
            def key(row):
                p=row['port'];date=row['acquisition_identity'][0][:10]
                hint=model_hints.get(row['sample_id'],{})
                priority=0 if hint.get('predicted_medium_large') else (1 if hint.get('predicted_harbor') else (2 if p['size'] in ('L','M') else 3))
                return (priority if stratum=='flagged_borderline' else 0,countries[p['country']],ports[p['id']],dates[date],tie(row))
            row=min(remaining,key=key);remaining.remove(row);p=row['port'];date=row['acquisition_identity'][0][:10]
            reason=dict(risk_score=risk(row),risk_quantile_bin=quantile,country_count_before=countries[p['country']],
                        port_count_before=ports[p['id']],date_count_before=dates[date],harbor_size_prior=p['size'],
                        model_hint=model_hints.get(row['sample_id']),model_ground_truth=False)
            selected.append(dict(row,development_review_stratum=stratum,selection_reason=reason))
            countries[p['country']]+=1;ports[p['id']]+=1;dates[date]+=1
    for q in range(4):choose(clear[len(clear)*q//4:len(clear)*(q+1)//4],30,'unflagged',q)
    closest=flagged[:max(40,len(flagged)//5)]
    identities={r['sample_id'] for r in closest}
    closest.extend(r for r in flagged if r['sample_id'] not in identities and
                   (model_hints.get(r['sample_id'],{}).get('predicted_medium_large') or model_hints.get(r['sample_id'],{}).get('predicted_harbor')))
    choose(closest,40,'flagged_borderline')
    return selected


def select_actual_hr_review(audit):
    mandatory=[row for row in audit['rows'] if row['status'] in ('EXCLUDE_OBSERVED_QUALITY_DEFECT','QUALITY_REVIEW_PENDING')]
    controls=[row for row in audit['rows'] if row['status']=='NO_OBVIOUS_OVERVIEW_DEFECT']
    controls.sort(key=lambda row:hashlib.sha256(('earthview-hr-controls-v1:'+row['sample_id']).encode()).hexdigest())
    if len(mandatory)!=31 or len(controls)<10:raise ValueError('Original17 rejected plus14 pending and at least10 overview controls required')
    return sorted(mandatory+controls[:10],key=lambda row:row['annotation_id'])


def canonical_source_diagnostics(store,row,seed):
    """Replay existing SkySat/Maxar canonical pipeline; flags remain descriptive."""
    if row.get('source_kind') not in ('skysat','skysat_raw_window','maxar'):
        raise ValueError('Existing SkySat/Maxar canonical dispatch required')
    from scripts.maritime_ultimate_release import process_sample
    common,context,_,_=process_sample(store,row,seed)
    if context.shape!=(3,544,544) or context.dtype!=np.float32:
        raise ValueError('Existing float32 canonical544 context required')
    hr=np.ascontiguousarray(context[:,16:-16,16:-16])
    metrics=quality_diagnostics(hr)
    flags=list(metrics['flags'])
    if metrics['luminance_high_frequency_spectral_flatness']>.35:
        flags.append('HIGH_SPECTRAL_FLATNESS_REVIEW')
    evidence=dict(sample_id=common['sample_id'],source_kind=row['source_kind'],
        status='DESCRIPTIVE_SOURCE_CALIBRATION_PENDING',flags=flags,
        quality_accepted=False,final_eligible=False,
        caveat='Measurements replay existing shared process_sample. EarthView exploratory thresholds are not calibrated for this source.')
    return hr,metrics,evidence


def quality_diagnostics(hr):
    a=np.asarray(hr,dtype=np.float64)
    if a.shape!=(3,512,512) or not np.isfinite(a).all() or a.min()<0 or a.max()>1:
        raise ValueError('Finite canonical RGB 512 HR in [0,1] required')
    luminance=.2126*a[0]+.7152*a[1]+.0722*a[2]
    var=float(np.var(luminance))
    smooth4=gaussian_filter(luminance,4,mode='reflect')
    chroma=np.stack((a[0]-a[1],a[2]-a[1]))
    residual=chroma-gaussian_filter(chroma,(0,1,1),mode='reflect')
    noise=float(np.sqrt(np.mean(residual**2)))
    fraction=float(np.var(smooth4)/var) if var>1e-15 else 0.
    fine=luminance-gaussian_filter(luminance,1,mode='reflect')
    dx=np.diff(luminance,axis=1);dy=np.diff(luminance,axis=0)
    cx=((chroma[:,1:-1,2:]-chroma[:,1:-1,:-2])*.5).ravel()
    cy=((chroma[:,2:,1:-1]-chroma[:,:-2,1:-1])*.5).ravel()
    tensor=np.array([[np.mean(cx*cx),np.mean(cx*cy)],[np.mean(cx*cy),np.mean(cy*cy)]])
    eigen=np.linalg.eigvalsh(tensor)
    direction=float(eigen[-1]/eigen.sum()) if eigen.sum()>1e-15 else .5
    window=np.outer(np.hanning(512),np.hanning(512))
    power=np.abs(np.fft.rfft2((luminance-luminance.mean())*window))**2
    frequencies=np.hypot(np.fft.fftfreq(512)[:,None],np.fft.rfftfreq(512)[None,:])
    spectrum=power[(frequencies>=1/32)&(frequencies<=.5)]
    flatness=float(np.exp(np.mean(np.log(spectrum+1e-20)))/np.mean(spectrum)) if np.mean(spectrum)>1e-15 else 0.
    scales={f'sigma{sigma}':float(np.var(gaussian_filter(luminance,sigma,mode='reflect'))/var) if var>1e-15 else 0. for sigma in (1,2,4,8)}
    tiles=[]
    for y in range(0,512,64):
        for x in range(0,512,64):
            tile=luminance[y:y+64,x:x+64]
            tv=float(np.var(tile));smooth=gaussian_filter(tile,4,mode='reflect')
            tiles.append(float(np.var(smooth)/tv) if tv>1e-15 else 0.)
    flags=[]
    if fraction<.15:flags.append('LOW_SPATIAL_STRUCTURE_REVIEW')
    if noise>.02:flags.append('CHROMATIC_FINE_NOISE_REVIEW')
    if fraction<.15 and noise>.02:flags.append('NOISE_DOMINATED_CONTEXT_REVIEW')
    return dict(luminance_std=float(np.sqrt(var)),
        luminance_p01_p99_span=float(np.percentile(luminance,99)-np.percentile(luminance,1)),
        structure_variance_fraction_sigma4=fraction,
        structure_variance_fractions=scales,
        chromatic_gradient_direction_fraction=direction,
        luminance_high_frequency_spectral_flatness=flatness,
        patch_low_structure_fraction=float(np.mean(np.asarray(tiles)<.15)),
        chromatic_highpass_rms=noise,luminance_highpass_rms=float(np.sqrt(np.mean(fine**2))),
        horizontal_vertical_gradient_rms_ratio=float(np.sqrt(np.mean(dx**2))/max(np.sqrt(np.mean(dy**2)),1e-12)),
        flags=flags,quality_accepted=False)


def main():
    from google.cloud import storage
    from scripts.maritime_cloud_inventory import require_cloud_vm,verify_output,digest_bytes,json_bytes,create_only,parse_gs,file_sha256
    require_cloud_vm()
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',required=True);p.add_argument('--output-prefix',required=True);p.add_argument('--scratch',required=True)
    args=p.parse_args();audit_bytes=Path(args.audit).read_bytes();audit=json.loads(audit_bytes)
    client=storage.Client();rows=[]
    for row in audit['rows']:
        candidate=json.loads(verify_output(client,row['candidate_pin']))
        payload=verify_output(client,candidate['hr'])
        with np.load(io.BytesIO(payload),allow_pickle=False) as z:hr=z['hr']
        rows.append(dict(annotation_id=row['annotation_id'],sample_id=row['sample_id'],port=row['port'],visual_status=row['status'],
            candidate_pin=row['candidate_pin'],hr_pin=candidate['hr'],hr_float32_sha256=digest_bytes(hr.tobytes()),metrics=quality_diagnostics(hr)))
    result=dict(schema='earthview-hr-quality-diagnostics-v2',audit_sha256=digest_bytes(audit_bytes),program_sha256=file_sha256(__file__),
        source_resolution_m=2.5,shape=[3,512,512],rows=rows,
        recipe=dict(gaussian_boundary='reflect',structure_sigmas=[1,2,4,8],chroma_sigma=1,patch_size=64,review_structure_threshold=.15,review_chroma_rms_threshold=.02,
            spectral_window='separable Hann',spectral_radius_cycles_per_pixel=[1/32,.5],spectral_flatness='geometric / arithmetic mean power',
            chromatic_direction='largest eigenvalue / trace of uncentered RG/BG central-gradient tensor, excluding boundary'),
        caveat='Descriptive exploratory thresholds, not a calibrated classifier. Visual audit is assistant review, not independent ground truth. Open water can have low structure. No denoising, acceptance or pixel modification.')
    out=json_bytes(result);scratch=Path(args.scratch);scratch.mkdir(parents=True,exist_ok=True);(scratch/'hr-quality-diagnostics.json').write_bytes(out)
    bucket,prefix=parse_gs(args.output_prefix.rstrip('/'));pin=dict(uri=args.output_prefix.rstrip('/')+'/hr-quality-diagnostics.json',**create_only(client.bucket(bucket).blob(prefix+'/hr-quality-diagnostics.json'),out))
    print(json.dumps(pin))


if __name__=='__main__':main()

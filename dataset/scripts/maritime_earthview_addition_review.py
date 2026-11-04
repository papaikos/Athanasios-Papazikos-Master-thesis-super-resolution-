"""Provisional EarthView addition priors and deterministic audit sampling; no qualification."""
from collections import Counter,defaultdict
import hashlib
import math

def stable(identity):
 return hashlib.sha256(('earthview-addition-audit-v1:'+identity).encode()).hexdigest()

def semantic_prior(row):
 area=row.get('marine_screening',{}).get('marine_area_m2')
 if area is None or not math.isfinite(area) or not 0<=area<=1280**2:raise ValueError('Known valid marine area prior required')
 fraction=area/(1280**2);model=row.get('model_evidence') or {}
 if model.get('predicted_medium_large'):return 'MODEL_MEDIUM_LARGE_PREDICTION'
 if model.get('predicted_harbor') and fraction>=.02:return 'MODEL_HARBOR_PREDICTION'
 if .02<=fraction<=.98:return 'MIXED_SHORE_GEOGRAPHIC_PRIOR'
 return None

def risk(row):
 m=row['metrics']
 return max(m['chromatic_highpass_rms']/.02,m['luminance_high_frequency_spectral_flatness']/.35,.15/max(m['structure_variance_fraction_sigma4'],1e-15))

def provisional_pool(rows,excluded,held):
 if len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('Duplicate candidate IDs')
 out=[]
 for row in rows:
  identity=row['sample_id']
  if identity in excluded or identity in held or row['policy']['status']!='REQUIRES_FINAL_STRATIFIED_REVIEW':continue
  if row['policy']['quality_accepted'] or row['policy']['final_eligible']:raise ValueError('Provisional input required')
  semantic=semantic_prior(row)
  if semantic is None:continue
  out.append(dict(row,semantic_prior=semantic,quality_accepted=False,final_eligible=False,requires_actual_added_selection_audit=True))
 ranked=sorted(out,key=lambda r:(risk(r),stable(r['sample_id'])))
 quartiles={r['sample_id']:min(3,i*4//len(ranked)) for i,r in enumerate(ranked)}
 for row in out:
  row['audit_stratum']='|'.join([row['semantic_prior'],'LM' if row['port']['size'] in ('L','M') else 'SV','Q'+str(quartiles[row['sample_id']])])
 return sorted(out,key=lambda r:(0 if r['port']['size'] in ('L','M') else 1,0 if r['semantic_prior']=='MODEL_MEDIUM_LARGE_PREDICTION' else 1,stable(r['sample_id'])))

def audit_rows(pool,previously_reviewed,target=300):
 count=max(target,math.ceil(.05*len(pool)));available=[r for r in pool if r['sample_id'] not in previously_reviewed]
 if len(available)<count:raise ValueError('Insufficient new audit sources; preserve prior-review exclusions')
 groups=defaultdict(list)
 for row in available:groups[row['audit_stratum']].append(row)
 allcounts=Counter(r['audit_stratum'] for r in pool)
 if set(groups)!=set(allcounts):raise ValueError('No new sources for a required audit stratum')
 countries=Counter();ports=Counter();captures=Counter();selected={}
 def choose(rows):
  row=min(rows,key=lambda r:(ports[r['port']['id']],captures[tuple(r['acquisition_identity'])],countries[r['port']['country']],stable(r['sample_id'])))
  selected[row['sample_id']]=row;ports[row['port']['id']]+=1;captures[tuple(row['acquisition_identity'])]+=1;countries[row['port']['country']]+=1
  return row
 for key in sorted(groups):
  quota=math.ceil(.05*allcounts[key])
  if len(groups[key])<quota:raise ValueError('Insufficient new sources for 5 percent audit stratum')
  for _ in range(quota):choose([r for r in groups[key] if r['sample_id'] not in selected])
 if len(selected)>count:count=len(selected)
 while len(selected)<count:choose([r for r in available if r['sample_id'] not in selected])
 return [dict(row,annotation_id='EVADD'+str(i).zfill(3)) for i,row in enumerate(sorted(selected.values(),key=lambda r:(r['audit_stratum'],stable(r['sample_id']))),1)]

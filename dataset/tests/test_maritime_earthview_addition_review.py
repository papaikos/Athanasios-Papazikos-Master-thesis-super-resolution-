import pytest
from scripts.maritime_earthview_addition_review import semantic_prior, provisional_pool, audit_rows

def row(i,frac=.5,model=None,status='REQUIRES_FINAL_STRATIFIED_REVIEW'):
 return dict(sample_id=str(i),policy=dict(status=status,quality_accepted=False,final_eligible=False),port=dict(id=str(i%12),country=str(i%4),size='L' if i%2 else 'S'),acquisition_identity=[str(i%30),'p',32631,1,'L1'],metrics=dict(chromatic_highpass_rms=.005,luminance_high_frequency_spectral_flatness=.1,structure_variance_fraction_sigma4=.5),marine_screening=dict(marine_area_m2=frac*1280**2),model_evidence=model)

def test_semantic_prior_excludes_allwater_filler_and_land_without_models():
 assert semantic_prior(row(1,1)) is None
 assert semantic_prior(row(1,0)) is None
 assert semantic_prior(row(1))=='MIXED_SHORE_GEOGRAPHIC_PRIOR'
 assert semantic_prior(row(1,1,dict(predicted_medium_large=True,predicted_harbor=False)))=='MODEL_MEDIUM_LARGE_PREDICTION'
 assert semantic_prior(row(1,0,dict(predicted_medium_large=False,predicted_harbor=True))) is None

def test_pool_preserves_locks_pending_only_and_truth_status():
 rows=[row(i) for i in range(6)]+[row(6,status='EXCLUDE_CONSERVATIVE_DIAGNOSTIC_FLAGS')]
 pool=provisional_pool(rows,{'1'},{'2'})
 assert {r['sample_id'] for r in pool}=={'0','3','4','5'}
 assert all(not r['final_eligible'] and not r['quality_accepted'] for r in pool)
 assert [r['sample_id'] for r in provisional_pool(list(reversed(rows)),{'1'},{'2'})]==[r['sample_id'] for r in pool]

def test_audit_new_deterministic_atleast5percent_and_stratum_coverage():
 pool=provisional_pool([row(i) for i in range(1000)],set(),set())
 selected=audit_rows(pool,{'0','1'},target=300)
 assert len(selected)==300 and len({r['sample_id'] for r in selected})==300
 assert not {r['sample_id'] for r in selected}&{'0','1'}
 assert selected==audit_rows(list(reversed(pool)),{'0','1'},target=300)
 assert {r['audit_stratum'] for r in selected}=={r['audit_stratum'] for r in pool}
 with pytest.raises(ValueError):audit_rows(pool,{r['sample_id'] for r in pool},target=300)


def test_addition_gce_entrypoint_imports_existing_helpers():
 import importlib
 module=importlib.import_module('reports.maritime-expansion-20261007.earthview-adapter.addition_review_on_gce')
 assert callable(module.main)

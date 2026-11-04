import pytest
from research.data_audit import audit_rows, compact_sample, membership, verify_file, file_identity
from research.experiment import KEY

# These tests pin the D2 identities; D1 and D3 runs (MARITIME_DATASET=d1|d3) skip them.
pytestmark = pytest.mark.skipif(KEY != 'd2', reason='D2-specific protocol test')


def test_d2_compaction_uses_d2_uint8_hash():
    row = compact_sample({'sample_id':'a','uint8_storage':{'d2':{'hr_sha256':'d2-hash'},'d3':{'hr_sha256':'d3-hash'}}}, 'main')
    assert row['uint8_hr_sha256'] == 'd2-hash'


@pytest.mark.parametrize('foreign', ['d1','d3'])
def test_d2_protocol_accepts_only_d2_experiment(foreign):
    from copy import deepcopy
    from research.model_factory import CONFIG, validate_config
    value=deepcopy(CONFIG); value['experiment']='maritime-d2-swinir-confirmation-100k-v1'
    assert validate_config(value) == value
    value['experiment']=f'maritime-{foreign}-swinir-confirmation-100k-v1'
    with pytest.raises(ValueError):validate_config(value)


def test_d2_metadata_rejects_wrong_recipe():
    from research.data_audit import validate_metadata
    with pytest.raises(ValueError, match='recipe'):
        validate_metadata({'dataset':'d2','recipe':'d3-variable-psf-noise-v1','release_seed':20261006}, auxiliary=False)


def test_d2_metadata_rejects_auxiliary_training():
    from research.data_audit import validate_metadata
    with pytest.raises(ValueError):
        validate_metadata({'dataset':'d2','recipe':'d2-gaussian-area-context-v1','separate_auxiliary_evaluation':True,'train_sample_count':1}, auxiliary=True)


def test_d2_shard_membership_seals_test_and_keeps_aerial_separate():
    rows=[{'sample_id':'train-id','split':'train'},{'sample_id':'test-id','split':'test'}]
    shards=[{'split':'train','object':{'sha256':'a'*64},'sample_ids':['train-id']}, {'split':'test','object':{'sha256':'b'*64},'sample_ids':['test-id']}]
    selected=membership(rows,shards,'D2')
    assert list(selected)==['train']
    assert selected['train'][0]['shard']=='D2/train/'+ 'a'*64+'.tar'
    shards[0]['sample_ids'].append('test-id')
    with pytest.raises(ValueError):membership(rows,shards,'D2')


@pytest.mark.parametrize('field', ['sample_id','site_id','geographic_group_id','acquisition_id','hr_sha256','uint8_hr_sha256'])
def test_d2_cross_split_lineage_leaks_fail(field):
    rows=[dict(sample_id='a',split='train',**{}),dict(sample_id='b',split='validation')]
    for row in rows:row[field]='shared'
    result=audit_rows(rows, {'train':1,'validation':1})
    assert result['status']=='FAIL'
    assert any(leak['field']==field for leak in result['leakage'])


def test_d2_corrupt_shard_rejected(tmp_path):
    shard=tmp_path/'opaque.tar';shard.write_bytes(b'original bytes')
    pin=file_identity(shard); shard.write_bytes(b'corrupt! bytes')
    with pytest.raises(ValueError,match='identity mismatch'):verify_file(shard,pin)


def test_d2_adapter_rejects_aerial_or_test_in_train_before_loading(tmp_path):
    from research.data_adapter import datasets_from_audit
    bad={'status':'PASS','recipe':'d2-gaussian-area-context-v1','staged_shard_integrity':{'status':'PASS'},'membership':{'train':[{'sample_id':'sealed','scope':'aerial','split':'test','shard':'D2/test/x.tar','member':'sealed.npz'}]}}
    with pytest.raises(ValueError,match='selection'):
        datasets_from_audit(tmp_path,tmp_path,bad)

@pytest.mark.parametrize("dataset,recipe", [("d1","d1-bicubic-aa-v1"),("d3","d3-variable-psf-noise-v1"),("d2","d1-bicubic-aa-v1")])
def test_d2_metadata_rejects_other_dataset_identities(dataset,recipe):
    from research.data_audit import validate_metadata
    with pytest.raises(ValueError):
        validate_metadata(dict(dataset=dataset,recipe=recipe,release_seed=20261006),auxiliary=False)


def test_d2_metadata_accepts_exact_frozen_recipe():
    from research.data_audit import validate_metadata
    validate_metadata(dict(dataset="d2",recipe="d2-gaussian-area-context-v1",release_seed=20261006),auxiliary=False)
    validate_metadata(dict(dataset="d2",recipe="d2-gaussian-area-context-v1",train_sample_count=0,separate_auxiliary_evaluation=True),auxiliary=True)


@pytest.mark.parametrize("recipe", ["d1-bicubic-aa-v1","d3-variable-psf-noise-v1"])
def test_d2_adapter_rejects_foreign_recipe_before_loading(tmp_path,recipe):
    from research.data_adapter import datasets_from_audit
    with pytest.raises(ValueError,match="recipe"):
        datasets_from_audit(tmp_path,tmp_path,dict(status="PASS",recipe=recipe,staged_shard_integrity=dict(status="PASS")))


def test_d2_yaml_matches_exact_frozen_protocol():
    from pathlib import Path
    import yaml
    from research.model_factory import CONFIG,validate_config
    config=Path(__file__).parents[1]/"research/configs/d2_swinir_100k.yaml"
    assert validate_config(yaml.safe_load(config.read_text()))==CONFIG


def test_d2_report_names_exact_training_scope(tmp_path):
    from research.reporting import scientific_report
    report=scientific_report(tmp_path,dict(outcome="TARGET_REACHED",completed_updates=100000)).read_text()
    assert "# D2 SwinIR confirmation" in report
    assert "Training scope: D2 TRAIN17952, recipe seed20261006; training seed42." in report
    assert "Tests remain sealed." in report

import json
import pytest
from research.data_audit import manifest_items, audit_rows, file_identity


def row(identity, split, **extra):
    return dict(sample_id=identity, split=split, source_kind='maxar', site_id=identity,
                geographic_group_id=identity, acquisition_id=identity, source_scene_id=identity,
                hr_sha256=identity, uint8_hr_sha256=identity, crs='EPSG:32631',
                core_bounds=[0,0,1,1], **extra)


def test_stream_arrays_with_tiny_chunks(tmp_path):
    p=tmp_path/'manifest.json'
    data={'configuration':{'text':'samples: [ and escaped \\"'},'samples':[{'sample_id':'α','nested':[1,2]},{}], 'shards':[{'split':'train'}], 'status':'READY'}
    p.write_text(json.dumps(data),encoding='utf8')
    items=list(manifest_items(p,chunk_size=9))
    assert [v for k,v in items if k=='samples']==data['samples']
    assert dict((k,v) for k,v in items if k not in ('samples','shards'))['status']=='READY'


def test_confirmed_hash_and_identity_leakage_fails():
    a=row('same','train'); b=row('same','validation')
    result=audit_rows([a,b], expected_counts={'train':1,'validation':1})
    assert result['status']=='FAIL'
    assert any(x['field']=='sample_id' for x in result['leakage'])
    assert any(x['field']=='uint8_hr_sha256' for x in result['leakage'])


def test_different_crs_never_treated_as_overlap():
    a=row('a','train');b=row('b','validation'); b['crs']='EPSG:32632'
    result=audit_rows([a,b],expected_counts={'train':1,'validation':1})
    assert result['status']=='PASS'
    assert result['footprints']['cross_crs_pairs_not_compared']==1


def test_same_crs_overlap_is_confirmed_and_touching_is_not():
    a=row('a','train'); b=row('b','validation')
    assert audit_rows([a,b],expected_counts={'train':1,'validation':1})['status']=='FAIL'
    b['core_bounds']=[1,0,2,1]
    assert audit_rows([a,b],expected_counts={'train':1,'validation':1})['status']=='PASS'


def test_missing_metadata_recorded_without_inventing_values():
    r={'sample_id':'a','split':'train'}
    result=audit_rows([r],expected_counts={'train':1})
    assert result['unknown_counts']['acquisition_id']==1
    assert result['status']=='PASS'


def test_count_mismatch_fails_closed():
    assert audit_rows([],expected_counts={'train':1})['status']=='FAIL'


def test_identity_checks_detect_modified_bytes(tmp_path):
    p=tmp_path/'x';p.write_bytes(b'hello')
    pin=file_identity(p)
    p.write_bytes(b'HELLO')
    assert file_identity(p)['sha256']!=pin['sha256']


def test_constituent_scene_overlap_is_not_hidden_by_mosaic_identity():
    a=row('a','train');b=row('b','validation');b['core_bounds']=[1,0,2,1]
    a['source_scene_id']=['tile-1','tile-2'];b['source_scene_id']=['tile-2','tile-3']
    result=audit_rows([a,b],expected_counts={'train':1,'validation':1})
    assert result['status']=='FAIL'
    assert any(x['field']=='source_scene_id' and x['value']=='tile-2' for x in result['leakage'])


def test_truncated_stream_fails(tmp_path):
    p=tmp_path/'bad';p.write_text('{"samples":[{"sample_id":"a"}')
    with pytest.raises(ValueError):list(manifest_items(p,chunk_size=5))


def test_membership_uses_exact_split_hash_path_and_seals_tests():
    from research.data_audit import membership
    rows=[{'sample_id':'a','split':'train'},{'sample_id':'b','split':'test'}]
    shards=[{'split':'train','object':{'sha256':'hash-a'},'sample_ids':['a']},
            {'split':'test','object':{'sha256':'hash-b'},'sample_ids':['b']}]
    selected=membership(rows,shards,'D3')
    assert set(selected)=={'train'}
    assert selected['train'][0]['shard']=='D3/train/hash-a.tar'
    assert selected['train'][0]['member']=='a.npz'
    shards[0]['sample_ids'].append('b')
    with pytest.raises(ValueError):membership(rows,shards,'D3')


def test_pinned_file_verification_rejects_crc_or_sha_mismatch(tmp_path):
    from research.data_audit import verify_file
    p=tmp_path/'opaque';p.write_bytes(b'opaque archive fixture')
    pin=file_identity(p)
    assert verify_file(p,pin)==pin
    with pytest.raises(ValueError):verify_file(p,dict(pin,crc32c='wrong'))
    with pytest.raises(ValueError):verify_file(p,dict(pin,sha256='wrong'))


def test_compact_lineage_reads_recorded_constituent_ids():
    from research.data_audit import compact_sample
    sample={'sample_id':'mosaic','source_provenance':{'source_items':[{'item':{'id':'s1'}},{'item':{'id':'s2'}}]}}
    assert compact_sample(sample,'main')['source_scene_id']==['s1','s2']
    sample['source_provenance']={'asset_id':'original/recorded/skysat'}
    assert compact_sample(sample,'main')['source_scene_id']=='original/recorded/skysat'


def test_cli_refuses_to_write_outputs_inside_input_root(tmp_path):
    from research.data_audit import main
    frozen=tmp_path/'frozen';frozen.mkdir();manifest=frozen/'manifest.json';manifest.write_text('frozen bytes')
    with pytest.raises(SystemExit) as outcome:
        main(['--data-root',str(frozen),'--output',str(manifest)])
    assert outcome.value.code==2 and manifest.read_text()=='frozen bytes'
    outside=tmp_path/'report.json'
    with pytest.raises(SystemExit) as outcome:
        main(['--data-root',str(frozen),'--output',str(outside),'--source-map',str(frozen/'groups.csv')])
    assert outcome.value.code==2 and not outside.exists() and not (frozen/'groups.csv').exists()


def test_cli_allows_separate_failure_evidence_without_touching_inputs(tmp_path):
    from research.data_audit import main
    frozen=tmp_path/'frozen';frozen.mkdir();out=tmp_path/'run'/'audit.json'
    assert main(['--data-root',str(frozen),'--output',str(out)])==1
    assert json.loads(out.read_text())['status']=='FAIL' and list(frozen.iterdir())==[]


def test_cli_rejects_json_source_map_destination_collision(tmp_path):
    from research.data_audit import main
    target=tmp_path/'same.json'
    with pytest.raises(SystemExit) as outcome:
        main(['--data-root',str(tmp_path/'frozen'),'--output',str(target),'--source-map',str(target)])
    assert outcome.value.code==2 and not target.exists()

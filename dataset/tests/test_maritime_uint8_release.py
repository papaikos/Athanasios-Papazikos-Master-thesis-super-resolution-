import copy
import io
import json
import tarfile

import numpy as np
import pytest


def test_added_batches_group_aerial_parent_replay_and_preserve_every_sample():
    from scripts.maritime_uint8_release import added_batches
    def row(identity, parent):
        return dict(sample_id=identity, split='train', source_kind='aerial_rgb',
                    source=dict(source_original_object={'uri':parent,'generation':'1'},
                                native_context={'uri':identity,'generation':'1'}))
    rows = [row('a','parent2'), row('b','parent1'), row('c','parent2'), row('d','parent1')]
    batches = added_batches(rows, 3)
    assert [[r['sample_id'] for r in b] for b in batches] == [['b','d','a'],['c']]
    assert added_batches(list(reversed(rows)),3) == batches
    assert sorted(r['sample_id'] for b in batches for r in b) == ['a','b','c','d']
    assert all(len(b) <= 3 for b in batches)


def test_added_batches_never_mix_sky_original_ownership_or_splits():
    from scripts.maritime_uint8_release import added_batches
    rows = [dict(sample_id=str(i), source_kind='skysat_raw_window',split='train',
                 source={'source_object':{'uri':'parent'+str(i//2),'generation':'1'}}) for i in range(4)]
    assert [len(b) for b in added_batches(rows,32)] == [2,2]
    rows.append(dict(sample_id='eval', source_kind='skysat_raw_window',split='test',
                     source={'source_object':{'uri':'parent0','generation':'1'}}))
    assert all(len({r['split'] for r in b}) == 1 for b in added_batches(rows,32))


def test_added_batches_fail_on_aerial_missing_original_parent():
    from scripts.maritime_uint8_release import added_batches
    with pytest.raises(KeyError):
        added_batches([dict(sample_id='x',split='train',source_kind='aerial_rgb',
                            source={'native_context':{'uri':'fake','generation':'1'}})],32)


def test_new_uint8_hr_duplicate_never_counts_as_distinct_imagery_but_baseline_is_preserved():
    from scripts.maritime_uint8_release import register_hr_hash
    def sample(identity, digest):
        return dict(sample_id=identity,uint8_storage={k:{'hr_sha256':digest} for k in RECIPE_VERSIONS})
    seen = {}
    assert register_hr_hash(seen,sample('old','a'*64),baseline=True) is None
    assert register_hr_hash(seen,sample('preserved','a'*64),baseline=True) == 'old'
    with pytest.raises(ValueError,match='Duplicate added uint8 HR'):
        register_hr_hash(seen,sample('new','a'*64),baseline=False)
    assert register_hr_hash(seen,sample('distinct','b'*64),baseline=False) is None
    assert seen == {'a'*64:'old','b'*64:'distinct'}

from src.maritime_recipes import RECIPE_VERSIONS, array_hash


def baseline():
    row = dict(sample_id='old', split='train', source_kind='maxar', site_id='port',
        geographic_group_id='g', acquisition_id='capture', source=dict(native_hash='unchanged'))
    common = {key: value for key, value in row.items() if key != 'source'}
    common.update(source_provenance=row['source'], decoded_native_core_sha256='a' * 64)
    return dict(selection=dict(selection=[row]), samples=[common])


def test_baseline_preservation_allows_distinct_training_additions():
    from scripts.maritime_uint8_release import preserve_baseline
    old = baseline()
    selection = copy.deepcopy(old['selection'])
    selection['selection'].append(dict(sample_id='new', split='train', source_kind='maxar'))
    assert [row['sample_id'] for row in preserve_baseline(old, selection)] == ['new']


@pytest.mark.parametrize('mutation', ['missing', 'split', 'acquisition', 'native_source', 'new_holdout'])
def test_baseline_mutations_and_holdout_expansion_rejected(mutation):
    from scripts.maritime_uint8_release import preserve_baseline
    old = baseline()
    selection = copy.deepcopy(old['selection'])
    if mutation == 'missing': selection['selection'] = []
    elif mutation == 'split': selection['selection'][0]['split'] = 'test'
    elif mutation == 'acquisition': selection['selection'][0]['acquisition_id'] = 'replacement'
    elif mutation == 'native_source': selection['selection'][0]['source']['native_hash'] = 'replacement'
    elif mutation == 'new_holdout': selection['selection'].append(dict(sample_id='new', split='test'))
    with pytest.raises(ValueError): preserve_baseline(old, selection)


def float_pair(recipe='d2'):
    hr, lr = np.full((3, 512, 512), 0.5, np.float32), np.full((3, 128, 128), 0.25, np.float32)
    metadata = dict(sample_id='old', split='train', site_id='port', geographic_group_id='g',
        lr_origin='synthetic_hr_only', hr_sha256=array_hash(hr),
        hr_transform=[2.5, 0, 0, 0, -2.5, 1280], lr_transform=[10, 0, 0, 0, -10, 1280],
        recipe=dict(recipe=RECIPE_VERSIONS[recipe], sample_id='old', release_seed=20261006,
            variant_id='0', hr_sha256=array_hash(hr), lr_sha256=array_hash(lr)))
    return dict(hr=hr, lr=lr, hr_mask=np.ones((512, 512), bool), lr_mask=np.ones((128, 128), bool)), metadata


def test_baseline_float_conversion_preserves_recipe_and_writes_real_uint8(tmp_path):
    from scripts.maritime_uint8_release import convert_baseline_shard
    from src.maritime_uint8_dataset import MaritimeUint8Dataset
    pair, metadata = float_pair()
    stream = io.BytesIO()
    np.savez_compressed(stream, **pair, provenance=np.asarray(json.dumps(metadata)))
    old_path, new_path = tmp_path/'old.tar', tmp_path/'new.tar'
    with tarfile.open(old_path, 'w') as archive:
        info = tarfile.TarInfo('old.npz'); info.size = len(stream.getvalue())
        archive.addfile(info, io.BytesIO(stream.getvalue()))
    common = dict(metadata, acquisition_id='capture', source_provenance=dict(license='research'))
    stored = convert_baseline_shard(old_path, new_path, 'd2', ['old'], {'old': common}, 20261006)
    dataset = MaritimeUint8Dataset([new_path], 'd2')
    arrays, encoded = dataset.read_arrays(0)
    assert encoded['recipe'] == metadata['recipe']
    assert encoded['acquisition_id'] == 'capture'
    assert encoded['source_provenance']['license'] == 'research'
    assert arrays['hr'].dtype == arrays['lr'].dtype == np.uint8
    assert stored['old']['rgb_payload_bytes'] == 835584


def test_release_readiness_counts_only_training_rgb_and_separates_target_shortfall():
    from scripts.maritime_uint8_release import release_status, EXPANSION_GATE_NAMES
    from src.maritime_uint8 import STORAGE_VERSION
    storage = dict(version=STORAGE_VERSION, rgb_payload_bytes=835584, hr_sha256='a' * 64)
    samples = [dict(sample_id='a', split='train', uint8_storage={name:storage for name in RECIPE_VERSIONS}),
               dict(sample_id='b', split='test', uint8_storage={name:storage for name in RECIPE_VERSIONS})]
    gates = {name:True for name in EXPANSION_GATE_NAMES}
    status = release_status(samples, gates, set(gates), True, minimum=835584, target=1671168)
    assert status['status'] == 'READY'
    assert status['train_rgb_bytes_per_child'] == 835584
    assert status['target_shortfall_rgb_bytes_per_child'] == 835584
    status = release_status(samples, {'quality':False}, set(), True, minimum=835584, target=1671168)
    assert status['status'] == 'REVIEW_PENDING'
    assert 'expansion_gate:quality' in status['readiness_gaps']
    assert release_status(samples, {}, set(), True, minimum=1671168, target=1671168)['status'] != 'READY'


def test_arbitrary_gate_name_cannot_replace_the_four_research_evidence_categories():
    from scripts.maritime_uint8_release import release_status, EXPANSION_GATE_NAMES
    from src.maritime_uint8 import STORAGE_VERSION
    storage=dict(version=STORAGE_VERSION,rgb_payload_bytes=835584,hr_sha256='a'*64)
    sample=dict(sample_id='a',split='train',uint8_storage={r:storage for r in RECIPE_VERSIONS})
    status=release_status([sample],{'quality':True},{'quality'},True,minimum=835584,target=835584)
    assert status['status']=='REVIEW_PENDING'
    assert all('expansion_gate:'+name in status['readiness_gaps'] for name in EXPANSION_GATE_NAMES)


def test_gate_evidence_requires_truthful_pass_and_exact_release_population_code_baseline():
    from scripts.maritime_uint8_release import validate_expansion_gate_evidence, EXPANSION_GATE_NAMES
    gate=EXPANSION_GATE_NAMES[0]
    document=dict(schema='maritime-expansion-gate-evidence-v1',gate=gate,passed=True,
        selected_population_sha256='a'*64,baseline_index_sha256='b'*64,code_sha256='c'*64,
        scope='Exact selected source rights and delivered grid contracts.',
        checks={'all_selected_rights_and_resolution_contracts_valid':True})
    validate_expansion_gate_evidence(document,gate,population_sha256='a'*64,
        baseline_index_sha256='b'*64,code_sha256='c'*64)
    for key,value in [('passed',False),('schema','unrelated'),('gate',EXPANSION_GATE_NAMES[1]),
        ('selected_population_sha256','d'*64),('baseline_index_sha256','d'*64),
        ('code_sha256','d'*64),('checks',{}),('checks',{'claimed_check':False}),('scope','')]:
        changed=dict(document,**{key:value})
        with pytest.raises(ValueError,match='evidence'):
            validate_expansion_gate_evidence(changed,gate,population_sha256='a'*64,
                baseline_index_sha256='b'*64,code_sha256='c'*64)


def test_shared_hr_byte_mismatch_rejected_and_stratified_recompute_is_stable():
    from scripts.maritime_uint8_release import release_status, stratified_recompute
    rows = [dict(sample_id=str(i), split='train', source_kind='maxar', primary_category='shoreline') for i in range(21)]
    assert len(stratified_recompute(rows)) == 2
    assert stratified_recompute(rows) == stratified_recompute(list(reversed(rows)))
    sample = dict(sample_id='s', split='train', uint8_storage={name:dict(rgb_payload_bytes=835584,
        hr_sha256=name) for name in RECIPE_VERSIONS})
    with pytest.raises(ValueError, match='HR'): release_status([sample], {}, set(), True)


def test_cloud_guard_precedes_baseline_access(monkeypatch):
    from scripts.maritime_uint8_release import run
    from scripts import maritime_cloud_inventory
    def deny(): raise RuntimeError('GCE required')
    monkeypatch.setattr(maritime_cloud_inventory, 'require_cloud_vm', deny)
    with pytest.raises(RuntimeError, match='GCE required'): run(object())


def test_readiness_rejects_storage_that_is_not_versioned_uint8():
    from scripts.maritime_uint8_release import release_status
    sample = dict(sample_id='s', split='train', uint8_storage={name:dict(version='float32',
        rgb_payload_bytes=835584, hr_sha256='a'*64) for name in RECIPE_VERSIONS})
    with pytest.raises(ValueError, match='storage'): release_status([sample], {}, set(), True)


def test_baseline_seed_cannot_be_changed_during_conversion(tmp_path):
    from scripts.maritime_uint8_release import convert_baseline_shard
    pair, metadata = float_pair()
    stream=io.BytesIO()
    np.savez_compressed(stream, **pair, provenance=np.asarray(json.dumps(metadata)))
    old_path=tmp_path/'old.tar'
    with tarfile.open(old_path,'w') as archive:
        info=tarfile.TarInfo('old.npz');info.size=len(stream.getvalue())
        archive.addfile(info,io.BytesIO(stream.getvalue()))
    with pytest.raises(ValueError, match='seed'):
        convert_baseline_shard(old_path,tmp_path/'new.tar','d2',['old'],{'old':metadata},999)


def test_pinned_baseline_loader_cross_checks_manifest_sources_and_layout():
    from scripts.maritime_uint8_release import load_baseline
    from scripts.maritime_cloud_inventory import digest_bytes, json_bytes
    records={}
    def pin(identity, data):
        payload=json_bytes(data)
        ref=dict(uri='gs://bucket/'+identity,generation='1',bytes=len(payload),
            crc32c='AAAAAA==',sha256=digest_bytes(payload))
        records[ref['uri']]=payload
        return ref
    old=baseline()
    selection_pin=pin('selection',old['selection'])
    configuration=dict(release_seed=20261006,selection_sha256=selection_pin['sha256'])
    shards=[dict(split='train',sample_ids=['old'],object=selection_pin)]
    manifests={}
    for name in ['mother',*RECIPE_VERSIONS]:
        manifest=dict(dataset=name,samples=old['samples'],sample_count=1,
            training_format_verified=True,configuration=configuration,
            shards=shards,selection=selection_pin,recipe=RECIPE_VERSIONS.get(name))
        manifests[name]=pin(name,manifest)
    receipt_pin=pin('receipt',dict(state='COMPLETE',training_format_verified=True,manifests=manifests))
    index_pin=pin('index',dict(datasets=manifests,release_receipt=receipt_pin,sample_count=1))
    class Store:
        def get(self,ref): return records[ref['uri']]
    result=load_baseline(Store(),index_pin)
    assert result['samples']==old['samples']
    corrupted=json.loads(records[manifests['d2']['uri']])
    corrupted['samples'][0]['source_provenance']['native_hash']='replacement'
    records[manifests['d2']['uri']]=json_bytes(corrupted)
    with pytest.raises(ValueError,match='structure/provenance'):
        load_baseline(Store(),index_pin)

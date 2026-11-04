from scripts.maritime_ultimate_select import byte_balanced, count_bytes, physical_key, disjoint_native_rows, assign_mixed_splits, quality_rejected_keys


def row(identity, site, size):
    return dict(sample_id=identity,site_id=site,source_kind='maxar',source={'useful_native_rgb_bytes':size})


def test_native_target_uses_actual_source_bytes_not_sample_count():
    values=[row('a','one',52_770_000),row('b','two',7_680_000)]
    assert count_bytes(values)==60_450_000


def test_byte_limit_never_padding_or_duplicate():
    values=[row('a','one',7),row('a','one',7),row('b','two',8),row('c','three',10)]
    chosen=byte_balanced(values,20,25,1)
    assert len({r['sample_id'] for r in chosen})==len(chosen)
    assert 20<=count_bytes(chosen)<=25


def test_site_fraction_is_computed_against_native_bytes():
    chosen=byte_balanced([row('a','one',60),row('b','one',60),row('c','two',50)],100,110,.6)
    assert count_bytes(chosen)==110


def test_shortfall_keeps_best_available_and_reports_actual_size():
    values=[row('a','one',10),row('b','two',10)]
    assert count_bytes(byte_balanced(values,100,110,1))==20


def test_shifted_world_cells_share_rejection_identity():
    a=dict(source_kind='skysat_raw_window',source={},acquisition_id='ee_catalog:c',crs='EPSG:32631',core_bounds=[0,0,1280,1280])
    b=dict(a,acquisition_id='c',core_bounds=[.4,.4,1280.4,1280.4])
    assert physical_key(a)==physical_key(b)


def test_adjacent_cells_cannot_mix_overlapping_native_grid_phases():
    a=dict(sample_id='a',source_kind='skysat_raw_window',source={},acquisition_id='c',crs='EPSG:32631',core_bounds=[.4,0,1280.4,1280])
    b=dict(a,sample_id='b',core_bounds=[1280,0,2560,1280])
    assert disjoint_native_rows([a,b])==[a]


def test_mixed_split_ratios_use_native_bytes_and_preserve_training_history():
    groups=[dict(geographic_group_id='old',site_ids=['old'],prior_training=True,development=False)]
    values=[dict(row(str(i),'old',1),source_kind='skysat_raw_window',geographic_group_id='old') for i in range(100)]
    for i in range(30):
        gid=f'new{i:02d}'
        groups.append(dict(geographic_group_id=gid,site_ids=[gid],prior_training=False,development=False))
        values.append(dict(row(gid,gid,52_770_000),geographic_group_id=gid))
    result=assign_mixed_splits(groups,values)
    assert result['assignment']['old']=='train'
    assert sum(g['split']=='test' for g in groups)==10
    assert sum(g['split']=='validation' for g in groups)==5
    assert sum(g['split']=='train' for g in groups)==16
    for group in groups:
        assert group['selected_core_bytes']==count_bytes([r for r in values if r['geographic_group_id']==group['geographic_group_id']])


def test_mixed_split_preserves_frozen_site_lock_after_group_identity_changes():
    groups=[dict(geographic_group_id='merged',site_ids=['locked','other'],prior_training=False,development=False)]
    frozen=dict(assignment={'previous':'validation'},groups=[dict(geographic_group_id='previous',site_ids=['locked'],split='validation')])
    result=assign_mixed_splits(groups,[dict(row('x','locked',100),geographic_group_id='merged')],frozen)
    assert result['assignment']['merged']=='validation'


def test_quality_rejection_excludes_physical_alias_without_promoting_unreviewed_rows():
    a=dict(sample_id='a',acquisition_id='c',crs='EPSG:32631',core_bounds=[0,0,1280,1280])
    b=dict(a,sample_id='alias',core_bounds=[.4,.4,1280.4,1280.4])
    ledger=dict(reviews=[dict(sample_id='a',decision='reject',reason='inland pond')])
    keys=quality_rejected_keys([a,b],[ledger])
    assert physical_key(a) in keys and physical_key(b) in keys

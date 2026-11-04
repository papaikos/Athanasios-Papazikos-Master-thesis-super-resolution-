import pytest
from scripts.maritime_candidate_manifest import normalize_candidate, consolidate, read_json_pinned


def pin(name):
    return dict(uri='gs://bucket/'+name,generation='1',bytes=10,crc32c='crc',sha256='a'*64)


def source():
    return dict(asset_id='asset',catalog_id='capture',site_id='site',uri='gs://bucket/raw.tif',
        generation='5',bytes=100,crc32c='rawcrc',export_recipe={'unmask':0,'noData':0})


def reuse(sid='reuse'):
    return dict(sample_id=sid,source_object=source(),source_sha256='b'*64,
        crs='EPSG:32632',core_bounds=[0,0,1280,1280],core_transform=[.8,0,0,0,-.8,1280],
        mask_provenance='legacy_nodata_zero_conservative',
        artifacts=dict(rgb=pin('rgb.tif'),mask=pin('mask.tif'),native_preview=pin('native.png'),overview=pin('overview.png')))


def test_reuse_and_pilot_geometry_and_mask_schemas():
    a=normalize_candidate(reuse(),{},pin('report.json'))
    p=dict(sample_id='pilot',source=source(),source_sha256='b'*64,site_id='site',crs='EPSG:32632',
        mother_core_transform=[.8,0,0,0,-.8,1280],mother=pin('mother.tif'),mother_mask=pin('mask.tif'),
        native_review_preview=pin('native.png'),source_support='documented_unmask_zero_nodata_zero_conservative')
    b=normalize_candidate(p,{},pin('report2.json'))
    assert a['canonical_core_id']==b['canonical_core_id']
    assert a['semantic_status']=='pending' and b['conservative_mask']
    assert a['country'] is None and a['country_status']=='unresolved'
    assert a['context_bounds']==[-40,-40,1320,1320]


def test_planet_and_disaster_original_pin_distinctions():
    planet_source=dict(asset_id='planet',site_id='san_francisco',country='USA',license='CC-BY-4.0',
        native_gsd_m=.7,uri='gs://open-cogs/orig.tif',generation='7',bytes=100,crc32c='crc',
        stac={'properties':{'pl:strip_id':'strip'}})
    c=dict(sample_id='p',core_bounds=[0,0,1280,1280],mother=pin('p.tif'),preview=pin('p.png'),
        original_mask_provenance='original_embedded_alpha_mask',near_stereo_review_pending=True)
    p=normalize_candidate(c,dict(source=planet_source,source_sha256='c'*64,crs='EPSG:32632'),pin('p-report.json'))
    assert p['native_gsd_m']==.7 and p['mask']['kind']=='embedded_original_support'
    assert 'near_stereo_review_pending' in p['flags']
    disaster=dict(id='d',site_id='coast',site_country='Philippines',license='CC-BY-NC-4.0',native_gsd_m=.71,strip_id='dstrip')
    d=normalize_candidate(dict(c, sample_id='d',mask_provenance='original_udm_all_unusable_bits_zero'),
        dict(source=disaster,original_http={'sha256':'d'*64},original_mirror=pin('d-original.tif'),crs='EPSG:32632'),pin('d-report.json'))
    assert d['license']=='CC-BY-NC-4.0' and d['source_objects'][0]['uri'].endswith('d-original.tif')


def test_missing_provenance_rejected():
    value=reuse()
    del value['source_object']['generation']
    with pytest.raises(ValueError,match='generation'): normalize_candidate(value,{},pin('r.json'))
    value=reuse()
    del value['mask_provenance']
    with pytest.raises(ValueError,match='mask'): normalize_candidate(value,{},pin('r.json'))
    value=reuse()
    value['mask_provenance']='unresolved_all_valid'
    with pytest.raises(ValueError,match='mask'): normalize_candidate(value,{},pin('r.json'))


def test_full_ee_asset_and_short_catalog_share_acquisition_key():
    a=reuse('short')
    b=reuse('full')
    b['source_object']['catalog_id']='SKYSAT/GEN-A/PUBLIC/ORTHO/RGB/capture'
    aa=normalize_candidate(a,{},pin('a.json'))
    bb=normalize_candidate(b,{},pin('b.json'))
    assert aa['canonical_core_id']==bb['canonical_core_id']


def test_dedup_exact_and_same_acquisition_overlap_capacity():
    a=normalize_candidate(reuse('a'),{},pin('a-report.json'))
    b=normalize_candidate(reuse('b'),{},pin('b-report.json'))
    shifted=reuse('c')
    shifted['core_bounds']=[640,0,1920,1280]
    shifted['core_transform']=[.8,0,640,0,-.8,1280]
    c=normalize_candidate(shifted,{},pin('c-report.json'))
    manifest=consolidate([a,b,c])
    assert manifest['summary']['distinct_candidate_count']==2
    assert manifest['summary']['disjoint_candidate_core_bytes_upper_bound']==7680000
    assert manifest['summary']['semantic_accepted_core_bytes']==0
    assert manifest['candidates'][0]['alternate_candidate_ids']==['b']
    assert 'same_acquisition_overlap' in manifest['candidates'][1]['flags']


def test_metadata_get_generation_locked_and_hash_checked():
    import hashlib
    class Blob:
        name='report.json';generation='12';size=2;crc32c='crc'
        def download_as_bytes(self,**kwargs):
            assert kwargs==dict(if_generation_match=12,checksum='crc32c')
            return b'{}'
    value,ref=read_json_pinned(Blob(),'bucket')
    assert value=={} and ref['sha256']==hashlib.sha256(b'{}').hexdigest()


def test_country_lists_normalize_without_inventing_combined_country():
    row=reuse()
    row['source_object']['country']=['Greece']
    record=normalize_candidate(row,{},pin('r.json'),{'site':{'country':'Greece'}})
    assert record['country']=='Greece'
    assert consolidate([record])['summary']['countries']==['Greece']
    row['source_object']['country']=['Greece','Turkey']
    record=normalize_candidate(row,{},pin('r.json'),{'site':{'country':'Greece'}})
    assert record['country'] is None and record['country_status']=='unresolved'
    assert record['country_candidates']==['Greece','Turkey']
    assert consolidate([record])['summary']['unresolved_country_count']==1


def test_reuse_producer_schema_preserves_valid_black_pixels_and_mask_provenance():
    import numpy as np
    from rasterio.io import MemoryFile
    from rasterio.enums import ColorInterp
    from rasterio.transform import from_origin
    from scripts.maritime_reuse_candidates import supported_windows
    src=dict(source(),mask_provenance='producer_original_min_rgb_mask',
        producer_mask=pin('producer-mask.tif'),country=['Greece'])
    transform=from_origin(-80.8,1360.8,.8,.8)
    with MemoryFile() as rgb_mem, MemoryFile() as mask_mem:
        with rgb_mem.open(driver='GTiff',width=1800,height=1800,count=3,dtype='uint8',
                crs='EPSG:32632',transform=transform,nodata=0) as rgb:
            rgb.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue)
            rgb.write(np.zeros((3,1800,1800),np.uint8))
            with mask_mem.open(driver='GTiff',width=1800,height=1800,count=1,dtype='uint8',
                    crs=rgb.crs,transform=transform) as mask:
                mask.write(np.full((1,1800,1800),255,np.uint8))
                extracted=list(supported_windows(rgb,src,mask))
                assert len(extracted)==1
                assert not rgb.read_masks().any()
                candidate=dict(extracted[0],sample_id='producer',source_object=src,
                    source_sha256='b'*64,producer_mask_sha256='c'*64,
                    artifacts=reuse()['artifacts'])
                record=normalize_candidate(candidate,{},pin('producer-report.json'))
                assert record['mask']['provenance']=='producer_original_min_rgb_mask'
                assert not record['conservative_mask']
                assert record['original_mask_object']['uri']==src['producer_mask']['uri']
                assert record['original_mask_object']['sha256']=='c'*64
                assert consolidate([record])['summary']['countries']==['Greece']


def test_reuse_and_pilot_raw_windows_grid_catalog_time_and_angles_survive():
    row=reuse()
    row.update(core_window=[101,202,1600,1600],context_window=[51,152,1700,1700])
    row['source_object'].update(grid={'affineTransform':{'scaleX':.8,'scaleY':-.8}},
        properties={'collectionStartTime':'2016-04-11T08:15:55Z','satelliteAzimuthAngleMean':94.3},
        acquisition_utc='2016-04-11T08:16:00Z')
    out=normalize_candidate(row,{},pin('reuse-report.json'))
    assert out['core_window']==row['core_window']
    assert out['context_window']==row['context_window']
    assert out['source_grid']==row['source_object']['grid']
    assert out['catalog_id']=='capture'
    assert out['acquisition_utc']=='2016-04-11T08:15:55Z'
    assert out['source_view_angles']['satelliteAzimuthAngleMean']==94.3
    pilot=dict(sample_id='pilot',source=row['source_object'],source_sha256='b'*64,
        site_id='site',crs='EPSG:32632',core_window=[101,202,1600,1600],
        mother_core_transform=[.8,0,0,0,-.8,1280],mother=pin('p.tif'),mother_mask=pin('p-mask.tif'),
        native_review_preview=pin('p.png'),source_support='documented_unmask_zero_nodata_zero_conservative')
    out=normalize_candidate(pilot,{},pin('pilot-report.json'))
    assert out['core_window']==pilot['core_window'] and out['context_window'] is None
    assert out['source_window_provenance_status']=='core_recorded_context_unresolved'


def test_vendor_integer_read_window_and_original_point_five_grid_survive_without_invention():
    for disaster in (False,True):
        src=dict(asset_id='vendor',site_id='coast',country='USA',license='CC-BY-4.0',native_gsd_m=.7,
            uri='gs://open-cogs/orig.tif',generation='7',bytes=100,crc32c='crc',
            stac={'properties':{'pl:strip_id':'strip','datetime':'2020-01-01T00:00:00Z','view:off_nadir':6.4}})
        report=dict(source=src,source_sha256='c'*64,crs='EPSG:32632',
            original_transform=[.5,0,100,0,-.5,200],resampling='exact_area_0.5_to_0.8')
        if disaster:
            src.update(id='vendor',strip_id='strip',license='CC-BY-NC-4.0',site_country='USA')
            report.update(original_http={'sha256':'c'*64},original_mirror=pin('mirror.tif'))
        candidate=dict(sample_id='v',core_bounds=[0,0,1280,1280],mother=pin('v.tif'),preview=pin('v.png'),
            source_pixel_window=[0,1,2720,2721],original_mask_provenance='original_embedded_alpha_mask')
        out=normalize_candidate(candidate,report,pin('vendor-report.json'))
        assert out['source_pixel_window']==candidate['source_pixel_window']
        assert out['source_raster_transform']==report['original_transform']
        assert out['source_grid']['original_transform'][0]==.5
        assert out['core_window'] is None
        assert out['source_fractional_window'] is None
        assert out['source_window_provenance_status']=='integer_read_window_recorded_fractional_support_unresolved'
        candidate['source_fractional_window']=[.3,1.6,2720,2720]
        assert normalize_candidate(candidate,report,pin('vendor-report.json'))['source_fractional_window']==candidate['source_fractional_window']


def test_explicit_catalog_join_restores_producer_properties_without_guessing_time():
    row=reuse()
    catalog={'asset':dict(id='asset',properties={'catalogID':'capture','collectionStartTime':'2014-07-03T19:02:40Z',
        'satelliteElevationAngleMean':70.5},bands=[{'grid':{'crsWkt':'known','dimensions':{'width':10,'height':20}}}])}
    out=normalize_candidate(row,{},pin('r.json'),catalog_metadata=catalog)
    assert out['acquisition_utc']=='2014-07-03T19:02:40Z'
    assert out['source_view_angles']['satelliteElevationAngleMean']==70.5
    assert out['source_grid']==catalog['asset']['bands'][0]['grid']
    assert out['catalog_metadata_asset_id']=='asset'


def test_observed_tile_header_preserved_separately_from_native_export_plan():
    row=reuse()
    row['source_object']['native_plan']=dict(crs='EPSG:32632',crs_transform=[.8,0,100,0,-.8,9000])
    header=dict(crs='EPSG:32632',affine=[.8,0,6653.6,0,-.8,2446.4],width=8192,height=8192)
    record=normalize_candidate(row,dict(native_header=header),pin('partial-report.json'))
    assert record['source_native_header']==header
    assert record['source_raster_transform']==header['affine']
    assert record['source_grid']==row['source_object']['native_plan']
    assert record['source_grid_scope']=='native_export_plan'
    record=normalize_candidate(row,{},pin('aggregate.json'))
    assert record['source_native_header'] is None
    assert record['source_raster_transform'] is None


@pytest.mark.parametrize('aggregate',[False,True])
def test_partial_sources_and_reuse_source_reports_discovery_pinned_no_duplicates(tmp_path,monkeypatch,aggregate):
    import json
    from types import SimpleNamespace
    from scripts import maritime_candidate_manifest as manifest
    from scripts import maritime_build_control as control
    downloads=[]
    class Blob:
        def __init__(self,name,document):
            self.name=name; self.generation=12; self.crc32c='crc'
            self.payload=json.dumps(document).encode(); self.size=len(self.payload)
        def download_as_bytes(self,**kwargs):
            assert kwargs==dict(if_generation_match=12,checksum='crc32c')
            downloads.append(self.name)
            return self.payload
    report=dict(candidates=[reuse('reuse-partial')],native_header=dict(
        crs='EPSG:32632',affine=[.8,0,100,0,-.8,9000],width=8192,height=8192))
    vendor=reuse('vendor-partial')
    vendor['source_object']['catalog_id']='vendor-other'
    blobs=[Blob('partial/source-reports/source-hash.json',report),
        Blob('partial/sources/vendor.json',dict(candidates=[vendor])),
        Blob('partial/control/checkpoint.json',dict(candidates=[reuse('ignore-control')]))]
    if aggregate:
        blobs.append(Blob('partial/candidate-manifest.json',dict(candidates=[reuse('aggregate-authoritative')])))
    client=SimpleNamespace(list_blobs=lambda bucket,prefix:[blob for blob in blobs if blob.name.startswith(prefix)])
    monkeypatch.setattr(control,'client_from_gcloud',lambda:client)
    args=SimpleNamespace(prefix=['gs://bucket/partial'],publish=None,project='project',
        output=str(tmp_path/'inventory.json'),site_metadata=None,catalog_metadata=None)
    result=manifest.run(args)
    assert result['summary']['semantic_accepted_count']==0
    assert result['summary']['semantic_accepted_core_bytes']==0
    assert all(row['semantic_status']=='pending' and not row['final_eligible'] for row in result['candidates'])
    if aggregate:
        assert downloads==['partial/candidate-manifest.json']
        assert [row['candidate_id'] for row in result['candidates']]==['aggregate-authoritative']
    else:
        assert set(downloads)=={'partial/source-reports/source-hash.json','partial/sources/vendor.json'}
        assert result['summary']['distinct_candidate_count']==2
        original=next(row for row in result['candidates'] if row['candidate_id']=='reuse-partial')
        assert original['source_native_header']==report['native_header']

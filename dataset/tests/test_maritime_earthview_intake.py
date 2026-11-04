import copy
from types import SimpleNamespace

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.enums import ColorInterp,MaskFlags
from rasterio.transform import Affine

from scripts import maritime_earthview_intake as intake


def test_immutable_cache_singleflight_and_frozen_pin(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    calls=[]
    data=b'original metadata'
    url=intake.ROOT+'test.json'
    pin=dict(url=url,etag='"v1"',bytes=len(data),sha256=intake.digest_bytes(data))
    def fetch(value):
        calls.append(value)
        return data,pin
    cache=intake.ImmutableSourceCache(tmp_path,fetch=fetch)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results=list(pool.map(lambda _:cache.get(url,pin),range(32)))
    assert len(calls)==1 and all(value==(data,pin) for value in results)
    assert intake.ImmutableSourceCache(tmp_path,fetch=lambda _:pytest.fail('cache miss')).get(url,pin)==(data,pin)
    with pytest.raises(ValueError,match='frozen'):
        cache.get(url,dict(pin,sha256='0'*64))


def test_immutable_cache_rejects_corruption(tmp_path):
    url=intake.ROOT+'test.json';data=b'original'
    pin=dict(url=url,etag='v1',bytes=len(data),sha256=intake.digest_bytes(data))
    cache=intake.ImmutableSourceCache(tmp_path,fetch=lambda _:(data,pin))
    cache.get(url)
    (tmp_path/'objects'/pin['sha256']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='corrupt'):
        cache.get(url)


def test_immutable_cache_uses_verified_staged_metadata(tmp_path):
    data=b'{}';url=intake.ROOT+'test.json'
    frozen=dict(url=url,etag='v1',bytes=2,sha256=intake.digest_bytes(data),object={'uri':'gs://test/item'})
    cache=intake.ImmutableSourceCache(tmp_path,fetch=lambda _:pytest.fail('HTTP fetch'),staged=lambda pin:data)
    result,pin=cache.get(url,frozen)
    assert result==data and pin['sha256']==frozen['sha256']


def test_parallel_results_are_ordered_and_bound_pending_population():
    import threading
    first=threading.Event();started=[]
    def process(value):
        started.append(value)
        if value==0:
            assert first.wait(2)
        else:first.set()
        return value
    result=intake.ordered_parallel_results(process,range(100),2)
    assert next(result)==0
    # No unbounded submission of the remaining98 mosaics while consumer paused.
    assert set(started)<=set(range(2))
    assert [0,*result]==list(range(100))


def test_cache_eviction_retains_identity_and_reacquires_exact_original(tmp_path):
    calls=[]
    def fetch(url):
        calls.append(url);data=url[-6:].encode()
        return data,dict(url=url,etag='v1',bytes=len(data),sha256=intake.digest_bytes(data))
    cache=intake.ImmutableSourceCache(tmp_path,fetch=fetch,max_bytes=6)
    first=intake.ROOT+'a.json';second=intake.ROOT+'b.json'
    original,pin=cache.get(first);cache.get(second);cache.trim()
    assert sum(p.stat().st_size for p in (tmp_path/'objects').iterdir())<=6
    assert len(list((tmp_path/'receipts').iterdir()))==2
    assert cache.get(first)==(original,pin)
    assert calls.count(first)==2


def test_evicted_source_cannot_change_identity(tmp_path):
    url=intake.ROOT+'a.json';data=b'original'
    pin=dict(url=url,etag='v1',bytes=len(data),sha256=intake.digest_bytes(data))
    cache=intake.ImmutableSourceCache(tmp_path,fetch=lambda _:(data,pin),max_bytes=1)
    cache.get(url);cache.trim()
    cache.fetch=lambda _:(b'changed',dict(pin,etag='v2',bytes=7,sha256=intake.digest_bytes(b'changed')))
    with pytest.raises(ValueError,match='frozen'):
        cache.get(url)


def test_cache_concurrent_eviction_does_not_invalidate_loaded_bytes(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    def fetch(url):
        data=url.encode();return data,dict(url=url,etag='v1',bytes=len(data),sha256=intake.digest_bytes(data))
    cache=intake.ImmutableSourceCache(tmp_path,fetch=fetch,max_bytes=100)
    urls=[intake.ROOT+str(i%12)+'.json' for i in range(96)]
    def get_and_trim(url):
        data,pin=cache.get(url);cache.trim()
        assert data==url.encode() and pin['sha256']==intake.digest_bytes(data)
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(get_and_trim,urls))
    cache.trim()
    assert sum(p.stat().st_size for p in (tmp_path/'objects').iterdir())<=100


def test_source_fetch_retries_transient_but_not_changed_identity(monkeypatch):
    from urllib.error import HTTPError
    calls=[]
    def transient(url,expected=None):
        calls.append(url)
        if len(calls)<3:raise HTTPError(url,503,'temporary',{},None)
        return b'original',{'etag':'v1'}
    monkeypatch.setattr(intake,'_fetch_pinned_once',transient)
    monkeypatch.setattr(intake.time,'sleep',lambda _:None)
    assert intake.fetch_pinned(intake.ROOT+'test.json')[0]==b'original' and len(calls)==3
    def changed(url,expected=None):raise HTTPError(url,412,'changed',{},None)
    monkeypatch.setattr(intake,'_fetch_pinned_once',changed)
    with pytest.raises(ValueError,match='identity'):
        intake.fetch_pinned(intake.ROOT+'test.json')


def test_retryable_outcome_preserves_attempt_without_final_checkpoint():
    published=[]
    publish=lambda name,data:published.append((name,data))
    retry=dict(sample_id='example',retryable=True,reason='temporary')
    intake.persist_outcome(publish,retry)
    assert published[0][0].startswith('attempts/example/')
    normal=dict(sample_id='example',reason='invalid original support')
    intake.persist_outcome(publish,normal)
    assert published[-1][0]=='results/example.json'


def test_keepalive_source_transport_keeps_conditional_bytes_and_pin(monkeypatch):
    import io
    calls=[];data=b'original'
    class Raw(io.BytesIO):
        def read(self,size=-1,decode_content=False):
            assert decode_content is False
            return super().read(size)
    class Response:
        status_code=200
        headers={'ETag':'v1','Content-Length':str(len(data)),'Last-Modified':'date'}
        raw=Raw(data)
        def __enter__(self):return self
        def __exit__(self,*args):pass
    class Session:
        def head(self,url,**kwargs):calls.append(('HEAD',kwargs));return Response()
        def get(self,url,**kwargs):calls.append(('GET',kwargs));return Response()
    monkeypatch.setattr(intake,'public_session',lambda:Session())
    result,pin=intake.fetch_pinned(intake.ROOT+'test.json',transport='requests-keepalive')
    assert result==data and pin['sha256']==intake.digest_bytes(data)
    assert calls[1][1]['headers']=={'If-Match':'v1'} and calls[1][1]['allow_redirects'] is False
    calls.clear();Response.raw=Raw(data)
    assert intake.fetch_pinned(intake.ROOT+'test.json',expected=pin,transport='requests-keepalive')==(data,pin)
    assert [method for method,_ in calls]==['GET']


def test_staged_metadata_reference_avoids_redundant_copy():
    data=b'{}';ref=dict(uri='gs://test/item',generation='1',bytes=2,crc32c='crc',sha256=intake.digest_bytes(data))
    assert intake.metadata_reference(lambda *args:pytest.fail('redundant copy'),data,{'object':ref})==ref
    with pytest.raises(ValueError,match='Staged'):
        intake.metadata_reference(lambda *args:None,b'changed',{'object':ref})


def items():
    rows=[]
    for y in range(4):
        for x in range(4):
            identity=f'chip-{x}-{y}'
            rows.append(dict(id=identity,properties={'datetime':'2022-08-24T21:15:06.976418+00:00',
                'platform':'newsat29','proj:epsg':'32611','gsd':1.,'satl:product_name':'L1',
                'license':'CC-BY-4.0','proj:shape':[384,384],
                'proj:transform':[1.,0.,10000+x*384,0.,-1.,20000-y*384,0,0,1]},
                assets={'visual':{'href':f'https://satellogic-earthview.s3.us-west-2.amazonaws.com/data/tif/{identity}_VISUAL.tif'}}))
            rows[-1]['assets']['analytic']={'href':f'https://satellogic-earthview.s3.us-west-2.amazonaws.com/data/tif/{identity}_TOA.tif'}
    return rows


def evidence():
    return dict(license='CC-BY-4.0',license_evidence_url='https://registry.opendata.aws/satellogic-earthview/',
        native_resolution_m=1.,native_resolution_evidence_url='https://huggingface.co/datasets/satellogic/EarthView',
        attribution='Satellogic EarthView',upstream_processing_history=['Publisher VISUAL rendering; stretch version unknown'])


def test_exact_16_metadata_identity_and_adjacency():
    plan=intake.validate_mosaic(items(),evidence())
    assert plan['mosaic_bounds']==[10000,18464,11536,20000]
    assert plan['physical_core_bounds']==[10128,18592,11408,19872]
    assert plan['physical_context_bounds']==[10088,18552,11448,19912]
    assert plan['source_pixel_window']==[88,88,1360,1360]
    assert plan['owned_core_pixel_window']==[128,128,1280,1280]
    assert plan['useful_native_rgb_bytes']==1280*1280*3
    assert plan['hr_context_transform']==[2.5,0,10088,0,-2.5,19912]


@pytest.mark.parametrize('key,value',[('datetime','2022-08-24T21:15:06.123456+00:00'),
    ('platform','newsat28'),('proj:epsg','32612'),('gsd',2.),('satl:product_name','L3'),('license','unknown')])
def test_temporal_sensor_grid_or_license_mix_rejected(key,value):
    rows=items();rows[-1]['properties'][key]=value
    with pytest.raises(ValueError):intake.validate_mosaic(rows,evidence())


def test_hole_duplicate_origin_and_rotated_grid_rejected():
    rows=items()
    with pytest.raises(ValueError):intake.validate_mosaic(rows[:-1],evidence())
    rows[-1]=copy.deepcopy(rows[0])
    with pytest.raises(ValueError):intake.validate_mosaic(rows,evidence())
    rows=items();rows[0]['properties']['proj:transform'][1]=.1
    with pytest.raises(ValueError):intake.validate_mosaic(rows,evidence())


def test_stitch_preserves_pixels_and_rejects_invalid_observed_halo():
    rows=items();plan=intake.validate_mosaic(rows,evidence())
    chips={r['id']:np.full((3,384,384),i+20,np.uint8) for i,r in enumerate(rows)}
    masks={key:np.ones((384,384),bool) for key in chips}
    rgb,valid=intake.stitch_original(plan,chips,masks)
    assert rgb.shape==(3,1536,1536)
    np.testing.assert_array_equal(rgb[:,0:384,0:384],chips[rows[0]['id']])
    masks[rows[0]['id']][0,0]=False  # Outside real context, not a contributor.
    rgb,valid=intake.stitch_original(plan,chips,masks)
    context=intake.canonical_context(rgb,valid,plan)
    assert context.shape==(3,544,544) and context.dtype==np.float32
    masks[rows[0]['id']][100,100]=False
    rgb,valid=intake.stitch_original(plan,chips,masks)
    with pytest.raises(ValueError,match='support'):intake.canonical_context(rgb,valid,plan)


def test_declared_nodata_rgb_header_required_not_stac_band_count():
    row=items()[0];t=Affine(*row['properties']['proj:transform'][:6])
    ds=SimpleNamespace(crs=CRS.from_epsg(32611),transform=t,width=384,height=384,count=3,
        dtypes=('uint8',)*3,colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue),nodata=0,
        mask_flag_enums=([MaskFlags.nodata],)*3)
    intake.validate_header(ds,row)
    ds.mask_flag_enums=([MaskFlags.all_valid],)*3
    with pytest.raises(ValueError,match='nodata'):intake.validate_header(ds,row)


def test_cloud_guard_precedes_any_pixel_download(monkeypatch):
    monkeypatch.setattr(intake,'require_cloud_vm',lambda: (_ for _ in ()).throw(RuntimeError('GCE required')))
    with pytest.raises(RuntimeError,match='GCE required'):intake.run(SimpleNamespace())


def test_toa_stitch_and_native_accounting_preserve_uint16():
    rows=items();plan=intake.validate_mosaic(rows,evidence(),product='TOA')
    assert plan['useful_native_rgb_bytes']==1280*1280*3*2
    chips={r['id']:np.full((3,384,384),1234,np.uint16) for r in rows}
    masks={key:np.ones((384,384),bool) for key in chips}
    dn,valid=intake.stitch_original(plan,chips,masks)
    assert dn.dtype==np.uint16 and dn.min()==1234 and valid.all()
    with pytest.raises(ValueError):intake.canonical_context(dn,valid,plan)


def test_fixed_toa_mapping_applies_once_without_perchip_histograms():
    dn=np.broadcast_to(np.array([0,1,100,1000,10000,65535],np.uint16),(3,1,6)).copy()
    expected=np.floor(255*np.clip(dn.astype(np.float64)/10000,0,1)**(1/2.2)+.5).astype(np.uint8)
    np.testing.assert_array_equal(intake.toa_display_rgb(dn),expected)
    assert expected[0,0,0]==0 and expected[0,0,-1]==255
    # Equal original reflectance must produce equal output across chip boundaries.
    chips=np.full((3,1536,1536),1000,np.uint16)
    display=intake.toa_display_rgb(chips)
    assert np.unique(display).tolist()==[90]


def test_toa_header_uses_actual_rgb_band_colorinterp_and_retains_nir():
    row=items()[0];ds=SimpleNamespace(crs=CRS.from_epsg(32611),
        transform=Affine(*row['properties']['proj:transform'][:6]),width=384,height=384,count=4,
        dtypes=('uint16',)*4,colorinterp=(ColorInterp.blue,ColorInterp.green,ColorInterp.red,ColorInterp.undefined),
        nodata=0,mask_flag_enums=([MaskFlags.nodata],)*4)
    assert intake.validate_header(ds,row,product='TOA')==[3,2,1]
    ds.colorinterp=(ColorInterp.gray,)*4
    with pytest.raises(ValueError,match='RGB'):intake.validate_header(ds,row,product='TOA')


def test_native_uint16_context_export_is_lossless(tmp_path):
    import rasterio
    rgb=np.full((3,16,16),12345,np.uint16)
    path=tmp_path/'native-toa.tif'
    intake.write_native_rgb(path,rgb,np.ones((16,16),bool),CRS.from_epsg(32611),Affine(1,0,1,0,-1,17))
    with rasterio.open(path) as ds:
        assert ds.dtypes==('uint16',)*3
        np.testing.assert_array_equal(ds.read(),rgb)


def test_robust_curve_uses_whole_mosaic_valid_pixels_and_saved_parameters():
    dn=np.broadcast_to(np.arange(1536,dtype=np.uint16)[None,None,:]+100,(3,1536,1536)).copy()
    valid=np.ones((1536,1536),bool);valid[0,0]=False;dn[:,0,0]=65535
    display,mapping=intake.map_toa_mosaic(dn,valid,mode='robust')
    for band in range(3):
        low,high=np.percentile(dn[band][valid],[.1,99.9],method='linear').astype(np.float32)
        assert mapping['low_dn'][band]==float(low) and mapping['high_dn'][band]==float(high)
    assert mapping['percentile_method']=='linear' and mapping['parameter_dtype']=='float32'
    np.testing.assert_array_equal(intake.apply_toa_mapping(dn[:,88:1448,88:1448],mapping),display[:,88:1448,88:1448])
    # Recomputing percentiles only on context would change the chosen curve.
    assert mapping['low_dn'][0]<188
    assert display.dtype==np.uint8


def test_robust_mapping_rejects_missing_support_and_degenerate_dynamic_range():
    dn=np.ones((3,1536,1536),np.uint16)*1000
    with pytest.raises(ValueError):intake.map_toa_mosaic(dn,np.zeros((1536,1536),bool),mode='robust')
    with pytest.raises(ValueError):intake.map_toa_mosaic(dn,np.ones((1536,1536),bool),mode='robust')


def test_fixed_mapping_unchanged_when_selected_after_stitch():
    dn=np.ones((3,1536,1536),np.uint16)*1000
    display,mapping=intake.map_toa_mosaic(dn,np.ones((1536,1536),bool),mode='fixed')
    assert mapping==intake.TOA_DISPLAY_MAPPING
    np.testing.assert_array_equal(display,intake.toa_display_rgb(dn))

def test_outcome_spool_is_byte_identical_to_legacy_full_manifest(tmp_path):
    from scripts.maritime_earthview_intake import OutcomeSpool
    candidates=[dict(schema='earthview-candidate-v1',sample_id='a',source_items=[{'pin':'full'}]),dict(schema='earthview-candidate-v1',sample_id='c')]
    rejects=[dict(sample_id='b',reason='invalid')]
    spool=OutcomeSpool(tmp_path)
    for row in (candidates[0],rejects[0],candidates[1]):spool.append(row)
    common=dict(schema='earthview-candidate-v1',configuration={'science':'unchanged'},vm_instance_id='1',status='PROVISIONAL_NATIVE_CANDIDATES',final_dataset_ready=False,quality_accepted_count=0)
    output=spool.finalize(common)
    assert output.read_bytes()==intake.json_bytes(dict(common,candidates=candidates,rejections=rejects,candidate_count=2))
    assert spool.candidate_count==2 and spool.rejection_count==1 and not spool.retryable


def test_outcome_spool_bounds_memory_for1000_large_source_rows(tmp_path):
    import tracemalloc
    from scripts.maritime_earthview_intake import OutcomeSpool
    spool=OutcomeSpool(tmp_path);tracemalloc.start()
    for i in range(1000):spool.append(dict(sample_id=str(i),source_items=['x'*100000],retryable=i==999))
    path=spool.finalize(dict(schema='earthview-candidate-v1'))
    _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    assert spool.rejection_count==1000 and spool.retryable and path.stat().st_size>100_000_000
    assert peak<10_000_000


def test_create_only_file_verifies_full_sha_and_rejects_existing_conflict(tmp_path):
    from scripts.maritime_earthview_intake import create_only_file
    import hashlib
    class Blob:
        generation='1';crc32c='crc'
        def __init__(self):self.data=None
        def exists(self):return self.data is not None
        def upload_from_filename(self,path,**kwargs):
            assert kwargs['if_generation_match']==0;self.data=Path(path).read_bytes()
        def reload(self):self.size=len(self.data)
        def download_to_filename(self,path,**kwargs):
            assert kwargs['if_generation_match']==1;Path(path).write_bytes(self.data)
    from pathlib import Path
    path=tmp_path/'stream.json';path.write_bytes(b'original');blob=Blob();pin=create_only_file(blob,path)
    assert pin['sha256']==hashlib.sha256(b'original').hexdigest()
    blob.data=b'changed!'
    with pytest.raises(ValueError,match='conflict'):create_only_file(blob,path)

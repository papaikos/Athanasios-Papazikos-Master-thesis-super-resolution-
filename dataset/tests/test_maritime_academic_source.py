import copy
import io
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile
from rasterio.crs import CRS
from rasterio.enums import ColorInterp
from rasterio.transform import Affine

from scripts.maritime_academic_raster_intake import planned_cores
from scripts.maritime_cloud_inventory import digest_bytes
from src import maritime_academic_source as source


def fixture():
    ds=SimpleNamespace(crs=CRS.from_epsg(32631),transform=Affine(.5,0,590000,0,-.5,5754000),
        width=6000,height=6000,count=3,dtypes=('uint8',)*3,
        colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue))
    geometry=planned_cores(ds)[0];c,r,w,h=geometry['source_pixel_window']
    rgb=np.full((3,h,w),100,np.uint8)
    x,y,cw,ch=geometry['owned_core_pixel_window']
    pin=dict(uri='gs://test/obj',generation='1',bytes=42,crc32c='ABCD',sha256='a'*64)
    ev=dict(source_id='rotterdam',license='CC-BY-SA-4.0',license_evidence_url='https://spacenet.ai/sn6-challenge/',
        publisher_product_resolution_m=.5,resolution_evidence_url='https://spacenet.ai/sn6-challenge/',
        native_acquisition_gsd_m=None,attribution='SpaceNet / Maxar',upstream_processing_history=['pansharpened RGB'])
    record=dict(schema='academic-raster-candidate-v1',sample_id='academic-test',source_kind='academic_raster',
        source_provenance=ev,license=ev['license'],native_gsd_m=None,publisher_product_resolution_m=.5,
        native_context=pin,hr=pin,source_evidence=pin,source_http_pin={'etag':'abc','bytes':42},
        source_href='https://spacenet-dataset.s3.amazonaws.com/AOIs/test.tif',
        original_transform=list(ds.transform)[:6],native_window_transform=list(ds.transform*Affine.translation(c,r))[:6],
        output_crs='EPSG:32631',source_crs='EPSG:32631',validity_tier='B',
        original_mask_provenance='inferred_rgb_extremes_not_producer_verified',
        mask_inference_uncertainty='Inferred support cannot prove producer validity.',
        canonical_hr_method='exact_metric_fractional_area_float64_then_normalized_float32',
        decoded_native_context_sha256=digest_bytes(rgb.tobytes()),
        decoded_useful_core_sha256=digest_bytes(rgb[:,y-r:y-r+ch,x-c:x-c+cw].tobytes()),**geometry)
    row=dict(sample_id='academic-test',source_kind='academic_raster',source=record,split='train',
        site_id='rotterdam',geographic_group_id='rotterdam',acquisition_id='wv2-rotterdam')
    return row,rgb


def test_candidate_validation_preserves_unknown_native_and_tier_b():
    row,_=fixture();source.validate_source(row['source'])
    assert source.source_geometry(row)==('EPSG:32631',row['source']['physical_core_bounds'])
    assert source.native_bytes(row)==2560*2560*3


@pytest.mark.parametrize('key,value', [('validity_tier','A'),('native_gsd_m',.5),
    ('original_mask_provenance','original_embedded_dataset_mask'),('license','unknown'),
    ('output_crs','EPSG:4326'),('physical_context_bounds',[0,0,1360,1360]),
    ('hr_context_transform',[2.5,0,0,0,-2.5,0]),('mask_inference_uncertainty','')])
def test_inconsistent_or_forged_source_rejected(key,value):
    row,_=fixture();row['source'][key]=value
    with pytest.raises(ValueError):source.validate_source(row['source'])


def test_canonical_context_recomputes_original_pixels_and_stored_hr():
    row,rgb=fixture();record=row['source']
    with MemoryFile() as memory:
        with memory.open(driver='GTiff',width=rgb.shape[2],height=rgb.shape[1],count=3,dtype='uint8',
            crs=record['source_crs'],transform=Affine(*record['native_window_transform'])) as ds:
            ds.write(rgb);ds.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue)
        with memory.open() as ds:
            context,useful,*_=source.canonical_from_native(ds,record)
            assert context.shape==(3,544,544) and context.dtype==np.float32
            assert useful==2560*2560*3
            np.testing.assert_allclose(context,100/255,atol=1e-7)
            buffer=io.BytesIO();np.savez_compressed(buffer,hr_context=context,hr=context[:,16:-16,16:-16])
            source.canonical_from_native(ds,record,buffer.getvalue())
            corrupt=copy.deepcopy(record);corrupt['decoded_native_context_sha256']='b'*64
            with pytest.raises(ValueError,match='hash'):source.canonical_from_native(ds,corrupt)


def test_process_sample_contract_preserves_truthful_common_provenance(monkeypatch):
    from scripts import maritime_ultimate_release
    row,rgb=fixture();record=row['source']
    with MemoryFile() as memory:
        with memory.open(driver='GTiff',width=rgb.shape[2],height=rgb.shape[1],count=3,dtype='uint8',
            crs=record['source_crs'],transform=Affine(*record['native_window_transform'])) as ds:
            ds.write(rgb);ds.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue)
        native=memory.read()
    with MemoryFile(native) as memory, memory.open() as ds:
        context,*_=source.canonical_from_native(ds,record)
    buffer=io.BytesIO();np.savez_compressed(buffer,hr_context=context,hr=context[:,16:-16,16:-16])
    stored=buffer.getvalue()
    record['native_context']=dict(record['native_context'],uri='gs://test/native')
    record['hr']=dict(record['hr'],uri='gs://test/hr')
    class Store:
        def get(self,pin):
            return native if pin['uri']=='gs://test/native' else stored
    called=[]
    def pair(context,common,recipe,seed):
        called.append((recipe,seed));return b'original-recipe-contract',{}
    monkeypatch.setattr(maritime_ultimate_release,'pair_payload',pair)
    common,recomputed,shared,children=source.process_sample(Store(),row,20261006)
    assert common['original_full_support_verified'] is False
    assert common['inferred_full_support_verified'] is True and common['validity_tier']=='B'
    assert common['native_gsd_m'] is None and common['publisher_product_resolution_m']==.5
    assert set(children)=={'d1','d2','d3'} and len(called)==3
    np.testing.assert_array_equal(recomputed,context)
    with np.load(io.BytesIO(shared),allow_pickle=False) as arrays:
        assert arrays['hr'].shape==(3,512,512)
        assert arrays['hr_mask'].all()
    row['split']='test'
    with pytest.raises(ValueError,match='evaluation'):source.process_sample(Store(),row,20261006)


def test_shared_selection_accepts_truthful_academic_source_and_rejects_modified_grid():
    from scripts import maritime_ultimate_release as release
    row,_=fixture()
    document=dict(schema='ultimate-selection-v1',review_policy='explicit inferred tier B expansion; pending independent quality gates',
        splits={'assignment':{'rotterdam':'train'}},selection=[row])
    rows,gaps=release.validate_selection(document)
    assert rows[0]['source_kind']=='academic_raster'
    assert rows[0]['source']['native_gsd_m'] is None and rows[0]['source']['validity_tier']=='B'
    assert isinstance(gaps,list)
    document['selection'][0]['source']['physical_context_bounds']=[0,0,1360,1360]
    with pytest.raises(ValueError):release.validate_selection(document)


def test_shared_geometry_and_accounting_dispatch():
    from scripts import maritime_ultimate_release as release
    row,_=fixture()
    assert release.source_geometry(row)==source.source_geometry(row)
    assert release.native_bytes(row)==source.native_bytes(row)


def test_shared_process_dispatch_preserves_arguments(monkeypatch):
    from scripts import maritime_ultimate_release as release
    row,_=fixture();store=object();expected=('common','context','shared','children')
    called=[]
    def handler(s,r,seed):
        called.append((s,r,seed));return expected
    monkeypatch.setattr(source,'process_sample',handler)
    assert release.process_sample(store,row,20261006)==expected
    assert called==[(store,row,20261006)]

from types import SimpleNamespace

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.enums import ColorInterp
from rasterio.transform import Affine

from scripts import maritime_academic_raster_intake as intake


def header():
    return SimpleNamespace(crs=CRS.from_epsg(32631), transform=Affine(.5, 0, 590011.5, 0, -.5, 5753559.5),
        width=22800, height=16202, count=3, dtypes=('uint8',)*3,
        colorinterp=(ColorInterp.red, ColorInterp.green, ColorInterp.blue))


def evidence():
    return dict(source_id='spacenet-rotterdam', license='CC-BY-SA-4.0',
        license_evidence_url='https://spacenet.ai/sn6-challenge/',
        publisher_product_resolution_m=.5, resolution_evidence_url='https://spacenet.ai/rotterdam/',
        native_acquisition_gsd_m=None, attribution='SpaceNet / DigitalGlobe',
        upstream_processing_history=['publisher pansharpened RGB'])


def test_metric_header_keeps_product_and_native_resolution_separate():
    assert intake.validate_header(header(), evidence()) == .5
    assert evidence()['native_acquisition_gsd_m'] is None
    ds=header(); ds.crs=CRS.from_epsg(4326)
    with pytest.raises(ValueError, match='metric'):
        intake.validate_header(ds,evidence())


@pytest.mark.parametrize('key,value', [('license','unknown'), ('publisher_product_resolution_m',1.1),
    ('license_evidence_url',''), ('resolution_evidence_url',''), ('attribution','')])
def test_license_and_resolution_evidence_required(key,value):
    ev=evidence();ev[key]=value
    with pytest.raises(ValueError):
        intake.validate_evidence(ev)


def test_inference_excludes_black_white_but_preserves_colored_saturation():
    rgb=np.full((3,3,3),100,np.uint8)
    rgb[:,0,0]=0;rgb[:,1,1]=255;rgb[:,2,2]=[255,0,4]
    mask=intake.inferred_support(rgb)
    assert not mask[0,0] and not mask[1,1] and mask[2,2]
    assert mask.sum()==7


def test_no_padding_global_grid_nonoverlap_and_observed_halo():
    ds=header(); geometries=intake.planned_cores(ds)
    assert 0<len(geometries)<=48
    cells={tuple(g['world_grid_cell']) for g in geometries}
    assert len(cells)==len(geometries)
    for g in geometries:
        c,r,w,h=g['source_pixel_window']
        assert c>=0 and r>=0 and c+w<=ds.width and r+h<=ds.height
        assert g['physical_context_bounds'][2]-g['physical_context_bounds'][0]==1360
        assert g['physical_core_bounds'][2]-g['physical_core_bounds'][0]==1280


def test_fractional_area_reduction_and_fail_closed_inferred_support():
    geometry={'area_phase_m':[.1,.2],'product_spacing_m':.8}
    rgb=np.full((3,13,13),100,np.uint8);rgb[:,0,0]=200
    context=intake.canonical_context(rgb,geometry,count=4)
    assert context.shape==(3,4,4) and context.dtype==np.float32
    assert context[0,0,0]==pytest.approx((100+100*.7*.6/6.25)/255,abs=1e-7)
    rgb[:,12,12]=0
    with pytest.raises(ValueError,match='support'):
        intake.canonical_context(rgb,geometry,count=4)


def test_audit_grid_and_http_pins_must_match():
    ds=header()
    audit=dict(crs='EPSG:32631',transform=list(ds.transform),width=ds.width,height=ds.height,
        count=3,dtypes=['uint8']*3,colorinterp=['red','green','blue'],http={'etag':'"abc"','bytes':42})
    intake.validate_audit_grid(ds,audit)
    audit['width']+=1
    with pytest.raises(ValueError,match='audit'):
        intake.validate_audit_grid(ds,audit)


def test_http_pin_checks_bytes_and_official_public_origin(monkeypatch):
    audit=dict(url='https://spacenet-dataset.s3.amazonaws.com/AOIs/test.tif',http={'etag':'abc','bytes':42})
    monkeypatch.setattr(intake,'pinned_http',lambda source: {'etag':'abc','bytes':43})
    with pytest.raises(ValueError,match='byte length'):
        intake.check_http(audit)
    audit['url']='https://unverified.example/test.tif'
    with pytest.raises(ValueError,match='SpaceNet'):
        intake.check_http(audit)


def test_declared_footprint_excludes_context_not_only_core():
    from shapely.geometry import box
    ds=header();all_cores=intake.planned_cores(ds)
    g=all_cores[0]
    assert not intake.planned_cores(ds,box(*g['physical_core_bounds']))
    assert intake.planned_cores(ds,box(*g['physical_context_bounds']))==[g]


def test_lossless_native_export_does_not_claim_producer_mask(tmp_path):
    import rasterio
    rgb=np.full((3,16,16),127,np.uint8)
    path=tmp_path/'original-window.tif'
    transform=Affine(.5,0,123,0,-.5,456)
    intake.write_native(path,rgb,intake.inferred_support(rgb),CRS.from_epsg(32631),transform)
    with rasterio.open(path) as ds:
        np.testing.assert_array_equal(ds.read(),rgb)
        assert ds.transform==transform
        assert ds.dataset_mask().min()==255


def test_cloud_guard_precedes_source_pixel_access(monkeypatch):
    def deny():
        raise RuntimeError('GCE required')
    monkeypatch.setattr(intake,'require_cloud_vm',deny)
    with pytest.raises(RuntimeError,match='GCE required'):
        intake.run(SimpleNamespace())

import copy
import io

import numpy as np
import pytest
from rasterio.io import MemoryFile
from rasterio.enums import ColorInterp
from rasterio.transform import Affine

from scripts.maritime_cloud_inventory import digest_bytes
from scripts.maritime_earthview_intake import validate_mosaic,canonical_context,toa_display_rgb,TOA_DISPLAY_MAPPING,map_toa_mosaic
from src import maritime_earthview_source as source
from test_maritime_earthview_intake import items,evidence


def candidate(product='VISUAL'):
    rows=items();plan=validate_mosaic(rows,evidence(),product=product)
    rgb=np.full((3,1536,1536),1000 if product=='TOA' else 100,np.uint16 if product=='TOA' else np.uint8)
    display=toa_display_rgb(rgb) if product=='TOA' else rgb
    pin=dict(uri='gs://test/obj',generation='1',bytes=42,crc32c='ABCD',sha256='a'*64)
    context=canonical_context(display,np.ones((1536,1536),bool),plan)
    sources=[]
    for cell in plan['cells']:
        item=cell['item'];sources.append(dict(item=item,item_json=pin,object=pin,
            item_http_pin=dict(url='https://satellogic-earthview.s3.us-west-2.amazonaws.com/data/json/'+item['id']+'.json',etag='json-etag',bytes=42,sha256='a'*64),
            http_pin=dict(url=item['assets']['analytic' if product=='TOA' else 'visual']['href'],etag='tif-etag',bytes=42,sha256='a'*64),
            selected_rgb_band_indices=[1,2,3],
            decoded_rgb_sha256=digest_bytes(np.full((3,384,384),1000 if product=='TOA' else 100,np.uint16 if product=='TOA' else np.uint8).tobytes()),
            declared_nodata_mask_sha256=digest_bytes(np.ones((384,384),bool).tobytes()),
            mosaic_pixel_offset=[cell['col'],cell['row']]))
    record=dict(schema='earthview-candidate-v1',sample_id='earthview-test',source_kind='earthview',
        source_provenance=evidence(),source_items=sources,native_context=pin,hr=pin,
        license='CC-BY-4.0',native_gsd_m=1.,original_mask_provenance='original_declared_nodata_zero_rgb_band_masks',
        mask_caveat='Producer-declared per-band nodata; not separate producer dataset mask.',
        upstream_visual_rendering='Publisher VISUAL historical stretch version unknown.',
        rgb_mapping=copy.deepcopy(TOA_DISPLAY_MAPPING) if product=='TOA' else None,
        decoded_native_context_sha256=digest_bytes(rgb[:,88:1448,88:1448].tobytes()),
        decoded_useful_core_sha256=digest_bytes(rgb[:,128:1408,128:1408].tobytes()),
        decoded_display_context_sha256=digest_bytes(display[:,88:1448,88:1448].tobytes()),
        decoded_display_core_sha256=digest_bytes(display[:,128:1408,128:1408].tobytes()),
        requires_radiometry_review=True,final_eligible=False,
        **{k:v for k,v in plan.items() if k!='cells'})
    row=dict(sample_id=record['sample_id'],source_kind='earthview',source=record,split='train',
        site_id='port',geographic_group_id='port',acquisition_id='newsat29-20220824')
    return row,rgb,context


def test_validate_truthful_same_acquisition_geometry():
    row,_,_=candidate();source.validate_source(row['source'])
    assert source.native_bytes(row)==1280*1280*3
    assert source.source_geometry(row)==('EPSG:32611',[10128,18592,11408,19872])


@pytest.mark.parametrize('key,value',[('native_gsd_m',.5),('license','unknown'),
    ('original_mask_provenance','original_embedded_dataset_mask'),('mosaic_shape',[1600,1600]),
    ('physical_context_bounds',[0,0,1360,1360]),('source_pixel_window',[0,0,1360,1360])])
def test_source_forgery_and_unknown_geometry_rejected(key,value):
    row,_,_=candidate();row['source'][key]=value
    with pytest.raises(ValueError):source.validate_source(row['source'])


def test_native_canonical_reproduction_and_hash():
    row,rgb,expected=candidate();record=row['source'];native=rgb[:,88:1448,88:1448]
    with MemoryFile() as memory:
        with memory.open(driver='GTiff',width=1360,height=1360,count=3,dtype='uint8',
            crs=record['source_crs'],transform=Affine(*record['native_context_transform'])) as ds:
            ds.write(native);ds.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue)
            ds.write_mask(np.ones((1360,1360),np.uint8)*255)
        with memory.open() as ds:
            buffer=io.BytesIO();np.savez_compressed(buffer,hr_context=expected,hr=expected[:,16:-16,16:-16])
            context,useful,*_=source.canonical_from_native(ds,record,buffer.getvalue())
            np.testing.assert_array_equal(context,expected)
            assert useful==1280*1280*3
            changed=copy.deepcopy(record);changed['decoded_useful_core_sha256']='b'*64
            with pytest.raises(ValueError,match='hash'):source.canonical_from_native(ds,changed)


def test_toa_original_native_uint16_reproduces_fixed_display_hr():
    row,dn,expected=candidate('TOA');record=row['source']
    source.validate_source(record)
    assert source.native_bytes(row)==1280*1280*3*2
    with MemoryFile() as memory:
        with memory.open(driver='GTiff',width=1360,height=1360,count=3,dtype='uint16',
            crs=record['source_crs'],transform=Affine(*record['native_context_transform'])) as ds:
            ds.write(dn[:,88:1448,88:1448]);ds.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue)
            ds.write_mask(np.ones((1360,1360),np.uint8)*255)
        with memory.open() as ds:
            result,useful,*_=source.canonical_from_native(ds,record)
            np.testing.assert_array_equal(result,expected)
            assert useful==1280*1280*3*2
    modified=copy.deepcopy(record);modified['rgb_mapping']['gamma']=1.8
    with pytest.raises(ValueError,match='mapping'):source.validate_source(modified)


def test_original16_toa_files_stac_pins_and_canonical_hr_reproduced():
    from scripts.maritime_cloud_inventory import json_bytes
    row,dn,expected=candidate('TOA');record=row['source'];objects={}
    for number,chip in enumerate(record['source_items']):
        item=chip['item'];raw_item=json_bytes(item)
        with MemoryFile() as memory:
            with memory.open(driver='GTiff',width=384,height=384,count=4,dtype='uint16',nodata=0,
                crs=record['source_crs'],transform=Affine(*item['properties']['proj:transform'][:6])) as ds:
                ds.write(np.full((4,384,384),1000,np.uint16))
                ds.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue,ColorInterp.undefined)
            raw=memory.read()
        for key,data,name in [('item_json',raw_item,'item'),('object',raw,'object')]:
            chip[key]=dict(chip[key],uri=f'gs://test/{name}-{number}',bytes=len(data),sha256=digest_bytes(data))
            objects[chip[key]['uri']]=data
        chip['item_http_pin'].update(bytes=len(raw_item),sha256=digest_bytes(raw_item))
        chip['http_pin'].update(bytes=len(raw),sha256=digest_bytes(raw))
    class Store:
        def get(self,pin):return objects[pin['uri']]
    result=source.verify_original_assembly(Store(),record)
    np.testing.assert_array_equal(result,expected)
    corrupted=copy.deepcopy(record);corrupted['source_items'][0]['http_pin']['sha256']='b'*64
    with pytest.raises(ValueError,match='hash'):source.verify_original_assembly(Store(),corrupted)


def test_robust_saved_curve_applied_without_context_percentile_recomputation():
    row,_,_=candidate('TOA');record=row['source']
    dn=np.broadcast_to(np.arange(1536,dtype=np.uint16)[None,None,:]+100,(3,1536,1536)).copy()
    display,mapping=map_toa_mosaic(dn,np.ones((1536,1536),bool),mode='robust')
    record['rgb_mapping']=mapping
    record['decoded_native_context_sha256']=digest_bytes(dn[:,88:1448,88:1448].tobytes())
    record['decoded_useful_core_sha256']=digest_bytes(dn[:,128:1408,128:1408].tobytes())
    record['decoded_display_context_sha256']=digest_bytes(display[:,88:1448,88:1448].tobytes())
    record['decoded_display_core_sha256']=digest_bytes(display[:,128:1408,128:1408].tobytes())
    plan=validate_mosaic(items(),evidence(),product='TOA')
    expected=canonical_context(display,np.ones((1536,1536),bool),plan)
    with MemoryFile() as memory:
        with memory.open(driver='GTiff',width=1360,height=1360,count=3,dtype='uint16',
            crs=record['source_crs'],transform=Affine(*record['native_context_transform'])) as ds:
            ds.write(dn[:,88:1448,88:1448]);ds.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue)
            ds.write_mask(np.ones((1360,1360),np.uint8)*255)
        with memory.open() as ds:
            actual,*_=source.canonical_from_native(ds,record)
            np.testing.assert_array_equal(actual,expected)


def test_shared_earthview_selection_dispatch():
    from scripts import maritime_ultimate_release as release
    row,_,_=candidate('TOA')
    document=dict(schema='ultimate-selection-v1',review_policy='Exact original TOA assembly; pending quality gates',
        splits={'assignment':{'port':'train'}},selection=[row])
    rows,gaps=release.validate_selection(document)
    assert rows[0]['source']['source_rgb_dtype']=='uint16'
    assert release.source_geometry(row)==source.source_geometry(row)
    assert release.native_bytes(row)==1280*1280*3*2


def test_shared_earthview_process_dispatch(monkeypatch):
    from scripts import maritime_ultimate_release as release
    row,_,_=candidate('TOA');expected=('common','context','mother','children');store=object()
    monkeypatch.setattr(source,'process_sample',lambda s,r,seed: expected if s is store and r is row and seed==20261006 else None)
    assert release.process_sample(store,row,20261006)==expected

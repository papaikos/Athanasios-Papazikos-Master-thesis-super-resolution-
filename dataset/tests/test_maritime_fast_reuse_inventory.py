import pytest
from scripts.maritime_fast_reuse_inventory import unique_rows, usable_sources, virtual_row


def test_dedup_keeps_one_acquisition_world_core():
    rows=[dict(sample_id='a',site_id='s',acquisition_id='x',crs='EPSG:32631',core_bounds=[0,0,1280,1280]),
          dict(sample_id='b',site_id='s',acquisition_id='x',crs='EPSG:32631',core_bounds=[0,0,1280,1280])]
    assert len(unique_rows(rows,781))==1


def test_site_cap_does_not_count_duplicates():
    a=dict(sample_id='a',site_id='s',acquisition_id='x',crs='EPSG:32631',core_bounds=[0,0,1280,1280])
    b=dict(a,sample_id='b',core_bounds=[1280,0,2560,1280])
    assert unique_rows([a,a,b],2)==[a,b]


def test_development_and_unapproved_are_excluded():
    rows=[dict(asset_id='a',site_id='s'),dict(asset_id='b',site_id='s'),dict(asset_id='a',site_id='dev')]
    assert usable_sources(rows,{'a'},{'dev'})==[rows[0]]


def test_virtual_source_pin_keeps_actual_resolution_and_window():
    source=dict(uri='gs://bucket/object',generation='1',bytes=20,crc32c='abcd',site_id='s',catalog_id='c',asset_id='a')
    core=dict(global_core_identity={'catalog_id':'c','cell':[1,2]},crs='EPSG:32631',context_window=[0,0,1700,1700],
              core_window=[50,50,1600,1600],core_bounds=[0,0,1280,1280],context_transform=[.8,0,-40,0,-.8,1320],
              mask_provenance='legacy_nodata_zero_conservative')
    row=virtual_row(source,core,'a'*64)
    assert row['source']['source_object']['sha256']=='a'*64
    assert row['source']['native_gsd_m']==.8
    assert row['source']['context_window']==[0,0,1700,1700]
    assert row['source']['core_bytes']==7680000
    assert row['vessel_presence']=='unconfirmed'


def test_missing_actual_source_checksum_rejected():
    with pytest.raises(ValueError):
        virtual_row({}, {}, '')


@pytest.mark.parametrize('source_count,expected_complete,remaining',[(2,False,1),(1,True,0)])
def test_census_completion_requires_queue_exhaustion_even_at_cap(tmp_path,monkeypatch,source_count,expected_complete,remaining):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from google.cloud import storage
    from scripts import maritime_fast_reuse_inventory as census
    import json
    sources=[dict(asset_id='a',site_id='s',uri=f'gs://bucket/{i}') for i in range(source_count)]
    path=tmp_path/'input.json';path.write_text('{}')
    monkeypatch.setattr(census,'require_cloud_vm',lambda:'vm')
    monkeypatch.setattr(census,'source_rows',lambda *args:sources)
    monkeypatch.setattr(census,'approved_asset_ids',lambda *args:{'a'})
    monkeypatch.setattr(census,'create_only',lambda *args:{})
    monkeypatch.setattr(storage,'Client',lambda:SimpleNamespace(bucket=lambda name:
        SimpleNamespace(blob=lambda name:SimpleNamespace(exists=lambda:False))))
    downloaded=[]
    def download(client,source,path):
        downloaded.append(source['uri']);return 'a'*64
    monkeypatch.setattr(census,'download_pinned',download)
    monkeypatch.setattr(census.rasterio,'open',lambda *args:nullcontext(object()))
    monkeypatch.setattr(census,'supported_windows',lambda *args,**kwargs:[{}])
    monkeypatch.setattr(census,'virtual_row',lambda source,*args:dict(sample_id=source['uri'],
        site_id='s',acquisition_id=source['uri'],crs='EPSG:32631',core_bounds=[0,0,1280,1280]))
    result=census.run(SimpleNamespace(sources=str(path),approved_assets=str(path),workers=1,
        site_cap=1,scratch=str(tmp_path/'scratch'),output_prefix='gs://bucket/new'))
    assert len(downloaded)==1
    assert result['candidate_count']==1
    assert result['source_census_complete'] is expected_complete
    assert result['remaining_queued_source_count']==remaining
    assert result['remaining_queued_source_counts_by_site']==({'s':1} if remaining else {})
    assert result['site_cap_reached_sites']==['s']
    checkpoint=json.loads((tmp_path/'scratch'/'checkpoint.json').read_text())
    assert checkpoint['source_census_complete'] is expected_complete
    assert checkpoint['remaining_queued_source_count']==remaining

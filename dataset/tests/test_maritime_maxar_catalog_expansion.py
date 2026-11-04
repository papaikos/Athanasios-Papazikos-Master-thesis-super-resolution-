import copy
import math
import pytest
from pyproj import Transformer
from scripts import maritime_maxar_catalog_expansion as catalog


def fixture():
    lon,lat=Transformer.from_crs(32631,4326,always_xy=True).transform(500000,5000)
    ports=catalog.normalize_ports({'features':[{'attributes':dict(index_no=1,port_name='PORT',country='FR',
        harbortype='CN',harborsize='L',longitude=lon,latitude=lat)}]}, {'sha256':'a'*64})
    geom={'type':'Polygon','coordinates':[[[500000,0],[505000,0],[505000,5000],[500000,5000],[500000,0]]]}
    visual={'href':'./capture-visual.tif','proj:shape':[17408,17408],
        'proj:transform':[.30517578125,0,499843.75,0,-.30517578125,5156.25,0,0,1],
        'eo:bands':[{'common_name':name} for name in ('red','green','blue')]}
    item={'type':'Feature','id':'31/quad/capture','geometry':geom,'bbox':[2.9,0,3.1,.05],
        'properties':{'catalog_id':'capture','gsd':.5,'platform':'WV03','proj:epsg':32631,
                      'proj:geometry':geom,'tile:clouds_percent':0,'grid:code':'grid','datetime':'2020-01-01Z'},
        'assets':{'visual':visual}}
    ref={'url':'https://maxar-opendata.s3.amazonaws.com/events/event/ard/31/quad/2020-01-01/capture.json'}
    return ports,item,ref


def test_normalize_ports_excludes_river_harbors_and_orders_large_medium_first():
    features=[{'attributes':dict(index_no=i,port_name='p',country='XX',harbortype=t,harborsize=s,
                                latitude=0,longitude=3)} for i,t,s in [(1,'CN','S'),(2,'OR','M'),(3,'CB','L'),(4,'RN','L')]]
    result=catalog.normalize_ports({'features':features},{})
    assert [p['port_id'] for p in result]==['3','2','1']
    assert all(p['source_query']=={} for p in result)


def test_projected_valid_footprint_distance_includes_boundary_and_rejects_outside():
    ports,item,ref=fixture()
    point=Transformer.from_crs(32631,4326,always_xy=True)
    for x,expected in [(492000,1),(491999,0)]:
        p=dict(ports[0]);p['longitude'],p['latitude']=point.transform(x,2500)
        assert len(catalog.port_matches(item,[p]))==expected


def test_source_mapping_preserves_native_gsd_rgb_and_unresolved_license_conflict():
    ports,item,ref=fixture()
    source=catalog.source_from_item(item,ref,{'id':'event','license':'CC-BY-NC-4.0'}, {},
        {'license':'proprietary'},{},ports)
    assert source['native_gsd_m']==.5 and source['visual_pixel_size_m']==[.30517578125]*2
    assert source['rgb_band_names']==['red','green','blue']
    assert source['license_conflict'] is True
    assert source['event_license']=='CC-BY-NC-4.0'
    assert source['visual_href'].endswith('/capture-visual.tif')
    assert source['port_AOI_matches'][0]['distance_from_valid_item_footprint_m']==0
    assert source['final_eligible'] is False and source['vessel_presence']=='unconfirmed'
    assert source['field_provenance']['native_gsd_m']=='item.properties.gsd'


@pytest.mark.parametrize('key,value',[('gsd',None),('gsd',1.1),('gsd',math.nan),
    ('platform','WV01'),('tile:clouds_percent',None),('tile:clouds_percent',11),('tile:clouds_percent',-1)])
def test_source_rejects_unverified_or_coarse_metadata(key,value):
    ports,item,ref=fixture();item['properties'][key]=value
    with pytest.raises(ValueError):
        catalog.source_from_item(item,ref,{'id':'event','license':'CC-BY-NC-4.0'}, {},{}, {},ports)


def test_source_requires_rgb_event_license_and_valid_projected_geometry():
    ports,item,ref=fixture()
    for changed,event in [(copy.deepcopy(item),{'id':'event','license':'proprietary'}),
                          (copy.deepcopy(item),{'id':'event','license':'CC-BY-NC-4.0'})]:
        if event['license']=='CC-BY-NC-4.0': changed['properties']['proj:geometry']['type']='LineString'
        with pytest.raises(ValueError): catalog.source_from_item(changed,ref,event,{}, {},{},ports)
    item['assets']['visual']['eo:bands'][0]['common_name']='gray'
    with pytest.raises(ValueError): catalog.source_from_item(item,ref,{'id':'event','license':'CC-BY-NC-4.0'}, {},{}, {},ports)


def test_scheduling_cap_does_not_claim_catalog_incomplete():
    ports,item,ref=fixture();a={'source_id':'a','port_AOI_matches':ports,'tile_cloud_percent':0}
    b=dict(a,source_id='b')
    result=catalog.schedule_sources([b,a],1)
    assert result['sources']==[a] and result['available_source_count']==2
    assert result['selection_cap_reached'] is True and result['unscheduled_source_count']==1


def test_metadata_url_guard_blocks_raster_and_unofficial_downloads():
    assert catalog.official_json('https://maxar-opendata.s3.amazonaws.com/events/catalog.json')
    for url in ['https://evil.example/events/catalog.json','https://maxar-opendata.s3.amazonaws.com/events/x.tif']:
        with pytest.raises(ValueError): catalog.official_json(url)


def test_exact_url_cache_receipt_reuses_pinned_bytes_and_rejects_tampering(tmp_path):
    legacy=tmp_path/'legacy';legacy.mkdir()
    data=catalog.encoded({'type':'Collection','id':'event'})
    pin=dict(url='https://maxar-opendata.s3.amazonaws.com/events/event/collection.json',sha256=catalog.sha(data),bytes=len(data))
    original=legacy/(pin['sha256']+'.json');original.write_bytes(data)
    cache=catalog.MetadataCache(tmp_path/'new',legacy,[pin])
    document,receipt=cache.get(pin['url'],'Collection','event')
    assert document['id']=='event' and receipt['cached_metadata_response'] is True
    assert receipt['cache_identity_check']=='exact prior pinned URL receipt'
    assert original.read_bytes()==data
    (tmp_path/'new'/(pin['sha256']+'.json')).write_bytes(b'{}')
    with pytest.raises(ValueError,match='cache bytes'):cache.get(pin['url'],'Collection','event')


def test_unique_stac_identity_cache_requires_matching_type_and_id(tmp_path):
    legacy=tmp_path/'legacy';legacy.mkdir()
    data=catalog.encoded({'type':'Collection','id':'event'})
    (legacy/(catalog.sha(data)+'.json')).write_bytes(data)
    cache=catalog.MetadataCache(tmp_path/'new',legacy,[])
    document,receipt=cache.get('https://maxar-opendata.s3.amazonaws.com/events/event/collection.json','Collection','event',True)
    assert document['id']=='event'
    assert 'unique' in receipt['cache_identity_check']


def test_conflicting_prior_url_receipts_are_not_treated_as_unique(tmp_path):
    legacy=tmp_path/'legacy';legacy.mkdir();pins=[]
    for version in (1,2):
        data=catalog.encoded({'type':'Collection','id':'event','version':version})
        (legacy/(catalog.sha(data)+'.json')).write_bytes(data)
        pins.append(dict(url='https://maxar-opendata.s3.amazonaws.com/events/event/collection.json',
                         sha256=catalog.sha(data),bytes=len(data)))
    cache=catalog.MetadataCache(tmp_path/'new',legacy,pins)
    assert pins[0]['url'] not in cache.known
    assert pins[0]['url'] in cache.ambiguous_urls


def test_visual_head_identity_matches_parent_etag_and_bytes():
    headers={'Content-Length':'20','ETag':'"original"','Accept-Ranges':'bytes'}
    prior={'content_length':20,'etag':'"original"'}
    identity=catalog.visual_identity(headers,200,prior)
    assert identity['bytes']==20 and identity['etag']=='"original"'
    for changed in [dict(headers,ETag='"changed"'),dict(headers,**{'Content-Length':'21'})]:
        with pytest.raises(ValueError,match='changed'):catalog.visual_identity(changed,200,prior)
    with pytest.raises(ValueError,match='incomplete'):catalog.visual_identity({},200)


def test_partial_cache_reconstruction_requires_all_pinned_context_and_marks_novelty(tmp_path):
    ports,item,ref=fixture();directory=tmp_path/'metadata';directory.mkdir()
    receipts=directory/'url-receipts';receipts.mkdir()
    documents=[(ref['url'],item),
        ('https://maxar-opendata.s3.amazonaws.com/events/event/collection.json',{'type':'Collection','id':'event','license':'CC-BY-NC-4.0'}),
        ('https://maxar-opendata.s3.amazonaws.com/events/event/ard/acquisition_collections/capture_collection.json',{'type':'Collection','id':'capture','license':'CC-BY-NC-4.0'})]
    for url,document in documents:
        data=catalog.encoded(document);digest=catalog.sha(data)
        (directory/(digest+'.json')).write_bytes(data)
        (receipts/(catalog.sha(url.encode())+'.json')).write_bytes(catalog.encoded(dict(url=url,sha256=digest,bytes=len(data))))
    rows,errors,counts=catalog.cached_source_rows(directory,ports,set())
    assert len(rows)==1 and not errors and counts['cached_item_metadata_examined']==1
    assert rows[0]['novel_source_id_vs_parents'] is True
    rows,errors,counts=catalog.cached_source_rows(directory,ports,{'capture_grid'})
    assert rows[0]['novel_source_id_vs_parents'] is False
    event_url=documents[1][0]
    (receipts/(catalog.sha(event_url.encode())+'.json')).unlink()
    rows,errors,counts=catalog.cached_source_rows(directory,ports,set())
    assert not rows and errors

@pytest.mark.parametrize('zone,canonical', [('01','1'),('04','4'),('17','17')])
def test_official_item_url_identity_normalizes_numeric_utm_zone_only(zone, canonical):
    url=f'https://maxar-opendata.s3.amazonaws.com/events/event/ard/{zone}/300222120103/2022-01-18/1040010072C90800.json'
    assert catalog.item_identity(url)==f'{canonical}/300222120103/1040010072C90800'


def test_explicit_cloud_ceiling_preserves_original_stac_cloud_and_default():
    ports,item,ref=fixture();item['properties']['tile:clouds_percent']=30
    original=copy.deepcopy(item)
    with pytest.raises(ValueError,match='cloud_not_verified_le10'):
        catalog.source_from_item(item,ref,{'id':'event','license':'CC-BY-NC-4.0'}, {},{}, {},ports)
    source=catalog.source_from_item(item,ref,{'id':'event','license':'CC-BY-NC-4.0'}, {},{}, {},ports,
                                   max_tile_cloud_percent=30)
    assert source['tile_cloud_percent']==30 and item==original
    assert source['final_eligible'] is False and source['vessel_presence']=='unconfirmed'
    item['properties']['tile:clouds_percent']=30.01
    with pytest.raises(ValueError,match='cloud_not_verified_le30'):
        catalog.source_from_item(item,ref,{'id':'event','license':'CC-BY-NC-4.0'}, {},{}, {},ports,
                                 max_tile_cloud_percent=30)


@pytest.mark.parametrize('ceiling',[True,-1,101,math.nan,math.inf,'50'])
def test_cloud_ceiling_requires_finite_numeric_percentage(ceiling):
    ports,item,ref=fixture()
    with pytest.raises(ValueError,match='Invalid tile cloud ceiling'):
        catalog.source_from_item(item,ref,{'id':'event','license':'CC-BY-NC-4.0'}, {},{}, {},ports,
                                 max_tile_cloud_percent=ceiling)


def test_cached_reconstruction_applies_explicit_ceiling_without_changing_originals(tmp_path):
    ports,item,ref=fixture();item['properties']['tile:clouds_percent']=50
    directory=tmp_path/'metadata';directory.mkdir();receipts=directory/'url-receipts';receipts.mkdir()
    documents=[(ref['url'],item),
        ('https://maxar-opendata.s3.amazonaws.com/events/event/collection.json',{'type':'Collection','id':'event','license':'CC-BY-NC-4.0'}),
        ('https://maxar-opendata.s3.amazonaws.com/events/event/ard/acquisition_collections/capture_collection.json',{'type':'Collection','id':'capture','license':'CC-BY-NC-4.0'})]
    originals={}
    for url,document in documents:
        data=catalog.encoded(document);digest=catalog.sha(data);originals[digest]=data
        (directory/(digest+'.json')).write_bytes(data)
        (receipts/(catalog.sha(url.encode())+'.json')).write_bytes(catalog.encoded(dict(url=url,sha256=digest,bytes=len(data))))
    rows,errors,counts=catalog.cached_source_rows(directory,ports,set())
    assert not rows and not errors and counts['metadata_rejected']==1
    rows,errors,counts=catalog.cached_source_rows(directory,ports,set(),max_tile_cloud_percent=50)
    assert len(rows)==1 and not errors and rows[0]['tile_cloud_percent']==50
    assert all((directory/(digest+'.json')).read_bytes()==raw for digest,raw in originals.items())

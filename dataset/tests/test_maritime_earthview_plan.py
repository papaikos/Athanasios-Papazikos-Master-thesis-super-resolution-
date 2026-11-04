import copy
import json
import pytest
from scripts import maritime_earthview_plan as plan


def test_invalid_projected_coast_union_is_unknown_and_rejected():
 from shapely.geometry import Polygon
 from shapely.ops import unary_union
 class InvalidProjectedCoast:
  def classify(self,record,margin_m):
   assert margin_m==1000
   return unary_union([Polygon([(0,0),(2,2),(0,2),(2,0),(0,0)]),Polygon([(0,0),(3,0),(3,3),(0,3)])])
 result=plan.screen_marine(InvalidProjectedCoast(),mosaic())
 assert result['status']=='UNKNOWN_INVALID_GEOMETRY'
 assert result['marine_area_m2'] is None and not plan.marine_allowed(result)
 assert 'TopologyException' in result['geometry_error']


def mosaic(port='1',country='AA',date='2022-01-01',x=500000,size='M'):
 return dict(epsg=32631,core_bounds=[x,0,x+1280,1280],context_bounds=[x-40,-40,x+1320,1320],mosaic_bounds=[x-128,-128,x+1408,1408],acquisition_filename=date,port=dict(id=port,name=port,country=country,size=size),json_paths=[f'data/metadata/{date}-{i}.json' for i in range(16)])


def test_training_protection_exact_metric_4km_and_prior_quality_geometry():
 lock=dict(crs='EPSG:32631',core_bounds=[500000,0,501280,1280],acquisition_id='held',site_id='maxar-port-9')
 index=plan.ProtectedIndex([lock],[dict(crs='EPSG:32631',core_bounds=[520000,0,521280,1280])])
 assert index.reject(mosaic(x=505279))=='protected_core_within_4km'
 assert index.reject(mosaic(x=505281)) is None
 assert index.reject(mosaic(port='9',x=550000))=='protected_port_site'
 assert index.reject(mosaic(date='held',x=550000))=='protected_acquisition'
 assert index.reject(mosaic(x=520100))=='prior_quality_rejected_core_overlap'


def test_diverse_order_unique_geography_before_repeat_dates_and_countries():
 a=mosaic('1','AA','1');b=mosaic('2','BB','1',x=520000);c=mosaic('1','AA','2');d=mosaic('3','AA','1',x=550000,size='L')
 rows=plan.diverse_order([a,c,d,b],{plan.physical_key(a),plan.physical_key(b),plan.physical_key(d)})
 assert rows.index(c)==3
 assert {r['port']['country'] for r in rows[:2]}=={'AA','BB'}
 assert len(rows)==4


def test_verified_mosaic_requires_index_geometry_and_exact_full_timestamp():
 evidence=dict(license='CC-BY-4.0',native_resolution_m=1.,license_evidence_url='https://registry.opendata.aws/satellogic-earthview/',native_resolution_evidence_url='https://huggingface.co/datasets/satellogic/EarthView',attribution='Satellogic',upstream_processing_history=['original TOA'])
 row=mosaic();items=[]
 for y in range(4):
  for x in range(4):
   items.append(dict(id=f'{x}-{y}',properties={'datetime':'2022-01-01T01:02:03.456+00:00','platform':'newsat','proj:epsg':32631,'gsd':1.,'satl:product_name':'L1','license':'CC-BY-4.0','proj:shape':[384,384],'proj:transform':[1.,0.,499872+x*384,0.,-1.,1408-y*384,0,0,1]},assets={'analytic':{'href':f'https://satellogic-earthview.s3.us-west-2.amazonaws.com/{x}-{y}_TOA.tif'}}))
 verified=plan.verify_plan(row,items,evidence)
 assert verified['input_product']=='TOA' and verified['planned_split']=='train' and verified['final_eligible'] is False
 assert verified['acquisition_identity'][0]=='2022-01-01T01:02:03.456+00:00'
 mixed=copy.deepcopy(items);mixed[-1]['properties']['datetime']='2022-01-01T01:02:03.457+00:00'
 with pytest.raises(ValueError):plan.verify_plan(row,mixed,evidence)
 wrong=dict(row,core_bounds=[500001,0,501281,1280])
 with pytest.raises(ValueError,match='index geometry'):plan.verify_plan(wrong,items,evidence)


def test_cache_exact_url_and_sha_tamper_fails_closed(tmp_path):
 raw=b'{"id":"x"}';pin=dict(url='https://satellogic-earthview.s3.us-west-2.amazonaws.com/data/x.json',bytes=len(raw),etag='"x"',sha256=plan.sha(raw))
 (tmp_path/(pin['sha256']+'.json')).write_bytes(raw)
 assert plan.cached_metadata(tmp_path,pin)[0]=={'id':'x'}
 (tmp_path/(pin['sha256']+'.json')).write_bytes(b'{}')
 with pytest.raises(ValueError,match='pin'):plan.cached_metadata(tmp_path,pin)


def test_marine_land_interior_rejection_and_unknown_not_marine_claim():
 assert plan.marine_allowed(dict(status='INLAND_FAR_FROM_MARINE_COAST',shore_distance_m=1001,marine_area_m2=0)) is False
 assert plan.marine_allowed(dict(status='MARINE_INTERSECTION',shore_distance_m=0,marine_area_m2=10)) is True
 assert plan.marine_allowed(dict(status='COAST_MARGIN_UNCERTAIN',shore_distance_m=1000,marine_area_m2=0)) is True
 assert plan.marine_allowed(dict(status='UNKNOWN_INVALID_GEOMETRY',shore_distance_m=None,marine_area_m2=None)) is False


def test_protected_geometry_matches_across_projected_crs_without_pyproj():
 from rasterio.warp import transform
 lock=dict(crs='EPSG:32631',core_bounds=[500000,0,501280,1280])
 xs,ys=transform('EPSG:32631','EPSG:32632',[500000,501280],[0,1280])
 row=mosaic();row['epsg']=32632;row['core_bounds']=[min(xs),min(ys),max(xs),max(ys)]
 assert plan.ProtectedIndex([lock]).reject(row)=='protected_core_within_4km'


def test_geographic_proxy_priority_precedes_other_new_locations():
 a=mosaic('1','AA','1');b=mosaic('2','BB','1',x=520000);other=mosaic('3','AA','1',x=530000,size='L')
 ordered=plan.diverse_order([other,a,b],{plan.physical_key(a),plan.physical_key(b)})
 assert ordered[-1]==other


def test_resume_requires_same_scientific_input_pins_and_training_only_plans():
 row=mosaic();row.update(planned_split='train',final_eligible=False,input_product='TOA')
 pins=dict(input_sha256='a',distinct_sha256='b',locks_sha256='c',gshhg={'sha256':'d'})
 document=dict(schema='earthview-source-mosaics-v1',mosaics=[row],metadata_pins={},split_exclusion_evidence=pins)
 assert len(plan.validate_resume(document,pins))==1
 wrong=dict(pins,locks_sha256='changed')
 with pytest.raises(ValueError,match='input'):plan.validate_resume(document,wrong)
 row['planned_split']='test'
 with pytest.raises(ValueError,match='training'):plan.validate_resume(document,pins)


def test_site_scheduling_cap_default_and_explicit_uncapped_resume():
 assert plan.site_cap_reached(999,1000) is False
 assert plan.site_cap_reached(1000,1000) is True
 assert plan.site_cap_reached(1001,1000) is True
 # Already-capped immutable parent sites may grow in an explicit uncapped run.
 assert plan.site_cap_reached(1000,None) is False
 assert plan.site_cap_reached(1000000,None) is False

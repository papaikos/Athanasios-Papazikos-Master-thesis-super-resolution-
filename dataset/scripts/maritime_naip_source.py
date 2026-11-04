"""USDA NAIP original orthomosaic metadata; camera capture spacing is unknown."""
from datetime import datetime
import math,re
from urllib.parse import urlsplit
from rasterio.crs import CRS

def normalize_item(item,dataset):
    if dataset.get('license')!='Public-Domain' or not dataset.get('license_evidence_url') or not dataset.get('corroborating_license_url'):
        raise ValueError('Affirmative official Public Domain evidence required')
    if dataset.get('collection_license_link_title')!='Public Domain':
        raise ValueError('Explicit NAIP collection Public Domain license link required')
    properties=item['properties'];asset=item['assets']['image'];href=asset['href'];url=urlsplit(href)
    if url.scheme!='https' or url.hostname!='naipeuwest.blob.core.windows.net' or not url.path.startswith('/naip/') or url.query:
        raise ValueError('Unsigned official NAIP original Azure asset required')
    crs=CRS.from_epsg(properties['proj:epsg']);t=properties['proj:transform']
    if not crs.is_projected or crs.linear_units!='metre':raise ValueError('Original metric CRS required')
    if len(t)!=9 or t[6:]!=[0,0,1] or not all(math.isfinite(v) for v in t):raise ValueError('Finite affine9 grid required')
    dx,sx,x,sy,dy,y=t[:6]
    if not 0<dx<=1 or sx!=0 or sy!=0 or dy!=-dx or properties['gsd']!=dx:
        raise ValueError('Declared GSD and original square north-up grid <=1m required')
    height,width=properties['proj:shape']
    if min(height,width)*dx<1360:raise ValueError('Real original1360m context required')
    if [b.get('common_name') for b in asset['eo:bands'][:3]]!=['red','green','blue']:
        raise ValueError('Original RGB bands required')
    match=re.search(r'_(\d{8})(?:_(\d{8}))?$',item['id'])
    if not match:raise ValueError('Original NAIP acquisition date absent')
    day=datetime.strptime(match.group(1),'%Y%m%d').date().isoformat()
    if properties['datetime'][:10]!=day or str(properties['naip:year'])!=day[:4]:
        raise ValueError('Original filename/STAC acquisition calendar conflict')
    return dict(source_kind='aerial_rgb',source_type='naip_aerial',source_id='naip:'+item['id'],dataset=dataset,
        parent_tile_id=item['id'],href=href,epsg=properties['proj:epsg'],native_gsd_m=dx,delivered_gsd_m=dx,
        explicit_gsd_semantics='original_delivered_orthomosaic_grid_not_camera_capture',
        original_camera_capture_gsd_m=None,original_camera_capture_gsd_provenance='Numerical camera capture GSD unverified',
        shape=[height,width],transform_affine=t[:6],projected_bounds=[x,y+height*dy,x+width*dx,y],
        acquisition_day=day,acquisition_day_provenance='Original NAIP item date token agrees with STAC calendar/year',
        filename_second_date=datetime.strptime(match.group(2),'%Y%m%d').date().isoformat() if match.group(2) else None,
        filename_second_date_semantics='additional_filename_date_not_camera_capture',
        sensor='USDA NAIP aerial camera; exact instrument unverified',
        capture_identity_type='filename_acquisition_day_and_original_orthomosaic_tile',
        stac_datetime=properties['datetime'],stac_datetime_role='capture_calendar_day; nominal time not verified camera exposure',
        stac_license=dataset['collection_license'],license_conflict=dataset['collection_license']!='Public-Domain',
        rgb_indexes=[1,2,3],alpha_index=None,rgb_dtype_declaration='not_in_STAC_verified_at_intake',
        fourth_band_stac_common_name=asset['eo:bands'][3].get('common_name') if len(asset['eo:bands'])>3 else None,
        original_product='orthorectified_aerial_mosaic',radiometry='original_delivered_uint8_RGB_DN_no_new_stretch',
        final_eligible=False)

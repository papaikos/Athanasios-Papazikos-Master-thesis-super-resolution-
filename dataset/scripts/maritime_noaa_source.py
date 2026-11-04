"""Original NOAA aerial tile metadata; publication timestamps are not captures."""
import math
from urllib.parse import urlsplit
from rasterio.crs import CRS


def normalize_item(item, dataset):
    assets = list(item.get('assets', {}).values())
    if len(assets) != 1:
        raise ValueError('Exactly one original tile asset required')
    asset = assets[0]
    crs = CRS.from_epsg(asset['proj:epsg'])
    if not crs.is_projected or crs.linear_units != 'metre':
        raise ValueError('Original projected metre CRS required')
    href = asset['href']
    parsed = urlsplit(href)
    if parsed.scheme != 'https' or parsed.hostname != 'coastalimagery.blob.core.windows.net':
        raise ValueError('Official NOAA original imagery host required')
    if dataset['license'] != 'CC0-1.0' or not dataset['license_evidence_url']:
        raise ValueError('Affirmative dataset-level free-use grant required')
    transform = asset['proj:transform']
    if len(transform) != 6 or not all(math.isfinite(v) for v in transform):
        raise ValueError('Finite GDAL six-value transform required')
    x, dx, sx, y, sy, dy = transform
    if not 0 < dx <= 1 or dx != -dy or sx != 0 or sy != 0:
        raise ValueError('Original north-up square pixels <=1m required')
    bands = asset['eo:bands']
    if [b.get('common_name', b.get('description')) for b in bands[:3]] != ['red', 'green', 'blue']:
        raise ValueError('Original RGB declarations required')
    if any(b['data_type'] != 'uint8' for b in asset['raster:bands'][:3]):
        raise ValueError('Original uint8 RGB required')
    rows, cols = asset['proj:shape']
    if min(rows, cols) * dx < 1360:
        raise ValueError('Single parent tile must support real 1360m context')
    return dict(source_kind='aerial_rgb', source_type='noaa_aerial', source_id=dataset['id'] + ':' + item['id'],
        dataset=dataset, parent_tile_id=item['id'], href=href, epsg=asset['proj:epsg'],
        native_gsd_m=dx, delivered_gsd_m=dx,
        explicit_gsd_semantics='original_delivered_orthomosaic_grid_not_camera_capture',
        original_camera_capture_gsd_m=None,
        original_camera_capture_gsd_provenance='Dataset states source photography was finer; numerical camera GSD unverified',
        shape=[rows, cols], transform_gdal=transform,
        transform_affine=[dx, sx, x, sy, dy, y],
        projected_bounds=[x, y+rows*dy, x+cols*dx, y],
        acquisition_day=dataset['acquisition_day'], sensor=dataset['sensor'],
        capture_identity_type='dataset_acquisition_day_and_original_orthomosaic_tile',
        stac_datetime=item['properties'].get('datetime'),
        stac_datetime_role='publication_or_creation_not_verified_capture',
        stac_license=item['properties'].get('license'),
        license_conflict=item['properties'].get('license') != dataset['license'],
        rgb_indexes=[1, 2, 3], alpha_index=next((i+1 for i,b in enumerate(bands)
            if b.get('description') == 'alpha'), None),
        radiometry='original_delivered_uint8_RGB_DN_no_new_stretch',
        original_product='orthorectified_aerial_mosaic', final_eligible=False)

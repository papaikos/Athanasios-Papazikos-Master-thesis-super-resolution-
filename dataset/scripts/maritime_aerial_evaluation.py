"""Pre-pixel geographic reservation for separate aerial evaluation.

The stable port hash applies across dates and products. A parent near both
roles is held; training parents also avoid a14km evaluation-port reservation
(8km marine AOI, core half-diagonal and a4km separation rounded upward).
Evaluation candidates additionally need the final baseline/capture/geography
checks; this module does not declare an evaluation release ready.
"""
import hashlib
from scripts.maritime_release_select import polygon_wgs84,near
from shapely.geometry import Point

POLICY_VERSION='maritime-aerial-evaluation-v1'
RESERVATION_M=14000


def site_allocation(site_id):
    if not isinstance(site_id,str) or not site_id.strip():
        raise ValueError('Stable nonempty OCHA port ID required')
    score=int.from_bytes(hashlib.sha256((POLICY_VERSION+':'+site_id).encode()).digest(),'big')
    return 'evaluation' if score*10<2**256 else 'train'


def allocate_source(source,evaluation_ports):
    ids=sorted({p['port_id'] for p in source['port_AOI_matches']})
    if not ids:return dict(role='hold',reason='missing_stable_port_identity',port_ids=[])
    roles={site_allocation(identity) for identity in ids}
    if len(roles)!=1:return dict(role='hold',reason='mixed_role_original_parent',port_ids=ids)
    role=roles.pop()
    if role=='train':
        shape=polygon_wgs84(dict(crs='EPSG:'+str(source['epsg']),core_bounds=source['projected_bounds']))
        left,bottom,right,top=shape.bounds
        for port in evaluation_ports:
            if site_allocation(port['port_id'])!='evaluation':raise ValueError('Evaluation-only reservation ports required')
            lon,lat=port['longitude'],port['latitude']
            # One degree safely encloses14km at these NOAA coastal latitudes;
            # longitude is deliberately unfiltered to retain polar safety.
            if bottom-1<=lat<=top+1 and near(shape,Point(lon,lat),RESERVATION_M):
                return dict(role='hold',reason='evaluation_geography_reservation',port_ids=ids,
                            reserved_port_id=port['port_id'])
    return dict(role=role,reason='pre_pixel_stable_port_hash',port_ids=ids,
                policy_version=POLICY_VERSION,evaluation_reservation_m=RESERVATION_M)

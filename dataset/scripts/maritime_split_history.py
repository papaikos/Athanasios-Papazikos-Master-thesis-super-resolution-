"""Conservative metadata footprints of every historical training source."""
import json
from pathlib import Path
from scripts.maritime_cloud_inventory import json_bytes

ROOT=Path('reports/maritime-build-20261006/control')


def coordinates(value):
    if isinstance(value,list) and len(value)>=2 and all(isinstance(v,(int,float)) for v in value[:2]):
        yield value[:2]
    elif isinstance(value,list):
        for part in value:yield from coordinates(part)


def main():
    inventory=json.loads((ROOT/'source-inventory.json').read_text())
    prior=[];seen=set()
    for source in inventory['sources']:
        if source['historical_split']!='train':continue
        identity=(source['site_id'],source['catalog_id'])
        if identity in seen:continue
        seen.add(identity)
        points=list(coordinates(source['geometry']['coordinates']))
        if not points:raise ValueError('Historical training footprint unresolved')
        prior.append(dict(site_id=source['site_id'],catalog_id=source['catalog_id'],crs='EPSG:4326',
            bounds=[min(p[0] for p in points),min(p[1] for p in points),max(p[0] for p in points),max(p[1] for p in points)],
            footprint_basis='Conservative complete historical source geometry bounds; includes every historical training acquisition',
            asset_id=source['asset_id']))
    candidates=json.loads((ROOT/'candidate-inventory-v2.json').read_text())['candidates']
    development=sorted({row['site_id'] for row in candidates if '/pilot-proof-v2/' in row['mother']['uri']})
    (ROOT/'prior-training-footprints.json').write_bytes(json_bytes(prior))
    (ROOT/'development-sites.json').write_bytes(json_bytes(development))
    print(json.dumps(dict(prior_training_acquisitions=len(prior),prior_training_sites=len({r['site_id'] for r in prior}),development_sites=len(development))))


if __name__=='__main__':main()

"""Serialize explicit catalog screening scope; selection never grants semantic acceptance."""
import json
from pathlib import Path

ROOT=Path('reports/maritime-build-20261006')
ADDITIONAL={'740e4c75c6b6a5ff','9ba51cb853426ff6','b48fac59fc4680ec','bdb19bff9235002d'}

def main():
    rows=json.loads((ROOT/'review/catalog-rows.json').read_text())
    chosen=[r for r in rows if r['candidate_kind']=='coast_candidate' or r['group_id'] in ADDITIONAL]
    document=dict(schema=1,status='approved_for_candidate_screening_only',
        asset_ids=sorted({a for r in chosen for a in r['asset_ids']}),
        review_metadata=dict(scope='All acquisitions at plausible coastal groups may undergo native mask and semantic inspection; only representative acquisition viewed at overview scale.',
            vessel_presence='unconfirmed',semantic_accepted_bytes=0,
            groups=[dict(group_id=r['group_id'],asset_ids=r['asset_ids'],bbox=r['bbox'],
                country=r.get('screening',{}).get('countries',[]),
                representative_asset_id=r['representative_asset_id'],preview=r.get('preview'),
                reason='Visible coastal context in representative' if r['candidate_kind']=='coast_candidate' else 'Independent reviewer marked plausible coastal or tidal context requiring native and geographic verification') for r in chosen]))
    (ROOT/'control/reuse-approved-assets.json').write_text(json.dumps(document,indent=2))
    sites={r['group_id']:dict(country=(r.get('screening',{}).get('countries',[])[0]
        if len(r.get('screening',{}).get('countries',[]))==1 else None),
        country_candidates=r.get('screening',{}).get('countries',[]),geographic_group_id=r['group_id']) for r in rows}
    (ROOT/'control/site-metadata.json').write_text(json.dumps(sites,indent=2))
    print(json.dumps(dict(screening_groups=len(chosen),assets=len(document['asset_ids']))))

if __name__=='__main__':main()

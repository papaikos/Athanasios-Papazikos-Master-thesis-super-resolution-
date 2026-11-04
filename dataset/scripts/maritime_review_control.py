"""Persist explicit visual decisions and cloud screening evidence; no pixel acceptance inference."""
import hashlib
import json
from pathlib import Path

from scripts.maritime_build_control import BUCKET, client_from_gcloud
from scripts.maritime_cloud_inventory import create_only, json_bytes

ROOT = Path('reports/maritime-build-20261006')
PREFIX = 'skysat/thesis-datasets/v1/control/runs/maritime-20261006'

DECISIONS = [
 ('01ea9de4168e057efe60f4ff','accept','marina',True,True,'Dense fishing boats in coastal cove; mild tonal seam and softness.'),
 ('054433deadcbbfd01f6b95cd','accept','commercial_port',True,True,'Industrial jetty and cargo ships with small accompanying craft; mild haze.'),
 ('09a65178e47fb596401c896a','reject','coast',True,True,'Displaced green RGB patch and geometric seam near coastal rocks.'),
 ('1406969983a99fe2a0f820b6','accept','commercial_port',True,False,'Moored ships and quays; small craft not confirmed.'),
 ('16d5a5cf628e9dfbc966f2e4','accept','commercial_port',True,True,'Bulk carriers and workboats; mild moving-wake fringing.'),
 ('1dcda782870dae618be24c05','reject','open_water_vessel',True,True,'Main moving vessels have separated colored or duplicate wakes and mosaic boundaries.'),
 ('20c4c4dbe2c93cf63d48280f','accept','marina',True,True,'Dense boats in protected basins beside commercial channel.'),
 ('2356539787c69ac3c076d331','accept','coast',True,True,'Breakwater, reef and piers with sheltered small boats; moderate softness.'),
 ('23b36a43ca40af056901a6fe','accept','coast',False,False,'Engineered shoreline and breakwater; possible tiny craft remain unconfirmed.'),
 ('246431b4cd8fcde8ba46c64f','accept','marina',True,True,'Dense moored and underway boats; mild bright-hull clipping and edge fringing.'),
 ('2489ee02d586168976f0aa08','accept','commercial_port',True,True,'Container ship and small craft at industrial waterfront; mild water tonal seam.'),
 ('2d560c02c2af66908d1fe793','reject','context_uncertain',True,True,'River channel, bridge and recreation roads; coastal or harbor context not established in core.'),
]

def main():
    client = client_from_gcloud()
    bucket = client.bucket(BUCKET)
    def publish(name, document):
        payload = json_bytes(document)
        local = ROOT / 'control' / name
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(payload)
        return dict(uri=f'gs://{BUCKET}/{PREFIX}/review/{name}',
                    **create_only(bucket.blob(f'{PREFIX}/review/{name}'), payload))

    summaries = []
    for part in range(4):
        blob = bucket.blob(f'{PREFIX}/catalog-screen-v2/part-{part}/summary.json')
        blob.reload()
        payload = blob.download_as_bytes(if_generation_match=int(blob.generation))
        document = json.loads(payload)
        summaries.append(dict(partition=part, document=document, source=dict(
            uri=f'gs://{BUCKET}/{blob.name}', generation=str(blob.generation),
            bytes=blob.size, crc32c=blob.crc32c, sha256=hashlib.sha256(payload).hexdigest())))
    publish('catalog-completed-v2.json', dict(status='screening_only', partitions=summaries))
    samples = {r['sample_id']: r for r in json.loads((ROOT/'control/pilot-samples-v1.json').read_text())}
    decisions = []
    for old_id, decision, category, vessel, small, reason in DECISIONS:
        sample = samples[old_id]
        preview = sample['native_review_preview']
        path = ROOT/'review/native'/f'{old_id}.png'
        if hashlib.sha256(path.read_bytes()).hexdigest() != preview['sha256']:
            raise ValueError('Reviewed native image hash differs from pinned sample')
        transform = sample['mother_core_transform']
        identity = [sample['source']['asset_id'], sample['crs'], round(transform[2],6), round(transform[5],6),1600,1600]
        new_id = hashlib.sha256(json_bytes(identity)).hexdigest()[:24]
        decisions.append(dict(historical_sample_id=old_id, world_sample_id=new_id,
            core_identity=identity, native_preview=preview, context_sha256=sample['context_sha256'],
            decision=decision, category=category, vessel_presence='confirmed' if vessel else 'unconfirmed',
            small_craft_presence='confirmed' if small else 'unconfirmed', hard_negative_confirmed=False,
            reviewer='root_native_visual_review', reason=reason,
            usable_for_final_selection=False, split='development',
            scope='pilot_review; final geographic split and quantitative quality gates pending'))
    publish('pilot-native-decisions-v1.json', dict(schema=1, decisions=decisions,
        accepted=sum(d['decision']=='accept' for d in decisions), rejected=sum(d['decision']=='reject' for d in decisions)))
    publish('quality-policy-v1.json', dict(schema=1,
        principal_target_rgb_misregistration_or_doubling='reject',
        displaced_source_artifact='reject',
        minor_tonal_seam_away_from_principal_targets='flag; accept only if structure remains usable',
        unknown_vessel_presence='unconfirmed; never count as confirmed or hard negative',
        semantic_review='native 1600-pixel RGB core plus geographic context',
        coarse_preview='source screening only; cannot confirm vessels or final eligibility',
        independent_audit='at least 10% of selected cores; stratify site, category and split',
        final_acceptance='requires numerical thresholds and independent audit in original plan'))
    print(json.dumps(dict(catalog_partitions=len(summaries), native_decisions=len(decisions))))

if __name__ == '__main__':
    main()

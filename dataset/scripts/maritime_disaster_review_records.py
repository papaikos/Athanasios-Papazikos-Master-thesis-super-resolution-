"""Join returned native reviewer decisions to immutable disaster preview pins."""
import json
from pathlib import Path
from scripts.maritime_cloud_inventory import json_bytes
ROOT=Path('reports/maritime-build-20261006')
DECISIONS=[
 (75,'accept','coastline','confirmed','confirmed',False,'Slender outrigger near upper-left coastal water; mild haze.'),
 (78,'accept','coastline','absent','absent',False,'Forested cape, breaking surf and pier; no identifiable hull.'),
 (79,'accept','coastline','absent','absent',False,'Curving beach, lagoon margin and continuous surf.'),
 (81,'reject','coastline','confirmed','unconfirmed',False,'Cloud obscures coastal target although rust-colored hull is visible.'),
 (85,'reject','coastline','absent','absent',False,'Inland forest/industrial development dominates; tiny coastal corner.'),
 (90,'accept','commercial_fishing_port','confirmed','unconfirmed',False,'Large ship along loading jetty near top edge; industrial shoreline and surf.'),
 (91,'reject','open_water','absent','absent',False,'Dark low-contrast water and chromatic speckles; no deliberate negative target.'),
 (92,'accept','commercial_fishing_port','confirmed','confirmed',False,'Harbor entrance, docked workboats and yachts in lower-left marina.'),
 (93,'accept','commercial_fishing_port','confirmed','unconfirmed',False,'Large vessels along port quays; peripheral cloud away from main ships.'),
 (94,'accept','commercial_fishing_port','confirmed','confirmed',False,'Tank farm/wharves with edge-clipped large ships and separate small hull lower-left.'),
 (95,'reject','coastline','unconfirmed','unconfirmed',False,'Haze/cloud reduces industrial peninsula and jetty clarity.'),
 (113,'accept','coastline','unconfirmed','unconfirmed',False,'Clear long jetty, breakwater and strong wave texture; peripheral cloud.'),
 (114,'accept','open_water','absent','absent',True,'Continuous wind-wave and whitecap field; no recognizable hulls/structures.'),
 (119,'accept','marina_small_craft','confirmed','confirmed',False,'Marina at southeast corner with clear finger piers and multiple yacht hulls; mild softness.'),
 (120,'accept','coastline','confirmed','confirmed',False,'Rocky developed coast and protected beaches; boat hulls in partial right-edge marina.'),
 (121,'accept','marina_small_craft','confirmed','confirmed',False,'Distinct yachts in left marina, breakwaters and protected beaches.'),
 (122,'accept','marina_small_craft','confirmed','confirmed',False,'Small protected basins along bottom shoreline contain discernible boats.'),
 (123,'accept','coastline','unconfirmed','unconfirmed',False,'Groynes, surf and coastal road clear; bright specks unresolved.'),
 (129,'accept','marina_small_craft','confirmed','confirmed',False,'Boats line protected shoreline basin lower-right; offshore points not counted.'),
 (130,'accept','coastline','unconfirmed','unconfirmed',False,'Developed waterfront and long narrow offshore structure; no confirmed hull.'),
 (136,'accept','coastline','absent','absent',False,'Industrial shoreline, projecting breakwater and surf without identifiable vessels.'),
 (137,'accept','commercial_fishing_port','confirmed','confirmed',False,'Container terminal/cranes and harbor breakwater; small workboat just above quay.'),
 (142,'accept','marina_small_craft','confirmed','confirmed',False,'Protected marina basins on right contain clearly separated yachts.'),
 (145,'accept','marina_small_craft','confirmed','confirmed',False,'Sheltered coastal basin contains moored hulls; additional boats along beach.'),
 (146,'accept','coastline','absent','absent',False,'Seawall, projecting groyne and breaking surf along developed coast.'),
]

AUDITS={75:('disagree','coastline','unconfirmed','unconfirmed',False,'Possible outrigger near (443,145) insufficiently distinct from floating square/linear structures.'),
 78:('agree','coastline','absent','absent',False,'Forested headland, surf and shore-connected pier/platform; no identifiable hulls.'),
 92:('agree','commercial_fishing_port','confirmed','confirmed',False,'Working harbor boats and distinct small hulls beside lower-edge finger piers.'),
 93:('agree','commercial_fishing_port','confirmed','unconfirmed',False,'Recognizable quay-side ships; smaller objects lack sufficient detail.'),
 113:('agree','coastline','unconfirmed','unconfirmed',False,'Clear jetty/breakwater and breaking waves; peripheral cloud avoids principal structures.'),
 114:('agree','open_water','absent','absent',True,'Continuous wave texture/whitecaps without discrete hulls or vessel wakes.'),
 119:('agree','marina_small_craft','confirmed','confirmed',False,'Separated yacht hulls beside southeast finger piers.')}


def main():
    candidates=json.loads((ROOT/'review/all/disaster-rows.json').read_text())
    reviews=json.loads((ROOT/'control/native-reviews-v2.json').read_text())
    added=[]
    for index,decision,category,vessel,small,negative,reason in DECISIONS:
        candidate=candidates[index]
        row=dict(candidate_id=candidate['candidate_id'],canonical_core_id=candidate['canonical_core_id'],
            decision=decision,reviewer='source_feasibility',native_scale_reviewed=True,primary_category=category,
            vessel_presence=vessel,small_craft_presence=small,hard_negative=negative,
            quality_status='pass' if decision=='accept' else 'reject',duplicate_status='clear',support_status='verified',
            preview=candidate['native_preview'],reason=reason,quality_flags=['low_contrast'],
            review_version='native-disaster-secondary-20261006-v1')
        added.append(row)
    (ROOT/'control/disaster-native-reviews-v1.json').write_bytes(json_bytes(added))
    (ROOT/'control/native-reviews-v3.json').write_bytes(json_bytes(reviews+added))
    adjudicated=dict(next(row for row in added if row['candidate_id']==candidates[75]['candidate_id']))
    adjudicated.update(reviewer='root',vessel_presence='unconfirmed',small_craft_presence='unconfirmed',
        reason='Root inspected original native PNG and adopts conservative independent presence labels; feature among floating structures is not confidently a hull.',
        review_version='native-disaster-adjudication-75-v1')
    (ROOT/'control/native-reviews-v4.json').write_bytes(json_bytes(reviews+added+[adjudicated]))
    audits=json.loads((ROOT/'control/native-audits-v2.json').read_text())
    for index,(decision,category,vessel,small,negative,reason) in AUDITS.items():
        candidate=candidates[index]
        audits.append(dict(candidate_id=candidate['candidate_id'],independent=True,reviewer='planet_intake',
            native_scale_reviewed=True,decision=decision,primary_category=category,vessel_presence=vessel,
            small_craft_presence=small,hard_negative=negative,preview=candidate['native_preview'],evidence=reason,
            review_version='native-disaster-independent-20261006-v1'))
    resolution=dict(next(row for row in audits if row['candidate_id']==candidates[75]['candidate_id']))
    resolution.update(decision='agree',resolution_reference='native-disaster-adjudication-75-v1',
        evidence='Independent native labels unconfirmed/unconfirmed adopted after root native inspection; earlier disagreement retained.',
        review_version='native-disaster-independent-resolution-75-v1')
    audits.append(resolution)
    (ROOT/'control/native-audits-v3.json').write_bytes(json_bytes(audits))
    print(json.dumps(dict(reviews=len(reviews+added),new_reviews=len(added))))


if __name__=='__main__':main()

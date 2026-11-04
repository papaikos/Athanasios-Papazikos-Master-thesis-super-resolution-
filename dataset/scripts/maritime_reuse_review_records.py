"""Append root native-scale decisions to pinned review records."""
import json
from pathlib import Path
from scripts.maritime_cloud_inventory import json_bytes

ROOT = Path('reports/maritime-build-20261006')
DECISIONS = [
    (12, 'accept', 'commercial_fishing_port', 'confirmed', 'confirmed', 'Working quays, large ships and distinct yacht hulls beside lower-right finger piers; mild softness.'),
    (13, 'accept', 'commercial_fishing_port', 'confirmed', 'confirmed', 'Tank farms and industrial quays with large vessels and distinct smaller docked workboats at lower left.'),
    (14, 'accept', 'coastline', 'confirmed', 'confirmed', 'Long seawall and surf dominate; harbor corner contains large vessels and a smaller working hull.'),
    (18, 'reject', 'coastline', 'confirmed', 'confirmed', 'Displaced RGB and bright green shoreline artifact near lower-left promontory; severe water mosaic.'),
    (19, 'reject', 'open_water', 'confirmed', 'confirmed', 'Principal moving boat and wake are doubled/displaced; severe stitched-water artifact.'),
    (20, 'accept', 'coastline', 'confirmed', 'confirmed', 'Continuous developed rocky/beach coastline with several distinct small moored hulls; no severe principal-target seam.'),
    (22, 'accept', 'marina_small_craft', 'confirmed', 'confirmed', 'Multiple sheltered finger-pier marinas and distinct yacht hulls across the inlet; mild image softness.'),
    (23, 'accept', 'commercial_fishing_port', 'confirmed', 'confirmed', 'Industrial channel, large loading-jetty vessel and container ship; distinct small tug beside lower-left ship.'),
]

def main():
    candidates = json.loads((ROOT/'review/all/reuse-rows.json').read_text())
    records = json.loads((ROOT/'control/native-reviews-v5.json').read_text())
    added = []
    for index, decision, category, vessel, small, reason in DECISIONS:
        candidate = candidates[index]
        added.append(dict(candidate_id=candidate['candidate_id'], canonical_core_id=candidate['canonical_core_id'],
            decision=decision, reviewer='root', native_scale_reviewed=True, primary_category=category,
            vessel_presence=vessel, small_craft_presence=small, hard_negative=False,
            quality_status='pass' if decision=='accept' else 'reject', support_status='verified',
            duplicate_status='clear', preview=candidate['native_preview'], reason=reason,
            quality_flags=['mild_softness'] if decision=='accept' else ['principal_target_artifact'],
            review_version='native-reuse-root-20261006-v1'))
    (ROOT/'control/reuse-native-reviews-v1.json').write_bytes(json_bytes(added))
    (ROOT/'control/native-reviews-v6.json').write_bytes(json_bytes(records+added))
    print(json.dumps(dict(native_decisions=len(added))))

if __name__ == '__main__': main()

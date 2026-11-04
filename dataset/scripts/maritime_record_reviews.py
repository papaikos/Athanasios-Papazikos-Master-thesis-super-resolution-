"""Versioned review records backed by pinned native previews; root-owned decisions."""
import json
from pathlib import Path
from scripts.maritime_cloud_inventory import json_bytes

ROOT=Path('reports/maritime-build-20261006')

PLANET=[
 (0,'accept','open_water','absent','absent',True,'Fully ship-free open water with resolved wave texture; deliberate negative, limit redundant water.'),
 (1,'accept','open_water','confirmed','unconfirmed',False,'Long recognizable ship at anchor with shadow; no severe target artifact or obscuring cloud.'),
 (2,'accept','open_water','confirmed','unconfirmed',False,'Two recognizable vessels including a broad barge; small-craft size not confidently established.'),
 (3,'accept','open_water','confirmed','unconfirmed',False,'Recognizable ship near upper edge; inspect edge clipping flag, real source halo available.'),
 (13,'reject','commercial_fishing_port','confirmed','confirmed',False,'Cloud and shadow obscure substantial industrial quays and target waterfront; water stitching boundaries also visible.'),
 (14,'accept','commercial_fishing_port','confirmed','confirmed',False,'Container terminal, breakwater and many individually recognizable small moored craft; stationary vessels clear, water stitching flagged.'),
 (15,'accept','commercial_fishing_port','confirmed','confirmed',False,'Industrial harbor with clear working boat in basin and ships along quays; cloud away from principal visible vessels, water seams flagged.'),
]

SECONDARY_ACCEPTED={
 70:('open_water','confirmed','unconfirmed','Isolated powered vessel and wake are clear over reef; mild glint.'),
 72:('open_water','confirmed','unconfirmed','Elongated vessel at left and partial upper-edge vessel; peripheral cloud avoids principal vessels.'),
 73:('coastline','confirmed','confirmed','Clear tidal reef and sandbars with several recognizable hulls, including small craft.'),
 77:('coastline','confirmed','unconfirmed','Reconciled category by predominant coastline; cropped basin at top center contains long rust hull and moored white/brown hull near x610-672,y55-173. Small-craft size remains unconfirmed.'),
 85:('coastline','confirmed','confirmed','Coastal piers and recognizable small hulls; haze and low contrast flagged.'),
 87:('marina_small_craft','confirmed','confirmed','Distinct marina fingers with many individually recognizable boats; haze and low contrast flagged.'),
}
SECONDARY_REJECTED={
 60:'Inland dunes and mosaic boundaries.',61:'Inland dunes with stitching.',62:'Inland dunes and vegetation.',
 63:'Inland dunes.',64:'Inland dunes with faint seam.',65:'Cloud and shadow over water.',
 66:'Cloud and shadow over water.',67:'Cloud obscures reef and water.',68:'Cloud and shadow over water.',
 69:'Cloud and vertical seam compromise soft vessel targets.',71:'Cloud shadow and horizontal stitching.',
 74:'Cloud obscures outer jetty and port approach.',75:'Cloud and rectangular stitching.',
 76:'Cloud obscures island shore.',78:'Magenta shoreline smear and haze.',79:'Cloud and vertical seams.',
 80:'Cloud and shadow obscure reef.',81:'Cloud and vertical seam.',82:'Cloud shadow and ambiguous linear feature.',
 83:'Cloud over water.',84:'Cloud and chromatic band.',86:'Inland housing, hills and recreation under haze.',
 88:'Severe stepped vertical mosaic boundaries and haze.',89:'Vertical seam, bright linear artifact and haze.',
}

SF_ACCEPTED={90:('coastline','unconfirmed','unconfirmed','Rocky developed headland and coastal water; tiny flecks unresolved.'),
 93:('open_water','confirmed','confirmed','Distinct small boat and trailing wake in upper-right water.'),
 97:('open_water','confirmed','confirmed','Small powered boat left of center, recognizable hull and diverging wake.'),
 100:('open_water','confirmed','confirmed','Distinct small powered vessel and short wake near lower-left edge.'),
 103:('coastline','unconfirmed','unconfirmed','Continuous beach, surf line, coastal lagoon and developed shore; offshore specks unresolved.')}
SF_REJECTED={91:'Stepped radiometric seam crosses coastal headland and water.',92:'Central seam and bright linear source artifacts compromise vessel core.',
 94:'Prominent seam and long bright artifact cross marina water.',95:'Uniform water with unresolved feature and no deliberate hard-negative target.',
 96:'Conspicuous seam divides vessel and wave field.',98:'Strong stepped and horizontal source boundaries.',99:'Central seam and conspicuous colored wake fringes.',
 101:'Strong mosaic boundaries cross partial marina.',102:'Seam crosses roads, vegetation and coastal infrastructure.',104:'Full-height seam crosses marina and urban area.',
 105:'Inland residential hills and sports facilities.',106:'Inland residential blocks and sports grounds.',107:'Entirely urban land and roads.',
 108:'Residential grid, freeway and sports field.',109:'Inland urban scene with strong source seam.'}

def main():
    primary=json.loads((ROOT/'control/pilot-native-decisions-v1.json').read_text())['decisions']
    categories={'marina':'marina_small_craft','commercial_port':'commercial_fishing_port','coast':'coastline','open_water_vessel':'open_water'}
    reviews=[]
    audits=[]
    for row in primary:
        category=categories.get(row['category'],'coastline')
        reviews.append(dict(candidate_id=row['world_sample_id'],historical_candidate_id=row['historical_sample_id'],
            decision=row['decision'],reviewer='root',native_scale_reviewed=True,
            primary_category=category,vessel_presence=row['vessel_presence'],small_craft_presence=row['small_craft_presence'],
            hard_negative=False,quality_status='pass' if row['decision']=='accept' else 'reject',
            duplicate_status='clear',support_status='verified',preview=row['native_preview'],reason=row['reason'],
            split_use='development',review_version='native-pilot-20261006-v1'))
        audits.append(dict(candidate_id=row['world_sample_id'],independent=True,reviewer='source_feasibility',
            native_scale_reviewed=True,decision='agree',primary_category=category,vessel_presence=row['vessel_presence'],
            small_craft_presence=row['small_craft_presence'],hard_negative=False,
            preview=row['native_preview'],evidence='Independent native reviewer returned 12 individual decisions; root and secondary agreed on nine accepts and three rejects.',
            review_version='native-pilot-secondary-20261006-v1'))
    candidates=json.loads((ROOT/'control/planet-candidate-inventory-v1.json').read_text())['candidates']
    for index,decision,category,vessel,small,negative,reason in PLANET:
        candidate=candidates[index]
        reviews.append(dict(candidate_id=candidate['candidate_id'],canonical_core_id=candidate['canonical_core_id'],
            decision=decision,reviewer='root',native_scale_reviewed=True,primary_category=category,
            vessel_presence=vessel,small_craft_presence=small,hard_negative=negative,quality_status='pass' if decision=='accept' else 'reject',
            duplicate_status='clear',support_status='verified',preview=candidate['native_preview'],reason=reason,
            review_version='native-planet-20261006-v1'))
    for index in range(60,90):
        candidate=candidates[index]
        accepted=index in SECONDARY_ACCEPTED
        category,vessel,small,reason=SECONDARY_ACCEPTED.get(index,
            (None,'unconfirmed','unconfirmed',SECONDARY_REJECTED.get(index,'Unresolved independent category and vessel disagreement; held out pending review.')))
        reviews.append(dict(candidate_id=candidate['candidate_id'],canonical_core_id=candidate['canonical_core_id'],
            decision='accept' if accepted else 'reject',reviewer='source_feasibility',native_scale_reviewed=True,
            primary_category=category,vessel_presence=vessel,small_craft_presence=small,hard_negative=False,
            quality_status='pass' if accepted else 'reject',duplicate_status='clear',support_status='verified',
            preview=candidate['native_preview'],reason=reason,review_version='native-planet-secondary-20261006-v2'))
        if accepted and index in (85,87):
            reviews[-1]['repeat_observation_reason']='Only earlier SF acquisition strip is eligible; later overlapping strip excluded'
        if accepted or index==77:
            audits.append(dict(candidate_id=candidate['candidate_id'],independent=True,reviewer='root',
                native_scale_reviewed=True,decision='agree' if accepted else 'disagree',
                primary_category=category if accepted else 'coastline',vessel_presence=vessel,small_craft_presence=small,
                hard_negative=False,preview=candidate['native_preview'],
                evidence=reason if accepted else 'Exact native PNG shows coastal urban strip and tidal shallows; secondary commercial-harbor and vessel labels are not established. Excluded pending reconciliation.',
                review_version='native-planet-root-audit-20261006-v2'))
    for index in range(90,110):
        candidate=candidates[index];accepted=index in SF_ACCEPTED
        category,vessel,small,reason=SF_ACCEPTED.get(index,(None,'unconfirmed','unconfirmed',SF_REJECTED.get(index)))
        review=dict(candidate_id=candidate['candidate_id'],canonical_core_id=candidate['canonical_core_id'],
            decision='accept' if accepted else 'reject',reviewer='source_feasibility',native_scale_reviewed=True,
            primary_category=category,vessel_presence=vessel,small_craft_presence=small,hard_negative=False,
            quality_status='pass' if accepted else 'reject',duplicate_status='clear',support_status='verified',
            preview=candidate['native_preview'],reason=reason,quality_flags=['haze','low_contrast'],
            review_version='native-sf-secondary-20261006-v3')
        if accepted:review['repeat_observation_reason']='Only earlier SF acquisition strip is eligible; later overlapping strip excluded'
        reviews.append(review)
        if index in (90,93):
            audits.append(dict(candidate_id=candidate['candidate_id'],independent=True,reviewer='root',
                native_scale_reviewed=True,decision='agree',primary_category=category,vessel_presence=vessel,
                small_craft_presence=small,hard_negative=False,preview=candidate['native_preview'],
                evidence=reason+' Native hull/wake or shoreline inspected at original 1600 scale; haze/low contrast flagged.',
                review_version='native-sf-root-audit-20261006-v3'))
    for index,reason in [(0,'Continuous resolved wave texture without identifiable hulls or boat wakes.'),
        (14,'Container terminal and breakwater; small hulls around (625,955), (628,1045) and lower basin remain clear despite water seams.'),
        (15,'Working vessel near (495,335), small craft near (1335,240), quay-side ships clear despite peripheral haze and water seams.')]:
        candidate=candidates[index]
        primary_row=next(row for row in reviews if row['candidate_id']==candidate['candidate_id'])
        audits.append(dict(candidate_id=candidate['candidate_id'],independent=True,reviewer='planet_intake',
            native_scale_reviewed=True,decision='agree',
            **{key:primary_row[key] for key in ('primary_category','vessel_presence','small_craft_presence','hard_negative')},
            preview=candidate['native_preview'],evidence=reason,
            review_version='native-planet-intake-audit-20261006-v2'))
    (ROOT/'control/native-reviews-v2.json').write_bytes(json_bytes(reviews))
    (ROOT/'control/native-audits-v2.json').write_bytes(json_bytes(audits))
    print(json.dumps(dict(reviews=len(reviews),independent_audits=len(audits))))

if __name__=='__main__':main()

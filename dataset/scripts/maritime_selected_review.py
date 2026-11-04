"""Render exact selected canonical HR on GCE, without granting source replay or QA."""
import argparse
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from scripts.maritime_canonical_screening import canonical_arrays
from scripts.maritime_cloud_inventory import digest_bytes, json_bytes, require_cloud_vm, verify_output


def bound_review_rgb(raw,row):
    _,rgb,hashes=canonical_arrays(raw)
    if any(row.get(key)!=value for key,value in hashes.items()):
        raise ValueError('Selected HR hash mismatch: '+row['sample_id'])
    return rgb


def png(image):
    stream=io.BytesIO();image.save(stream,format='PNG');return stream.getvalue()


def run(snapshot_pin,output,scratch):
    require_cloud_vm()
    from google.cloud import storage
    from scripts.maritime_ultimate_release import CloudStore
    from src.maritime_earthview_source import validate_source as validate_earth
    from src.maritime_naip_source import validate_source as validate_naip
    client=storage.Client();snapshot=json.loads(verify_output(client,snapshot_pin))
    if snapshot.get('schema')!='marine-selected-canonical-review-snapshot-v1':
        raise ValueError('Frozen selected canonical review snapshot required')
    rows=snapshot['candidates'];ids=[r['sample_id'] for r in rows]
    if not ids or len(set(ids))!=len(ids):raise ValueError('Distinct selected review identities required')
    store=CloudStore(client,output,scratch);records=[];contacts=[];tiles=[]
    def close_page():
        page=len(contacts);sheet=Image.new('RGB',(1024,1626),'white')
        for i,tile in enumerate(tiles):sheet.paste(tile,(i%2*512,i//2*542))
        contacts.append(store.publish('contacts/page-'+format(page,'03d')+'.png',png(sheet)))
        tiles.clear()
    for row in rows:
        source=row['source'];family=row['qualification']['source_family'];identity=row['sample_id']
        if family=='earthview':
            candidate_pin=row['candidate_pin'];original=json.loads(verify_output(client,candidate_pin))
            if original!=source:raise ValueError('Exact Earth candidate JSON required')
            validate_earth(source)
        elif family=='naip_aerial':
            candidate_pin=source['candidate_record'];original=json.loads(verify_output(client,candidate_pin))
            if original!={k:v for k,v in source.items() if k!='candidate_record'}:
                raise ValueError('Exact NAIP candidate JSON required')
            validate_naip(source)
        else:raise ValueError('Only newly screened Earth/NAIP canonical candidates supported')
        if candidate_pin['sha256']!=row['candidate_sha256']:raise ValueError('Selected candidate hash mismatch')
        rgb=bound_review_rgb(verify_output(client,source['hr']),row)
        image=Image.fromarray(np.moveaxis(rgb,0,-1))
        image_pin=store.publish('images/'+identity+'-hr.png',png(image))
        index=len(records);tile=Image.new('RGB',(512,542),'white');tile.paste(image,(0,30))
        port=source.get('nearest_port',source.get('port',{}))
        ImageDraw.Draw(tile).text((2,2),str(index)+' '+family+' '+str(port.get('name',''))[:36],fill='black')
        tiles.append(tile)
        records.append(dict(sample_id=identity,source=family,candidate_pin=candidate_pin,
            candidate_sha256=row['candidate_sha256'],hr_pin=source['hr'],
            hr_float32_sha256=row['hr_float32_sha256'],hr_uint8_chw_sha256=row['hr_uint8_chw_sha256'],
            actual512_png_pin=image_pin,page=index//6,row=index%6//2+1,column=index%2+1,
            canonical_hr_hashes_verified=True,original_assembly_replayed=False,
            final_eligible=False,dataset_ready=False))
        if len(tiles)==6:close_page()
        print(json.dumps(dict(completed=len(records),count=len(rows))),flush=True)
    if tiles:close_page()
    manifest=dict(schema='marine-selected-canonical-actual512-review-v1',snapshot_pin=snapshot_pin,
        selection_pin=snapshot['selection_pin'],audit_plan_pin=snapshot['audit_plan_pin'],
        code_archive_pin=snapshot['code_archive_pin'],count=len(records),records=records,contacts=contacts,
        contact_display='Exact uint8 RGB512, 1:1, six/page; no enhancement or resizing',
        original_assembly_replayed=False,original_full_replay_required_at_release=True,
        quality_accepted_count=0,actual_image_review_pending=True,dataset_ready=False)
    print(json.dumps(store.publish('manifest.json',json_bytes(manifest))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-pin',required=True)
    parser.add_argument('--output-prefix',required=True)
    parser.add_argument('--scratch',required=True)
    args=parser.parse_args()
    run(json.loads(Path(args.snapshot_pin).read_bytes()),args.output_prefix,args.scratch)

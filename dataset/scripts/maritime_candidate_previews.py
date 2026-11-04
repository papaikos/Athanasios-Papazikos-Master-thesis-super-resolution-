"""SDK download of bounded, pinned review PNGs only; never source rasters."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from scripts.maritime_build_control import client_from_gcloud
from scripts.maritime_cloud_inventory import parse_gs

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',required=True)
    parser.add_argument('--destination',required=True)
    parser.add_argument('--limit',type=int,default=1000)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error('--workers must be between 1 and 16')
    rows=json.loads(Path(args.inventory).read_text())['candidates'][:args.limit]
    root=Path(args.destination);root.mkdir(parents=True,exist_ok=True)
    client=client_from_gcloud()
    def download(row):
        ref=row['native_preview'];path=root/(row['candidate_id']+'.png')
        if not ref['uri'].endswith('.png') or not 0<ref['bytes']<=15_000_000:
            raise ValueError('Bounded native PNG required')
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==ref['sha256']:
            return dict(candidate_id=row['candidate_id'],path=str(path),cached=True)
        bucket,name=parse_gs(ref['uri'])
        payload=client.bucket(bucket).blob(name,generation=int(ref['generation'])).download_as_bytes(
            if_generation_match=int(ref['generation']),checksum='crc32c')
        if len(payload)!=ref['bytes'] or hashlib.sha256(payload).hexdigest()!=ref['sha256']:
            raise ValueError('Native preview readback differs')
        path.write_bytes(payload)
        return dict(candidate_id=row['candidate_id'],path=str(path),cached=False)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        receipts=list(pool.map(download,rows))
    (root/'preview-receipts.json').write_text(json.dumps(receipts,indent=2))
    print(json.dumps(dict(previews=len(receipts))))

if __name__=='__main__':main()

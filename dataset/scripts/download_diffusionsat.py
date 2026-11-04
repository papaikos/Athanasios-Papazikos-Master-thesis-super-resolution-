"""Download exact official inference members with ZIP CRC and SHA256 receipts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import time
import urllib.request
import zlib

URL = 'https://zenodo.org/api/records/13756246/files/controlnet_sd21_md7norm_fmow_condres256.zip/content'
PREFIX = 'controlnet_sd21_md7norm_fmow_condres256/'
COMMIT = '68259b074dfa567e6cc4667799dab0e37477fcf2'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def byte_range(start, end):
    request = urllib.request.Request(URL, headers={'Range': f'bytes={start}-{end}'})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                assert response.status == 206, 'Server did not honor bounded range'
                assert response.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
                data = response.read()
            assert len(data) == end - start + 1
            return data
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def selected(row):
    relative = row['name'].removeprefix(PREFIX)
    return bool(row['size']) and (relative == 'model_index.json' or relative.startswith(
        ('unet/', 'text_encoder/', 'vae/', 'tokenizer/', 'scheduler/', 'feature_extractor/',
         'checkpoint-50000/controlnet/')))


def extract(row, output):
    relative = row['name'].removeprefix(PREFIX)
    target = output / relative
    receipt_path = target.with_name(target.name + '.receipt.json')
    if target.exists() and receipt_path.exists():
        saved = json.loads(receipt_path.read_text())
        if target.stat().st_size == row['size'] and sha(target) == saved['sha256']:
            return saved
    target.parent.mkdir(parents=True, exist_ok=True)
    header = byte_range(row['header_offset'], row['header_offset'] + 29)
    assert header[:4] == b'PK\x03\x04'
    fields = struct.unpack('<4s5H3I2H', header)
    method = fields[3]
    assert method == 8, method
    start = row['header_offset'] + 30 + fields[-2] + fields[-1]
    temporary = target.with_name(target.name + '.partial')
    decompress = zlib.decompressobj(-15)
    crc, count = 0, 0
    with temporary.open('wb') as sink:
        for position in range(start, start + row['compressed'], 16 * 1024 * 1024):
            block = byte_range(position, min(position + 16 * 1024 * 1024, start + row['compressed']) - 1)
            decoded = decompress.decompress(block)
            sink.write(decoded)
            crc = zlib.crc32(decoded, crc)
            count += len(decoded)
        decoded = decompress.flush()
        sink.write(decoded)
        crc = zlib.crc32(decoded, crc)
        count += len(decoded)
    assert decompress.eof and count == row['size'] and crc == row['crc'], row['name']
    temporary.replace(target)
    receipt = {**row, 'path': relative, 'sha256': sha(target), 'crc_verified': True,
               'downloaded_utc': datetime.now(timezone.utc).isoformat(), 'url': URL}
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(json.dumps({'verified_member': relative, 'bytes': count}), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=Path, default=Path('reports/pretrained-comparison-20260929/diffusionsat-research/zip-list.json'))
    parser.add_argument('--output', type=Path, default=Path('checkpoints/pretrained-diffusionsat'))
    args = parser.parse_args()
    rows = [row for row in json.loads(args.index.read_text()) if selected(row)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        receipts = list(pool.map(lambda row: extract(row, args.output), rows))
    report = {'status': 'complete', 'source_commit': COMMIT, 'url': URL, 'members': receipts,
              'total_uncompressed_bytes': sum(r['size'] for r in receipts),
              'whole_archive_md5_verified': False, 'verification': 'ZIP member CRC32 plus local SHA256'}
    (args.output / 'download-receipt.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

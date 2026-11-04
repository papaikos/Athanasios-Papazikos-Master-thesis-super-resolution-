import base64
import gzip
import hashlib

import google_crc32c
import pytest

from scripts.maritime_metadata_publish_gce import expand_verified_metadata


def identity(raw):
    return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
        crc32c=base64.b64encode(google_crc32c.Checksum(raw).digest()).decode())


def test_compressed_metadata_retains_exact_original_bytes(tmp_path):
    raw=b'{"immutable":"metadata"}\n'*10000
    source=tmp_path/'wire.gz';source.write_bytes(gzip.compress(raw))
    destination=tmp_path/'plain.json'
    expand_verified_metadata(source,destination,identity(raw))
    assert destination.read_bytes()==raw


@pytest.mark.parametrize('changed',['bytes','sha256','crc32c'])
def test_wrong_decoded_metadata_identity_fails_closed(tmp_path,changed):
    raw=b'{"immutable":true}'
    source=tmp_path/'wire.gz';source.write_bytes(gzip.compress(raw));expected=identity(raw)
    expected[changed]=len(raw)-1 if changed=='bytes' else '0'*64 if changed=='sha256' else 'AAAAAA=='
    with pytest.raises(ValueError,match='metadata'):
        expand_verified_metadata(source,tmp_path/'plain.json',expected)

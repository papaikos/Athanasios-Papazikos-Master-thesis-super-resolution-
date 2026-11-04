import hashlib
import io
import pytest
import numpy as np
from scripts.maritime_disaster_intake import (download_pinned, validate_source,
    metadata_status, udm_support, mirror_original)


def test_cloudflare_forbidden_uses_verified_public_s3_transport(tmp_path):
    from urllib.error import HTTPError
    requests=[]
    def opener(request, timeout):
        requests.append(request)
        assert request.get_header('User-agent')=='MaritimeThesisDataset/1.0'
        assert request.get_header('If-match')=='"abc-2"'
        if request.full_url.startswith('https://data.source.coop/'):
            raise HTTPError(request.full_url,403,'Forbidden',{},None)
        return Response(b'' if request.get_method()=='HEAD' else b'data')
    url='https://data.source.coop/planet/disasterdata/event/visual.tif'
    receipt=download_pinned(url,dict(etag='"abc-2"',content_length=4,version_id='v1'),tmp_path/'image.tif',opener=opener)
    assert receipt['source_url']==url
    assert receipt['transport_url']=='https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/planet/disasterdata/event/visual.tif?versionId=v1'
    assert [r.get_method() for r in requests]==['GET','HEAD','GET']


def test_transport_fallback_never_relaxes_original_version_pin(tmp_path):
    from urllib.error import HTTPError
    requests=[]
    def opener(request, timeout):
        requests.append(request)
        if len(requests)==1: raise HTTPError(request.full_url,403,'Forbidden',{},None)
        return Response(b'',version='different')
    with pytest.raises(ValueError,match='version'):
        download_pinned('https://data.source.coop/planet/disasterdata/event/visual.tif',
            dict(etag='"abc-2"',content_length=4,version_id='v1'),tmp_path/'raw',opener=opener)
    assert len(requests)==2 and not list(tmp_path.iterdir())


class Response(io.BytesIO):
    def __init__(self, data, etag='"abc-2"', length=4, version='v1'):
        super().__init__(data)
        self.headers = {'ETag': etag, 'Content-Length': str(length), 'x-amz-version-id': version}
        self.status = 200


def test_download_uses_ifmatch_and_verifies_digest(tmp_path):
    seen = []
    def opener(request, timeout):
        seen.append(request.get_header('If-match'))
        return Response(b'data')
    result = download_pinned('https://data.source.coop/file', dict(etag='"abc-2"', content_length=4, version_id='v1'),
                             tmp_path / 'image.tif', opener=opener, checksum='1220' + hashlib.sha256(b'data').hexdigest())
    assert seen == ['"abc-2"']
    assert result['sha256'] == hashlib.sha256(b'data').hexdigest()


@pytest.mark.parametrize('response', [Response(b'data', etag='"wrong"'), Response(b'data', length=5), Response(b'data', version='wrong')])
def test_changed_headers_fail_closed(tmp_path, response):
    with pytest.raises(ValueError):
        download_pinned('https://data.source.coop/file', dict(etag='"abc-2"', content_length=4, version_id='v1'),
                        tmp_path / 'image.tif', opener=lambda *a, **k: response)
    assert not list(tmp_path.iterdir())


def test_interrupted_download_and_checksum_failure_remove_partial(tmp_path):
    class Interrupted(Response):
        def read(self, size=-1):
            raise IOError('interrupted')
    pin = dict(etag='"abc-2"', content_length=4, version_id='v1')
    with pytest.raises(IOError):
        download_pinned('https://data.source.coop/file', pin, tmp_path / 'raw', opener=lambda *a, **k: Interrupted(b'data'))
    with pytest.raises(ValueError, match='checksum'):
        download_pinned('https://data.source.coop/file', pin, tmp_path / 'raw', opener=lambda *a, **k: Response(b'data'), checksum='1220' + '0' * 64)
    assert not list(tmp_path.iterdir())


def test_license_native_gsd_and_metadata_skip():
    row = dict(license='CC-BY-NC-4.0', native_gsd_m=.7, product_grid_m=.5,
               constellation='skysat', strip_id='s1', source_stac={'properties': {'gsd': .7}})
    validate_source(row)
    row['license'] = 'commercial'
    with pytest.raises(ValueError): validate_source(row)
    row['license'], row['native_gsd_m'] = 'CC-BY-NC-4.0', .9
    with pytest.raises(ValueError): validate_source(row)
    assert metadata_status({'intake_status': 'rejected_too_narrow'}) == 'skipped_metadata_geometry'


def test_udm_bits_preserve_original_zero_brightness_and_reject_quality_flags():
    assert np.array_equal(udm_support(np.array([[0, 1, 2, 4, 128]], np.uint8)), [[True, False, False, False, False]])


def test_mirror_pins_generation_and_verifies_crc32c_and_sha(tmp_path):
    import base64
    import google_crc32c
    path = tmp_path / 'original.tif'
    path.write_bytes(b'data')
    class Blob:
        size, generation = 4, '17'
        crc32c = base64.b64encode(google_crc32c.Checksum(b'data').digest()).decode()
        def exists(self): return True
        def reload(self): pass
        def download_to_filename(self, filename, **kwargs):
            assert kwargs == dict(if_generation_match=17, checksum='crc32c')
            from pathlib import Path
            Path(filename).write_bytes(b'data')
    class Bucket:
        name = 'test-bucket'
        object = Blob()
        def blob(self, name): return self.object
    bucket = Bucket()
    ref = dict(bytes=4, sha256=hashlib.sha256(b'data').hexdigest())
    output = mirror_original(bucket, 'original.tif', path, ref, tmp_path)
    assert output['generation'] == '17' and output['sha256'] == ref['sha256']
    bucket.object.crc32c = 'bad'
    with pytest.raises(ValueError, match='CRC32C'):
        mirror_original(bucket, 'original.tif', path, ref, tmp_path)

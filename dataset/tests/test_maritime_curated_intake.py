import io
import zipfile
import json
from types import SimpleNamespace
import pytest
from PIL import Image
from scripts import maritime_curated_intake as intake


@pytest.mark.parametrize('value', [None, '', 'nan', 'inf', '0', '-1', '1.01'])
def test_only_documented_finite_submetre_gsd(value):
    assert intake.accepted_gsd(value) is None


def test_submetre_gsd_preserved_without_upsampling():
    assert intake.accepted_gsd('0.305') == .305
    assert intake.accepted_gsd('1') == 1.


def test_unsafe_zip_members_fail_before_extract():
    for name in ('../image.png', '/root/image.png', 'C:/image.png', 'a\\..\\image.png'):
        with pytest.raises(ValueError):
            intake.safe_member(name)
    assert intake.safe_member('dataset/images/a.png') == 'dataset/images/a.png'


def test_rgb_original_bytes_not_synthetic_colour_conversion():
    stream = io.BytesIO()
    Image.new('L', (8, 8)).save(stream, format='PNG')
    with pytest.raises(ValueError, match='RGB'):
        intake.inspect_rgb(stream.getvalue())
    stream = io.BytesIO()
    Image.new('RGB', (8, 6), (0, 0, 0)).save(stream, format='PNG')
    assert intake.inspect_rgb(stream.getvalue()) == (8, 6, 144)


def test_decoded_dedup_detects_equal_pixels_different_encodings():
    outputs = []
    for level in (0, 9):
        stream = io.BytesIO()
        Image.new('RGB', (8, 8), (3, 4, 5)).save(stream, format='PNG', compress_level=level)
        outputs.append(stream.getvalue())
    assert outputs[0] != outputs[1]
    assert intake.pixel_identity(outputs[0]) == intake.pixel_identity(outputs[1])


def test_confirmation_form_reuses_hidden_fields_and_validates_host():
    document = '<form action="https://drive.usercontent.google.com/download" method="get"><input name="id" value="abc"><input name="confirm" value="t"><input name="uuid" value="u"></form>'
    assert intake.confirm_url(document).startswith('https://drive.usercontent.google.com/download?')
    with pytest.raises(ValueError):
        intake.confirm_url(document.replace('drive.usercontent.google.com', 'evil.example'))


def test_annotation_gsd_is_used_without_source_family_inference():
    xml = b'<annotation><source><dataset_source>xView</dataset_source></source><filename>a.bmp</filename><Img_Resolution>0.3</Img_Resolution><size><width>8</width><height>6</height><depth>3</depth></size><object><name>Yacht</name><level_0>1</level_0></object><object><name>Dock</name><level_0>0</level_0></object></annotation>'
    stream = io.BytesIO()
    Image.new('RGB', (8, 6)).save(stream, format='BMP')
    row = intake.qualified_annotation(xml, stream.getvalue())
    assert row['native_gsd_m'] == .3
    assert row['native_rgb_bytes'] == 144
    assert row['ship_count'] == 1 and row['dock_count'] == 1
    assert row['physical_extent_m'] == pytest.approx([2.4, 1.8])
    assert row['georeferencing'] is None
    assert intake.qualified_annotation(xml.replace(b'0.3', b'1.07'), stream.getvalue())['status'] == 'REJECTED_GSD_COARSER_THAN_1M'
    assert intake.qualified_annotation(xml.replace(b'0.3', b''), stream.getvalue())['status'] == 'REJECTED_GSD_UNKNOWN'


def test_annotation_dimensions_must_match_original_rgb():
    xml = b'<annotation><filename>a.bmp</filename><Img_Resolution>0.3</Img_Resolution><size><width>9</width><height>6</height><depth>3</depth></size></annotation>'
    stream = io.BytesIO()
    Image.new('RGB', (8, 6)).save(stream, format='BMP')
    with pytest.raises(ValueError, match='dimensions'):
        intake.qualified_annotation(xml, stream.getvalue())


def test_qualified_manifest_preserves_split_pixels_and_rejects_unknown(tmp_path, monkeypatch):
    stream = io.BytesIO()
    Image.new('RGB', (8, 6), (4, 5, 6)).save(stream, format='BMP')
    pixels = stream.getvalue()
    archive = tmp_path / 'source.zip'
    xml_template = '<annotation><source><dataset_source>xView</dataset_source></source><filename>{name}.bmp</filename><Img_Resolution>{gsd}</Img_Resolution><size><width>8</width><height>6</height><depth>3</depth></size><object><name>Yacht</name></object></annotation>'
    with zipfile.ZipFile(archive, 'w') as target:
        for name, gsd in [('a', '0.3'), ('b', '0.3'), ('c', '')]:
            target.writestr(f'dataset/VOC_Format/Annotations/{name}.xml', xml_template.format(name=name, gsd=gsd))
            target.writestr(f'dataset/VOC_Format/JPEGImages/{name}.bmp', pixels)
        for split, text in [('train', 'a.bmp\n'), ('val', 'b.bmp\n'), ('test', 'c.bmp\n')]:
            target.writestr(f'dataset/VOC_Format/ImageSets/{split}.txt', text)
    uploaded = {}
    def upload(blob, data):
        uploaded[blob] = data
        return {'generation': '1', 'bytes': len(data), 'crc32c': 'test', 'sha256': intake.hashlib.sha256(data).hexdigest()}
    monkeypatch.setattr(intake, 'create_only', upload)
    client = SimpleNamespace(bucket=lambda name: SimpleNamespace(blob=lambda key: key))
    report = intake.qualify_archive(client, archive, {'uri': 'gs://b/source.zip'}, 'gs://b/intake', tmp_path)
    assert report['qualified_images'] == 1
    assert report['unique_native_rgb_bytes'] == 144
    assert report['qualified_nonoverlap_native_bytes'] is None
    assert report['status_counts'] == {'QUALIFIED_GSD_RGB_LABELED': 1, 'DUPLICATE_NATIVE_PIXELS': 1, 'REJECTED_GSD_UNKNOWN': 1}
    assert report['records'][0]['official_splits'] == ['train']
    assert next(data for name, data in uploaded.items() if name.endswith('.bmp')) == pixels
    stored = json.loads(next(data for name, data in uploaded.items() if name.endswith('qualified-manifest.json')))
    assert stored['final_dataset_ready'] is False

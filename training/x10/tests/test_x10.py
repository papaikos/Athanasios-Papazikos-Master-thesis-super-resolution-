"""CPU tests for the x10 pieces (run locally and on AUTH before the benchmark)."""
import io
import json
import os
from pathlib import Path
import tarfile

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
from research import data as d  # noqa: E402
from research.model_factory import CONFIG, validate_config, make_model  # noqa: E402
from research.train import learning_rate, masked_l1, pack_state, restore_state, optimizer_for  # noqa: E402

DATA_CANDIDATES = [os.environ.get('D3X10_DATA', ''), os.environ.get('DATA_DIR', ''),
                   str(ROOT.parent / 'Maritime_D3x10_1m10m')]
DATA = next((Path(p) for p in DATA_CANDIDATES if p and (Path(p) / 'manifest.json').exists()), None)


def synthetic_pair():
    rng = np.random.default_rng(0)
    lr = rng.integers(0, 256, (3, 128, 128), dtype=np.uint8)
    hr = np.kron(lr, np.ones((1, 10, 10), dtype=np.uint8))  # nearest x10: exact alignment probe
    return {'lr': lr, 'hr': hr, 'lr_mask': np.ones((128, 128), bool), 'hr_mask': np.ones((1280, 1280), bool)}


@pytest.mark.parametrize('position', range(40))
def test_x10_crop_alignment_all_geometries(position):
    crop, choice = d.aligned_crop(synthetic_pair(), position)
    assert crop['lr'].shape == (3, 64, 64) and crop['hr'].shape == (3, 640, 640)
    assert crop['hr_mask'].shape == (640, 640) and crop['hr_mask'].dtype == torch.bool
    up = crop['lr'].repeat_interleave(10, -2).repeat_interleave(10, -1)
    assert torch.equal(up, crop['hr']), choice
    assert 0 <= choice['x'] <= 64 and 0 <= choice['y'] <= 64 and 0 <= choice['geometry'] < 8


def test_all_eight_geometries_and_full_crop_range_occur():
    choices = [d.crop_choice(p) for p in range(4000)]
    assert {g for _, _, g in choices} == set(range(8))
    assert min(min(y, x) for y, x, _ in choices) == 0 and max(max(y, x) for y, x, _ in choices) == 64


def test_positions_restart_reproduces_stream():
    rows = [{'sample_id': str(i)} for i in range(37)]
    ds = d.TrainPositions('.', rows)
    first = [ds.sample_index(p) for p in range(0, 200)]
    again = d.TrainPositions('.', rows)
    assert [again.sample_index(p) for p in range(100, 200)] == first[100:]
    for epoch in range(5):  # each epoch is a permutation: every sample once
        assert sorted(first[epoch * 37:(epoch + 1) * 37]) == list(range(37))
    batches = iter(d.PositionBatches(48))
    assert next(batches) == list(range(48, 64)) and next(batches) == list(range(64, 80))


def test_schedule_matches_d_series():
    assert learning_rate(1) == pytest.approx(1e-7)
    assert learning_rate(2000) == pytest.approx(2e-4)
    assert learning_rate(100000) == pytest.approx(1e-6)
    with pytest.raises(ValueError):
        learning_rate(100001)


def test_masked_l1_is_batch_mean_with_full_support():
    p, t = torch.rand(16, 3, 8, 8), torch.rand(16, 3, 8, 8)
    mask = torch.ones(16, 8, 8, dtype=torch.bool)
    assert float(masked_l1(p, t, mask)) == pytest.approx(float((p - t).abs().mean()))
    # equals the D2 4x4 accumulation (mean of microbatch means) when support is full
    parts = sum(float(masked_l1(p[i:i + 4], t[i:i + 4], mask[i:i + 4])) * 4 / 16 for i in range(0, 16, 4))
    assert float(masked_l1(p, t, mask)) == pytest.approx(parts)


def test_config_file_is_the_frozen_protocol():
    path = ROOT / 'source/research/configs/d3x10_swinir_100k.yaml'
    assert validate_config(yaml.safe_load(path.read_text())) == CONFIG
    assert CONFIG['training']['effective_batch_size'] == 16 and CONFIG['training']['hr_crop_size'] == 640


def test_model_shapes_and_head():
    torch.manual_seed(42)
    model, info = make_model(ROOT / 'source/vendor/network_swinir.py')
    assert model.upscale == 10 and info['pretrained'] is False
    assert [type(m).__name__ for m in model.upsample] == ['Conv2d', 'PixelShuffle', 'Conv2d', 'PixelShuffle']
    assert model.upsample[1].upscale_factor == 2 and model.upsample[3].upscale_factor == 5
    with torch.inference_mode():
        assert model.eval()(torch.rand(1, 3, 64, 64)).shape == (1, 3, 640, 640)


def test_checkpoint_state_roundtrip():
    torch.manual_seed(0)
    model = torch.nn.Conv2d(3, 3, 1)
    opt = optimizer_for(model)
    model(torch.rand(1, 3, 4, 4)).sum().backward()
    opt.step()
    state = pack_state(model, opt, 48, 37, 3, {'x': 1}, {'id': 1}, None)
    buffer = io.BytesIO()
    torch.save(state, buffer)
    loaded = torch.load(io.BytesIO(buffer.getvalue()), weights_only=False)
    other = torch.nn.Conv2d(3, 3, 1)
    cursor = restore_state(loaded, other, optimizer_for(other), 37, {'id': 1})
    assert cursor == 48 and torch.equal(other.weight, model.weight)
    with pytest.raises(ValueError):
        restore_state(loaded, other, optimizer_for(other), 36, {'id': 1})


@pytest.mark.skipif(DATA is None, reason='D3x10 data not present')
def test_read_real_pair_by_offset():
    manifest = json.loads((DATA / 'manifest.json').read_text())
    shard = next(s for s in manifest['shards'] if s['split'] == 'validation')
    with tarfile.open(DATA / shard['file']) as tar:
        member = tar.getmembers()[0]
    row = {'sample_id': member.name[:-4], 'shard': shard['file'], 'offset': member.offset_data, 'size': member.size}
    pair = d.read_pair(DATA, row)
    assert pair['hr'].shape == (3, 1280, 1280) and pair['lr'].shape == (3, 128, 128)
    full = d.FullPairs(DATA, [row])[0]
    assert full['hr'].dtype == torch.float32 and float(full['hr'].max()) <= 1.0 and full['sample_id'] == row['sample_id']


@pytest.mark.skipif(DATA is None, reason='D3x10 data not present')
def test_pooled_scoring_is_identical_to_serial(tmp_path):
    from research.evaluate import evaluate
    manifest = json.loads((DATA / 'manifest.json').read_text())
    shard = next(s for s in manifest['shards'] if s['split'] == 'validation')
    with tarfile.open(DATA / shard['file']) as tar:
        members = tar.getmembers()[:3]
    rows = [{'sample_id': m.name[:-4], 'shard': shard['file'], 'offset': m.offset_data, 'size': m.size} for m in members]
    pairs, model = d.FullPairs(DATA, rows), torch.nn.Conv2d(3, 3, 1)
    serial = evaluate(model, pairs, 'cpu', tmp_path / 'a.csv', 3, bicubic=True)
    pooled = evaluate(model, pairs, 'cpu', tmp_path / 'b.csv', 3, bicubic=True, score_workers=2)
    assert serial == pooled
    assert (tmp_path / 'a.csv').read_bytes() == (tmp_path / 'b.csv').read_bytes()

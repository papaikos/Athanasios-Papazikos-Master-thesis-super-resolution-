"""Classical SwinIR x10 from the pinned D-series source; no weight loading.

Everything except the upsampling head equals the D1-D3 x4 protocol. SwinIR has
no x10 head, so (user decision 2026-10-21) the head is SwinIR's own x2 stage
[conv 64->256, PixelShuffle(2)] followed by a x5 stage [conv 64->1600,
PixelShuffle(5)], between the unchanged conv_before_upsample and conv_last.
"""
import importlib.util
from .checkpointing import sha256

MODEL_SHA = '9e143898679ebeebc5d2fc94ad1b89c38aa4a4d43da4e0fcba0f93e476994913'
MODEL = dict(in_chans=3, upscale=10, img_size=64, patch_size=1, window_size=8, embed_dim=180, depths=[6] * 6,
             num_heads=[6] * 6, mlp_ratio=2.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0,
             drop_path_rate=0.1, ape=False, patch_norm=True, use_checkpoint=False, img_range=1.0,
             upsampler='pixelshuffle', resi_connection='1conv')
HEAD = 'x2-then-x5-pixelshuffle-v1'
CONFIG = {
    'schema': 1,
    'experiment': 'maritime-d3x10-swinir-x10-100k-v1',
    'seed': 42,
    'initialization': 'random_from_scratch',
    'model': dict(MODEL, x10_head=HEAD),
    'optimizer': {'name': 'Adam', 'lr_peak': .0002, 'betas': [.9, .99], 'eps': 1e-8, 'weight_decay': 0.0},
    'training': {'target_updates': 100000, 'warmup_updates': 2000, 'min_lr': 1e-6, 'effective_batch_size': 16,
                 'microbatch_size': 16, 'accumulation_steps': 1, 'lr_crop_size': 64, 'hr_crop_size': 640,
                 'scale': 10, 'loss': 'masked_l1', 'precision': 'bf16', 'ema': False, 'early_stopping': False,
                 'gradient_clipping': False, 'deterministic_algorithms': False, 'tf32': False},
    'data': {'dataset': 'D3x10 1m/10m (rules_v5_d3x10 rejection + 88 manual restores)', 'train': 16835,
             'validation': 115, 'test': 'sealed, not on AUTH', 'recipe': 'd3x10-variable-psf-noise-v1'},
}


def validate_config(value):
    def matches(actual, expected):
        if type(actual) is not type(expected):
            return False
        if isinstance(expected, dict):
            return actual.keys() == expected.keys() and all(matches(actual[k], v) for k, v in expected.items())
        if isinstance(expected, list):
            return len(actual) == len(expected) and all(matches(a, b) for a, b in zip(actual, expected))
        return actual == expected
    if not matches(value, CONFIG):
        raise ValueError('Configuration differs from the frozen D3x10 protocol')
    return value


def make_model(source):
    import torch
    from torch import nn
    if sha256(source) != MODEL_SHA:
        raise ValueError('Unpinned SwinIR source')
    spec = importlib.util.spec_from_file_location('pinned_swinir', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # SwinIR's Upsample only supports 2^n and 3, so build with its x2 stage and append x5.
    model = module.SwinIR(norm_layer=nn.LayerNorm, **dict(MODEL, upscale=2))
    num_feat = model.conv_last.in_channels
    model.upsample = nn.Sequential(*model.upsample, nn.Conv2d(num_feat, 25 * num_feat, 3, 1, 1), nn.PixelShuffle(5))
    model.upscale = 10
    info = {'source_sha256': MODEL_SHA, 'constructor': dict(MODEL, norm_layer='torch.nn.LayerNorm'),
            'x10_head': HEAD, 'upsample_modules': [repr(m) for m in model.upsample],
            'parameter_count': sum(p.numel() for p in model.parameters()),
            'internal_rgb_mean': [.4488, .4371, .4040], 'pretrained': False}
    return model, info

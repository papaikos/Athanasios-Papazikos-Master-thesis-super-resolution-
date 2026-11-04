"""Explicit, immutable classical SwinIR configuration; no weight loading."""
import importlib.util
from .checkpointing import sha256
from .experiment import KEY

MODEL_SHA='9e143898679ebeebc5d2fc94ad1b89c38aa4a4d43da4e0fcba0f93e476994913'
MODEL=dict(in_chans=3,upscale=4,img_size=64,patch_size=1,window_size=8,embed_dim=180,depths=[6]*6,num_heads=[6]*6,mlp_ratio=2.0,qkv_bias=True,qk_scale=None,drop_rate=0.0,attn_drop_rate=0.0,drop_path_rate=0.1,ape=False,patch_norm=True,use_checkpoint=False,img_range=1.0,upsampler='pixelshuffle',resi_connection='1conv')
CONFIG={'schema':1,'experiment':f'maritime-{KEY}-swinir-confirmation-100k-v1','seed':42,'initialization':'random_from_scratch','model':MODEL,'optimizer':{'name':'Adam','lr_peak':.0002,'betas':[.9,.99],'eps':1e-8,'weight_decay':0.0},'training':{'target_updates':100000,'warmup_updates':2000,'min_lr':1e-6,'effective_batch_size':16,'lr_crop_size':64,'hr_crop_size':256,'loss':'masked_l1','precision':'bf16','ema':False,'early_stopping':False,'gradient_clipping':False}}

def validate_config(value):
    def matches(actual,expected):
        if type(actual) is not type(expected):return False
        if isinstance(expected,dict):return actual.keys()==expected.keys() and all(matches(actual[k],v) for k,v in expected.items())
        if isinstance(expected,list):return len(actual)==len(expected) and all(matches(a,b) for a,b in zip(actual,expected))
        return actual==expected
    if not matches(value,CONFIG):raise ValueError('Configuration schema or value differs from frozen confirmation protocol')
    return value

def make_model(source):
    import torch
    if sha256(source)!=MODEL_SHA:raise ValueError('Unpinned SwinIR source')
    spec=importlib.util.spec_from_file_location('pinned_swinir',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model=module.SwinIR(norm_layer=torch.nn.LayerNorm,**MODEL)
    return model,{'source_sha256':MODEL_SHA,'constructor':dict(MODEL,norm_layer='torch.nn.LayerNorm'),'parameter_count':sum(p.numel() for p in model.parameters()),'internal_rgb_mean':[.4488,.4371,.4040],'pretrained':False}

"""Frozen AlexNet LPIPS v0.1 diagnostic; separate from scratch SwinIR."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import torch
from .evaluate import rng_state,restore_rng
from .checkpointing import sha256

CALIBRATION_SHA256='df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0'
ALEXNET_SHA256='7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02'
SETTINGS={'net':'alex','version':'0.1','lpips':True,'pretrained':True,'pnet_rand':False,'pnet_tune':False,'spatial':False,'eval_mode':True,'use_dropout':True,'verbose':False}

class LPIPSMetric:
    def __init__(self,device,*,factory=None,alexnet_path=None,calibration_path=None):
        state=rng_state();self.device=torch.device(device);self.provenance={}
        try:
            if factory is None:
                import lpips
                if version('lpips')!='0.1.4':raise ValueError('Pinned lpips==0.1.4 required')
                calibration_path=Path(calibration_path) if calibration_path else Path(lpips.__file__).parent/'weights/v0.1/alex.pth'
                if sha256(calibration_path)!=CALIBRATION_SHA256:raise ValueError('LPIPS calibration pin mismatch')
                # Require the exact cache location torchvision will consume, before
                # construction; no implicit unverified backbone download is allowed.
                cached_alexnet=Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth'
                alexnet_path=Path(alexnet_path) if alexnet_path else cached_alexnet
                if alexnet_path.resolve()!=cached_alexnet.resolve():raise ValueError('AlexNet pin must identify the actual torchvision cache file')
                if not alexnet_path.is_file() or sha256(alexnet_path)!=ALEXNET_SHA256:raise ValueError('Verified official AlexNet cache pin required before LPIPS construction')
                factory=lpips.LPIPS
                kwargs=dict(SETTINGS,model_path=str(calibration_path))
                package='0.1.4'
            else:kwargs=dict(SETTINGS);package='injected-test-fixture'
            self.network=factory(**kwargs).to(self.device).float().eval()
            self.network.requires_grad_(False)
            digest=hashlib.sha256()
            for key,value in sorted(self.network.state_dict().items()):
                value=value.detach().cpu().contiguous()
                digest.update(json.dumps([key,str(value.dtype),list(value.shape)],separators=(',',':')).encode());digest.update(value.view(torch.uint8).numpy().tobytes())
            self.provenance={'metric_schema':'lpips-alex-v0.1-v1','package':'lpips','package_version':package,'settings':SETTINGS,'precision':'fp32','normalization':'prediction clipped [0,1]; RGB 2*x-1; normalize=False','spatial_support':'full-support only; partial masks rejected','aggregation':'arithmetic mean per image; lower is better','network_state_sha256':digest.hexdigest(),'role':'frozen pretrained diagnostic feature network; never optimizer or checkpoint selection'}
            if calibration_path:self.provenance['calibration_file']={'sha256':sha256(calibration_path),'bytes':Path(calibration_path).stat().st_size}
            if alexnet_path:self.provenance['alexnet_file']={'sha256':sha256(alexnet_path),'bytes':Path(alexnet_path).stat().st_size,'url':'https://download.pytorch.org/models/alexnet-owt-7be5be79.pth'}
        finally:restore_rng(state)

    def score(self,prediction,target,mask):
        if prediction.shape!=target.shape or prediction.ndim!=3 or prediction.shape[0]!=3 or mask.shape!=prediction.shape[-2:]:raise ValueError('Matching CHW RGB inputs required for LPIPS')
        if mask.dtype!=torch.bool or not bool(mask.all()):raise ValueError('LPIPS requires full support; masked feature distances are undefined')
        if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(target).all()):raise ValueError('LPIPS requires finite RGB')
        if bool(((target<0)|(target>1)).any()):raise ValueError('LPIPS HR must be normalized [0,1]')
        with torch.inference_mode(),torch.autocast(device_type=self.device.type,enabled=False):
            x=prediction.detach().to(device=self.device,dtype=torch.float32).clamp(0,1).unsqueeze(0)*2-1
            y=target.detach().to(device=self.device,dtype=torch.float32).unsqueeze(0)*2-1
            result=self.network(x,y,normalize=False)
        if result.numel()!=1 or not bool(torch.isfinite(result).all()):raise ValueError('LPIPS returned malformed/nonfinite distance')
        return float(result.item())

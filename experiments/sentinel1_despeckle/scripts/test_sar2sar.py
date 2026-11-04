import ddespeckle_stub  # noqa
import numpy as np, inspect
from deepdespeckling.sar2sar.sar2sar_denoiser import Sar2SarDenoiser
from deepdespeckling.utils import utils as U
print("normalize src:\n", inspect.getsource(U.normalize_sar_image))
print("denormalize src:\n", inspect.getsource(U.denormalize_sar_image))
d = Sar2SarDenoiser()
print("denoise_image sig:", inspect.signature(d.denoise_image))
# tiny smoke test
img = (np.random.rand(256,256).astype(np.float32)*300+20)
from deepdespeckling.utils.constants import PATCH_SIZE, STRIDE_SIZE
print("PATCH", PATCH_SIZE, "STRIDE", STRIDE_SIZE)
out = d.denoise_image(img, patch_size=PATCH_SIZE, stride_size=STRIDE_SIZE)
print("OUT type", type(out), "keys" if isinstance(out,dict) else "", out.keys() if isinstance(out,dict) else out.shape)

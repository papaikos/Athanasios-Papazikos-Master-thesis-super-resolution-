"""Run both despecklers on every extracted chip:
  model 1 = SAR2SAR (deep, self-supervised, deepdespeckling pretrained)
  model 2 = SAR-BM3D (homomorphic BM3D)
Saves amplitude .npy outputs, matched-stretch preview PNGs, and metrics.json."""
import ddespeckle_stub  # noqa: F401  (stubs osgeo + patches torch.load)
import os, json, glob, numpy as np
from PIL import Image
from deepdespeckling.sar2sar.sar2sar_denoiser import Sar2SarDenoiser
from deepdespeckling.utils.constants import PATCH_SIZE, STRIDE_SIZE
from sar_bm3d import sar_bm3d

ROOT = os.path.join(os.path.dirname(__file__), "..")
CHIPS = os.path.join(ROOT, "data", "chips")
OUT = os.path.join(ROOT, "out")
for d in ("sar2sar", "merlin", "orig"):
    os.makedirs(os.path.join(OUT, d), exist_ok=True)

def db_stretch(amp, lo=None, hi=None):
    a = np.where(np.isfinite(amp) & (amp > 0), amp, np.nan).astype(np.float32)
    db = 10.0 * np.log10(a + 1e-3)
    if lo is None:
        lo, hi = np.nanpercentile(db, 2), np.nanpercentile(db, 98)
    out = np.clip((db - lo) / max(hi - lo, 1e-6), 0, 1)
    return np.nan_to_num(out, nan=0.0), lo, hi

def flat_patch(amp, k=48):
    """Find flattest kxk region (min coeff of variation) -> homogeneous water for ENL."""
    from scipy.ndimage import uniform_filter
    a = amp.astype(np.float64)
    m = uniform_filter(a, k); m2 = uniform_filter(a * a, k)
    var = np.clip(m2 - m * m, 0, None)
    cv = np.sqrt(var) / (m + 1e-6)
    cv[m < np.percentile(m, 20)] = np.inf  # avoid pure-nodata/very dark edges
    h = k // 2
    cv[:h] = cv[-h:] = np.inf; cv[:, :h] = cv[:, -h:] = np.inf
    r, c = np.unravel_index(np.argmin(cv), cv.shape)
    return int(r - h), int(c - h), k

def enl(amp, box):
    r, c, k = box
    p = amp[r:r + k, c:c + k].astype(np.float64)
    return float((p.mean() / (p.std() + 1e-9)) ** 2)

# curated 10: maritime scenes with clear vessels on water (best boats + speckle demos)
SELECT = [
    "chittagong_anchor", "dover_channel", "port_klang_malacca", "istanbul_marmara",
    "bosphorus_north", "panama_pacific", "singapore_east", "singapore_strait",
    "hong_kong_anchorage", "piraeus_saronic",
]
den = Sar2SarDenoiser()
man_all = {m["name"]: m for m in json.load(open(os.path.join(CHIPS, "manifest.json")))}
manifest = [man_all[n] for n in SELECT if n in man_all]
metrics = []
for entry in manifest:
    name = entry["name"]
    amp = np.load(os.path.join(CHIPS, f"{name}.npy")).astype(np.float32)
    print(f"== {name} ==", amp.shape)

    # model 1: SAR2SAR (expects amplitude)
    s2s = den.denoise_image(amp, patch_size=PATCH_SIZE, stride_size=STRIDE_SIZE)["denoised"]
    s2s = np.asarray(s2s, dtype=np.float32)
    # model 2: SAR-BM3D
    bm = sar_bm3d(amp).astype(np.float32)

    np.save(os.path.join(OUT, "sar2sar", f"{name}.npy"), s2s)
    np.save(os.path.join(OUT, "merlin", f"{name}.npy"), bm)  # dir name kept generic

    # matched display stretch from ORIGINAL
    o_img, lo, hi = db_stretch(amp)
    s_img, _, _ = db_stretch(s2s, lo, hi)
    b_img, _, _ = db_stretch(bm, lo, hi)
    Image.fromarray((o_img * 255).astype(np.uint8)).save(os.path.join(OUT, "orig", f"{name}.png"))
    Image.fromarray((s_img * 255).astype(np.uint8)).save(os.path.join(OUT, "sar2sar", f"{name}.png"))
    Image.fromarray((b_img * 255).astype(np.uint8)).save(os.path.join(OUT, "merlin", f"{name}.png"))

    box = flat_patch(amp)
    m = {"name": name, "scene": entry["scene"], "band": entry.get("band"),
         "datetime": entry["datetime"], "n_ships": entry["n_ships"],
         "enl_orig": round(enl(amp, box), 2),
         "enl_sar2sar": round(enl(s2s, box), 2),
         "enl_bm3d": round(enl(bm, box), 2)}
    metrics.append(m)
    print("   ENL orig=%.1f  sar2sar=%.1f  bm3d=%.1f" %
          (m["enl_orig"], m["enl_sar2sar"], m["enl_bm3d"]))

json.dump(metrics, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
print("\nDONE ->", os.path.join(OUT, "metrics.json"))

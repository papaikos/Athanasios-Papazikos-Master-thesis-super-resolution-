"""Common-grid comparison of x4 and x10 on the stored SkySat x10 demonstration outputs (no model inference).

At 1 m:   bicubic x10 | D3 x4 output + bicubic 2.5->1 m | D3x10   vs the 1 m reference
At 2.5 m: bicubic x4  | D3 x4 | D3x10 output area-reduced 1->2.5 m   vs the 2.5 m reference
"""
import numpy as np
import pandas as pd
from PIL import Image

from common import DATA, RUN, bicubic_up, psnr


def area_matrix(n_in, n_out):
    """Exact 1-D area weights from n_in cells to n_out cells covering the same extent."""
    r = n_in / n_out
    W = np.zeros((n_out, n_in))
    for i in range(n_out):
        a, b = i * r, (i + 1) * r
        for p in range(int(np.floor(a)), int(np.ceil(b))):
            W[i, p] = (min(b, p + 1) - max(a, p)) / r
    return W


def area_reduce(chw, n_out):
    c, h, w = chw.shape
    Wh = area_matrix(h, n_out); Ww = area_matrix(w, n_out)
    return np.stack([Wh @ chw[k] @ Ww.T for k in range(c)])


rows = []
for i in range(1, 11):
    base = RUN["D3x10"] / "skysat-comparisons/results" / f"skysat-x10-{i:02d}"
    h1 = np.asarray(Image.open(base / "HR-reference-1m.png").convert("RGB"), np.float64).transpose(2, 0, 1) / 255
    h25 = np.asarray(Image.open(base / "HR-reference-2.5m.png").convert("RGB"), np.float64).transpose(2, 0, 1) / 255
    b10 = np.clip(np.load(base / "bicubic-x10-raw-fp32.npy"), 0, 1)
    b4 = np.clip(np.load(base / "bicubic-x4-raw-fp32.npy"), 0, 1)
    m10 = np.clip(np.load(base / "D3x10-best-raw-fp32.npy"), 0, 1)
    m4 = np.clip(np.load(base / "D3-x4-best-raw-fp32.npy"), 0, 1)
    assert h1.shape[1] == m10.shape[1] and h25.shape[1] == m4.shape[1]
    m4_up = np.clip(bicubic_up(m4, h1.shape[1] / m4.shape[1]), 0, 1)
    m10_dn = area_reduce(m10, h25.shape[1])
    rows += [dict(scene=i, grid="1 m", method="bicubic x10", psnr=psnr(b10, h1)),
             dict(scene=i, grid="1 m", method="D3 x4 + bicubic", psnr=psnr(m4_up, h1)),
             dict(scene=i, grid="1 m", method="D3x10", psnr=psnr(m10, h1)),
             dict(scene=i, grid="2.5 m", method="bicubic x4", psnr=psnr(b4, h25)),
             dict(scene=i, grid="2.5 m", method="D3 x4", psnr=psnr(m4, h25)),
             dict(scene=i, grid="2.5 m", method="D3x10 area-reduced", psnr=psnr(m10_dn, h25))]
    # sanity: reference consistency (1 m reference reduced to 2.5 m vs 2.5 m reference)
    rows.append(dict(scene=i, grid="check", method="ref1m->2.5m vs ref2.5m", psnr=psnr(area_reduce(h1, h25.shape[1]), h25)))
d = pd.DataFrame(rows)
d.to_csv(DATA / "x10_common_grid_per_scene.csv", index=False)
s = d.groupby(["grid", "method"]).psnr.agg(["mean", "std", "count"]).reset_index()
# paired differences
piv = d.pivot_table(index="scene", columns="method", values="psnr")
s.to_csv(DATA / "x10_common_grid.csv", index=False)
print(s.to_string())
for a, b in [("D3x10", "D3 x4 + bicubic"), ("D3x10 area-reduced", "D3 x4")]:
    diff = piv[a] - piv[b]
    print(a, "-", b, round(diff.mean(), 3), "wins", int((diff > 0).sum()), "/10")
print("shapes", h1.shape, h25.shape, m4.shape, m10.shape)

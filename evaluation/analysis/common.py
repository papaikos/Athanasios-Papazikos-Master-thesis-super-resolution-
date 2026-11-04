"""Shared paths, colours and plotting style for the thesis figures and tables.

Every input is only read. Outputs go to $ANALYSIS_OUTPUT/{figures,data}.
"""
import io
import json
import os
import tarfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Folder that holds the run archives ("D1 run", ..., "Jev check", ...); outputs go to ANALYSIS_OUTPUT.
DESK = Path(os.environ.get("MARITIME_ARCHIVE_ROOT", "."))
THESIS = Path(os.environ.get("ANALYSIS_OUTPUT", "analysis_output"))
FIG = THESIS / "figures"
DATA = THESIS / "data"
FIG.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

RUN = {k: DESK / f"{k} run" for k in ("D1", "D2", "D3", "D3x10")}
DATASET = DESK / "Maritime_RGB_Expanded_20261008"
X10DATA = DESK / "D3x10 build" / "Maritime_D3x10_1m10m"
JEV = DESK / "Jev check"
JEV_THESIS = JEV / "thesis_data" / "2026-10-20_105845Z_rules_v5_full_mother"
PILOT = DESK / "rgb-comp-SEN2SR-SwinIR"
RELEASED = DESK / "rgb-comp-DiffusionSat-SEN2SR-Pretrained"
S1 = DESK / "despecle sentinel 1"

# Validated categorical palette (dataviz reference palette, light mode).
C = {
    "D1": "#2a78d6",      # blue
    "D2": "#eb6834",      # orange
    "D3": "#1baf7a",      # aqua
    "D3x10": "#6250d6",   # violet
    "bicubic": "#8a8984", # neutral grey
    "SEN2SR-FT": "#e87ba4",
    "SwinIR-FT": "#008300",
}
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#dedcd6"

LABEL = {"D1": "D1 (bicubic AA)", "D2": "D2 (Gauss + area)", "D3": "D3 (μεταβλητό PSF + θόρυβος)",
         "D3x10": "D3x10 (×10)", "bicubic": "Bicubic"}


def style():
    plt.rcParams.update({
        "font.family": "Calibri",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.dpi": 110,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "lines.linewidth": 2.0,
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    })


style()


from matplotlib.ticker import ScalarFormatter


class CommaFormatter(ScalarFormatter):
    """Greek decimal comma on linear axes."""
    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace(".", ",")


def comma_axes(fig):
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if axis.get_scale() == "linear" and type(axis.get_major_formatter()) is ScalarFormatter:
                f = CommaFormatter(useOffset=False)
                f.set_scientific(False)
                axis.set_major_formatter(f)


def save(fig, name):
    comma_axes(fig)
    p = FIG / f"{name}.png"
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    print("figure", p.name)
    return p


def rgb(a):
    """CHW or HWC float array -> HWC float in [0,1] for display (clip only)."""
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 3 and a.shape[0] == 3:
        a = a.transpose(1, 2, 0)
    return np.clip(a, 0, 1)


def nearest_up(a, k):
    return np.repeat(np.repeat(a, k, axis=0), k, axis=1)


def bicubic_up(chw, scale):
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(np.ascontiguousarray(chw, dtype=np.float32))[None]
    out = F.interpolate(t, scale_factor=scale, mode="bicubic", align_corners=False)
    return out[0].numpy()


def area_down(chw, k):
    c, h, w = chw.shape
    return chw.reshape(c, h // k, k, w // k, k).mean((2, 4))


def psnr(a, b):
    a = np.clip(np.asarray(a, np.float64), 0, 1)
    b = np.asarray(b, np.float64)
    mse = np.mean((a - b) ** 2)
    return 10 * np.log10(1.0 / mse)


# ---------- dataset shard access (read only) ----------
_manifest_cache = {}


def manifest(recipe):
    if recipe not in _manifest_cache:
        _manifest_cache[recipe] = json.load(open(DATASET / recipe / "manifest.json", encoding="utf-8"))
    return _manifest_cache[recipe]


def child_sample(recipe, sample_id):
    """Return dict of arrays for one sample from a D1/D2/D3 child shard (uint8)."""
    m = manifest(recipe)
    split = None
    for s in m["samples"]:
        if s["sample_id"] == sample_id:
            split = s["split"]
            break
    for sh in m["shards"]:
        if sample_id in sh["sample_ids"]:
            sha = sh["object"]["sha256"]
            break
    else:
        raise KeyError(sample_id)
    cand = list((DATASET / recipe).rglob(f"{sha}.tar"))
    if not cand:
        raise FileNotFoundError(sha)
    with tarfile.open(cand[0]) as tf:
        for mem in tf.getmembers():
            if mem.name.endswith(f"{sample_id}.npz"):
                z = np.load(io.BytesIO(tf.extractfile(mem).read()), allow_pickle=False)
                return {k: z[k] for k in z.files}
    raise KeyError(sample_id)


def x10_sample(sample_id):
    import pandas as pd
    s = pd.read_csv(X10DATA / "samples.csv", usecols=["sample_id", "x10_split", "x10_shard", "x10_member"])
    r = s[s.sample_id == sample_id].iloc[0]
    path = X10DATA / r.x10_split / r.x10_shard
    with tarfile.open(path) as tf:
        z = np.load(io.BytesIO(tf.extractfile(r.x10_member).read()), allow_pickle=False)
        return {k: z[k] for k in z.files}

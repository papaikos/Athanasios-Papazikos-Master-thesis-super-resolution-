"""Derive every thesis table from the preserved run archives (read only).

No training, no new model inference. Only:
- reading recorded metrics, logs and per-image CSVs;
- statistics on them (bootstrap confidence intervals, groupings);
- simple diagnostics on the stored raw outputs of the 30 demonstration scenes
  and a bicubic reference for the SkySat x4 demonstration set.
Outputs: Desktop/Thesis/data/*.csv and summary.json
"""
import glob
import json

import numpy as np
import pandas as pd
from PIL import Image

from common import (DATA, DESK, JEV_THESIS, PILOT, RUN, X10DATA, area_down,
                    bicubic_up, psnr)

rng = np.random.default_rng(20261029)
OUT = {}


def wcsv(df, name):
    df.to_csv(DATA / f"{name}.csv", index=False)
    print("table", name, df.shape)


# ------------------------------------------------------------------ metadata
meta = pd.read_csv(DESK / "Jev check/thesis_data/mother_metadata.csv")
master = pd.read_csv(JEV_THESIS / "master_table.csv",
                     usecols=["sample_id", "scene", "vessels", "port_structures", "shoreline",
                              "water", "large_ship", "decision", "final_decision", "remove_reasons"])
meta = meta.merge(master, on="sample_id", how="left")

# ------------------------------------------------------------------ dataset composition
comp = meta.groupby(["split", "source_kind"]).size().unstack(fill_value=0)
comp["total"] = comp.sum(axis=1)
comp = comp.reset_index()
wcsv(comp, "dataset_split_by_source")
sites = meta.groupby("split").agg(samples=("sample_id", "size"), sites=("site_id", "nunique"),
                                  groups=("geographic_group_id", "nunique"),
                                  countries=("country", "nunique"),
                                  acquisitions=("acquisition_id", "nunique")).reset_index()
wcsv(sites, "dataset_split_sites")
gsd = meta.groupby("source_kind")["native_gsd_m"].describe()[["count", "min", "50%", "max"]].reset_index()
wcsv(gsd, "dataset_native_gsd")
dates = meta.assign(year=pd.to_datetime(meta.acquired, errors="coerce", utc=True).dt.year)
wcsv(dates.groupby(["source_kind"]).year.agg(["min", "max"]).reset_index(), "dataset_years")
scene_split = meta.groupby(["split", "scene"]).size().unstack(fill_value=0).reset_index()
wcsv(scene_split, "dataset_scene_by_split")

# ------------------------------------------------------------------ validation curves
curves = []
for r in ("D1", "D2", "D3"):
    a = pd.read_csv(RUN[r] / "reports/main-validation-available-aggregates.csv")
    a = a.rename(columns={"mean_per_image_rgb_psnr": "psnr", "mean_per_image_lpips": "lpips"})
    a = a[["step", "psnr", "ssim", "lpips"]].drop_duplicates("step").sort_values("step")
    a["run"] = r
    curves.append(a)
x = pd.read_csv(RUN["D3x10"] / "reports/tables/validation_vs_updates.csv")
x = x[x["update"] != "bicubic"].rename(columns={"update": "step", "psnr_db": "psnr"})
x["step"] = x.step.astype(int)
x = x[["step", "psnr", "ssim", "lpips"]]
x["run"] = "D3x10"
curves.append(x)
curves = pd.concat(curves, ignore_index=True)
wcsv(curves, "validation_curves")

# ------------------------------------------------------------------ per-image validation
def per_image(path, tag):
    d = pd.read_csv(path)[["sample_id", "psnr", "ssim", "lpips"]]
    return d.rename(columns={c: f"{tag}_{c}" for c in ("psnr", "ssim", "lpips")})

pi = per_image(RUN["D1"] / "reports/per-image-metrics/bicubic-main-validation-lpips-v1.csv", "D1bic")
for path, tag in [
    (RUN["D1"] / "reports/per-image-metrics/main-validation-step-84000-lpips-v1.csv", "D1"),
    (RUN["D2"] / "reports/per-image-metrics/bicubic-main-validation-lpips-v1.csv", "D2bic"),
    (RUN["D2"] / "reports/per-image-metrics/main-validation-step-90000-lpips-v1.csv", "D2"),
    (RUN["D3"] / "training-run/final/artifacts/metrics/bicubic-main-validation-lpips-v1.csv", "D3bic"),
]:
    pi = pi.merge(per_image(path, tag), on="sample_id")
x10 = pd.read_csv(RUN["D3x10"] / "reports/tables/per_image_selected_vs_bicubic.csv")
x10 = x10.rename(columns={"bicubic_psnr": "D3x10bic_psnr", "model_psnr": "D3x10_psnr",
                          "bicubic_ssim": "D3x10bic_ssim", "model_ssim": "D3x10_ssim",
                          "bicubic_lpips": "D3x10bic_lpips", "model_lpips": "D3x10_lpips"})
pi = pi.merge(x10.drop(columns=["psnr_gain"]), on="sample_id")
pi = pi.merge(meta[["sample_id", "site_id", "geographic_group_id", "country", "lat", "lon",
                    "native_gsd_m", "nearest_port", "scene", "vessels", "port_structures", "acquired"]],
              on="sample_id", how="left")
for r in ("D1", "D2", "D3x10"):
    pi[f"{r}_gain"] = pi[f"{r}_psnr"] - pi[f"{r}bic_psnr"]
    pi[f"{r}_dlpips"] = pi[f"{r}_lpips"] - pi[f"{r}bic_lpips"]
    pi[f"{r}_dssim"] = pi[f"{r}_ssim"] - pi[f"{r}bic_ssim"]
assert len(pi) == 115, len(pi)
wcsv(pi, "per_image_validation")


def boot(v, n=10000):
    v = np.asarray(v)
    idx = rng.integers(0, len(v), (n, len(v)))
    m = v[idx].mean(1)
    return np.percentile(m, [2.5, 97.5])


def cluster_boot(v, g, n=10000):
    df = pd.DataFrame({"v": v, "g": g})
    groups = [d.v.values for _, d in df.groupby("g")]
    k = len(groups)
    ms = []
    for _ in range(n):
        pick = rng.integers(0, k, k)
        allv = np.concatenate([groups[i] for i in pick])
        ms.append(allv.mean())
    return np.percentile(ms, [2.5, 97.5])


rows = []
for r in ("D1", "D2", "D3x10"):
    g = pi[f"{r}_gain"]
    lo, hi = boot(g)
    clo, chi = cluster_boot(g.values, pi.site_id.values)
    dl = pi[f"{r}_dlpips"]
    llo, lhi = boot(dl)
    rows.append(dict(run=r, n=len(g), mean_gain=g.mean(), median_gain=g.median(), ci_lo=lo, ci_hi=hi,
                     site_ci_lo=clo, site_ci_hi=chi, share_improved=(g > 0).mean(),
                     worst=g.min(), best=g.max(), mean_dlpips=dl.mean(), dlpips_ci_lo=llo, dlpips_ci_hi=lhi,
                     mean_dssim=pi[f"{r}_dssim"].mean(),
                     corr_bic_gain=np.corrcoef(pi[f"{r}bic_psnr"], g)[0, 1]))
gb = pd.DataFrame(rows)
wcsv(gb, "gain_bootstrap")

# D1 vs D2 bicubic difficulty on same HR (different LR operators)
OUT["bicubic_D1_minus_D2_mean"] = float((pi.D1bic_psnr - pi.D2bic_psnr).mean())
OUT["bicubic_D2_minus_D3_mean"] = float((pi.D2bic_psnr - pi.D3bic_psnr).mean())
OUT["D1_minus_D2_model_mean"] = float((pi.D1_psnr - pi.D2_psnr).mean())
d12 = (pi.D1_psnr - pi.D2_psnr)
OUT["D1_minus_D2_model_ci"] = list(boot(d12))
OUT["D1_minus_D2_model_share_D1_better"] = float((d12 > 0).mean())
OUT["D2_gain_minus_D1_gain_ci"] = list(boot(pi.D2_gain - pi.D1_gain))
OUT["D2_gain_minus_D1_gain_mean"] = float((pi.D2_gain - pi.D1_gain).mean())

# by scene type and by site
sc = []
for scene, d in pi.groupby("scene"):
    row = {"scene": scene, "n": len(d)}
    for r in ("D1", "D2", "D3x10"):
        row[f"{r}_bic"] = d[f"{r}bic_psnr"].mean()
        row[f"{r}_gain"] = d[f"{r}_gain"].mean()
    sc.append(row)
wcsv(pd.DataFrame(sc).sort_values("n", ascending=False), "gain_by_scene")

st = []
for site, d in pi.groupby("site_id"):
    row = {"site": site, "port": d.nearest_port.iloc[0], "country": d.country.iloc[0], "n": len(d),
           "acquired": str(d.acquired.iloc[0])[:10]}
    for r in ("D1", "D2", "D3x10"):
        row[f"{r}_bic"] = d[f"{r}bic_psnr"].mean()
        row[f"{r}_gain"] = d[f"{r}_gain"].mean()
        if len(d) > 2:
            lo, hi = boot(d[f"{r}_gain"], 4000)
        else:
            lo, hi = np.nan, np.nan
        row[f"{r}_lo"], row[f"{r}_hi"] = lo, hi
    st.append(row)
st = pd.DataFrame(st).sort_values("n", ascending=False)
wcsv(st, "gain_by_site")
OUT["site_macro_gain"] = {r: float(st[f"{r}_gain"].mean()) for r in ("D1", "D2", "D3x10")}

# ------------------------------------------------------------------ run summary table
def first_last(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    t = pd.to_datetime([r["utc"] for r in rows])
    us = [r["update_seconds"] for r in rows if r.get("update_seconds")]
    return t.min(), t.max(), float(np.median(us))

tele = {
    "D1": RUN["D1"] / "training-run/project/results/logs/telemetry.jsonl",
    "D2": RUN["D2"] / "training-run/project/results/logs/telemetry.jsonl",
    "D3": RUN["D3"] / "training-run/recovery/fc6060b1-05c9-493c-967e-e4c859975f92/logs/telemetry.jsonl",
    "D3x10": RUN["D3x10"] / "training-run/project/results/logs/telemetry.jsonl",
}
hw = {"D1": "NVIDIA A100 40 GB (AUTH Aristotle)", "D2": "NVIDIA A100 40 GB (AUTH Aristotle)",
      "D3": "NVIDIA L4 24 GB (Google Cloud)", "D3x10": "NVIDIA A100 40 GB (AUTH Aristotle)"}
summ = []
bic = {"D1": (pi.D1bic_psnr.mean(), pi.D1bic_ssim.mean(), pi.D1bic_lpips.mean()),
       "D2": (pi.D2bic_psnr.mean(), pi.D2bic_ssim.mean(), pi.D2bic_lpips.mean()),
       "D3": (pi.D3bic_psnr.mean(), pi.D3bic_ssim.mean(), pi.D3bic_lpips.mean()),
       "D3x10": (pi.D3x10bic_psnr.mean(), pi.D3x10bic_ssim.mean(), pi.D3x10bic_lpips.mean())}
best = {"D1": 84000, "D2": 90000, "D3": 82000, "D3x10": 72000}
for r in ("D1", "D2", "D3", "D3x10"):
    cv = curves[curves.run == r].set_index("step")
    t0, t1, med = first_last(tele[r])
    b = cv.loc[best[r]]
    f = cv.loc[cv.index.max()]
    init = cv.loc[0].psnr if 0 in cv.index else np.nan
    summ.append(dict(run=r, scale="×10" if r == "D3x10" else "×4", hardware=hw[r],
                     bic_psnr=bic[r][0], bic_ssim=bic[r][1], bic_lpips=bic[r][2],
                     init_psnr=init, best_step=best[r], best_psnr=b.psnr, best_ssim=b.ssim, best_lpips=b.lpips,
                     last_step=int(cv.index.max()), last_psnr=f.psnr, last_ssim=f.ssim, last_lpips=f.lpips,
                     gain=b.psnr - bic[r][0], start=str(t0)[:16], end=str(t1)[:16],
                     median_update_s=med, est_compute_h=med * 100000 / 3600))
summ = pd.DataFrame(summ)
wcsv(summ, "runs_summary")

# steps needed to beat bicubic and to reach 95% of final gain
reach = []
for r in ("D1", "D2", "D3", "D3x10"):
    cv = curves[curves.run == r].sort_values("step")
    b = bic[r][0]
    above = cv[cv.psnr > b]
    first = int(above.step.iloc[0]) if len(above) else None
    g = cv.psnr - b
    target = 0.95 * (summ.set_index("run").loc[r, "best_psnr"] - b)
    s95 = int(cv[g >= target].step.iloc[0])
    reach.append(dict(run=r, first_step_above_bicubic=first, step_95pct_gain=s95,
                      psnr_at_2000=float(cv[cv.step == 2000].psnr.iloc[0]),
                      gain_at_2000=float(cv[cv.step == 2000].psnr.iloc[0] - b),
                      gain_at_10000=float(cv[cv.step == 10000].psnr.iloc[0] - b)))
wcsv(pd.DataFrame(reach), "convergence")

# training loss curves
loss = []
for r in ("D1", "D2"):
    t = pd.read_csv(RUN[r] / "reports/training-interval-metrics.csv")
    loss.append(pd.DataFrame({"run": r, "step": t.step, "l1": t.raw_l1, "lr": t.lr}))
for r, p in (("D3", RUN["D3"] / "training-run/recovery/fc6060b1-05c9-493c-967e-e4c859975f92/logs/train.jsonl"),
             ("D3x10", RUN["D3x10"] / "training-run/project/results/logs/train.jsonl")):
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    k = [c for c in rows[0] if "l1" in c.lower() or "loss" in c.lower()]
    lk = "raw_l1" if "raw_l1" in rows[0] else k[0]
    loss.append(pd.DataFrame({"run": r, "step": [q["step"] for q in rows],
                              "l1": [q.get(lk) for q in rows], "lr": [q.get("lr") for q in rows]}))
loss = pd.concat(loss, ignore_index=True)
wcsv(loss, "training_loss")

# ------------------------------------------------------------------ aerial diagnostic
aer = pd.DataFrame([
    dict(run="D1", n=15, psnr=30.767542, ssim=0.867967, lpips=0.212850),
    dict(run="D2", n=15, psnr=30.678628, ssim=0.864491, lpips=0.221982),
    dict(run="D3", n=15, psnr=30.338067, ssim=0.856044, lpips=0.238115)])
wcsv(aer, "aerial_diagnostic")

# ------------------------------------------------------------------ SkySat x4 demonstration
def ssim_valid(a, b):
    """11x11 Gaussian (sigma 1.5) SSIM at valid window centres, data range 1, mean over RGB."""
    from scipy.signal import convolve2d
    g = np.exp(-((np.arange(11) - 5) ** 2) / (2 * 1.5 ** 2)); g /= g.sum(); w = np.outer(g, g)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    vals = []
    for c in range(3):
        x = a[c].astype(np.float64); y = b[c].astype(np.float64)
        f = lambda z: convolve2d(z, w, mode="valid")
        mx, my = f(x), f(y)
        sxx, syy, sxy = f(x * x) - mx ** 2, f(y * y) - my ** 2, f(x * y) - mx * my
        s = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (sxx + syy + C2))
        vals.append(s.mean())
    return float(np.mean(vals))

sky = []
for i in range(1, 11):
    sid = f"skysat-{i:02d}"
    base = RUN["D1"] / "skysat-comparisons/results" / sid
    ref = np.asarray(Image.open(base / "HR-reference.png"), np.float64).transpose(2, 0, 1) / 255
    lr = np.load(base / "LR-x4-fp32.npy")
    preds = {"bicubic": np.clip(bicubic_up(lr, 4), 0, 1),
             "D1": np.load(base / "best-raw-fp32.npy"),
             "D3": np.load(base / "D3-best-raw-fp32.npy"),
             "D2": np.load(RUN["D2"] / "skysat-comparisons/results" / sid / "best-raw-fp32.npy")}
    for k, p in preds.items():
        p = np.clip(p, 0, 1)
        sky.append(dict(scene=sid, model=k, psnr=psnr(p, ref), ssim=ssim_valid(p, ref),
                        hf_ratio=None))
sky = pd.DataFrame(sky)
wcsv(sky, "skysat_x4_per_image")
# check: recomputed D1/D3 PSNR must equal the archived metrics
m1 = pd.read_csv(RUN["D1"] / "skysat-comparisons/metrics.csv")
arch = m1[m1.model == "D1-best"].set_index("sample_id").psnr
mine = sky[sky.model == "D1"].set_index("scene").psnr
OUT["skysat_recompute_max_abs_psnr_diff"] = float((arch - mine).abs().max())
archs = m1[m1.model == "D1-best"].set_index("sample_id").ssim
mines = sky[sky.model == "D1"].set_index("scene").ssim
OUT["skysat_recompute_max_abs_ssim_diff"] = float((archs - mines).abs().max())
skym = sky.groupby("model")[["psnr", "ssim"]].mean().reset_index()
lp = {"D1": 0.097363, "D2": 0.194942, "D3": 0.102259}
skym["lpips"] = skym.model.map(lp)
wcsv(skym, "skysat_x4_summary")

# SkySat x10 demonstration (archived metrics, x10 set)
s10 = pd.read_csv(RUN["D3x10"] / "skysat-comparisons/metrics.csv")
wcsv(s10, "skysat_x10_per_image")

# ------------------------------------------------------------------ Sentinel-2 diagnostics
def grad_energy(chw):
    g = np.mean(np.asarray(chw, np.float64), axis=0)
    gx = np.diff(g, axis=1)[:-1, :]
    gy = np.diff(g, axis=0)[:, :-1]
    return float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))


def s2_dirs():
    out = []
    for group in ("sentinel2-comparisons", "sentinel2-additional-ports"):
        base = RUN["D1"] / group / "results"
        for d in sorted(p.name for p in base.iterdir() if p.is_dir()):
            out.append((group, d))
    return out

diag = []
for group, scene in s2_dirs():
    lr = np.load(RUN["D1"] / group / "results" / scene / "LR-input-fp32.npy")
    outs = {"bicubic": bicubic_up(lr, 4)}
    for r in ("D1", "D2", "D3"):
        f = "D3-best-raw-fp32.npy" if r == "D3" else "best-raw-fp32.npy"
        src = RUN["D1"] if r == "D3" else RUN[r]
        outs[r] = np.load(src / group / "results" / scene / f)
    x10 = np.load(RUN["D3x10"] / group / "results" / scene / "best-raw-fp32.npy")
    pz = PILOT / "examples" / scene / "arrays.npz"
    if pz.exists():
        z = np.load(pz)
        assert np.abs(z["lr"] - lr).max() == 0
        outs["SEN2SR-FT"] = z["sen2sr"]
        outs["SwinIR-FT"] = z["swinir"]
    ge_bic = grad_energy(np.clip(outs["bicubic"], 0, 1))
    for k, o in outs.items():
        oc = np.clip(o, 0, 1)
        diag.append(dict(group=group, scene=scene, model=k,
                         out_of_range=float(((o < 0) | (o > 1)).mean()),
                         lr_consistency_psnr=psnr(area_down(oc, 4), lr),
                         grad_ratio=grad_energy(oc) / ge_bic))
    oc = np.clip(x10, 0, 1)
    ge10 = grad_energy(np.clip(bicubic_up(lr, 10), 0, 1))
    diag.append(dict(group=group, scene=scene, model="D3x10",
                     out_of_range=float(((x10 < 0) | (x10 > 1)).mean()),
                     lr_consistency_psnr=psnr(area_down(oc, 10), lr),
                     grad_ratio=grad_energy(oc) / ge10))
diag = pd.DataFrame(diag)
wcsv(diag, "s2_diagnostics_per_scene")
ds = diag.groupby("model").agg(n=("scene", "size"), out_of_range=("out_of_range", "mean"),
                               lr_consistency_psnr=("lr_consistency_psnr", "mean"),
                               grad_ratio=("grad_ratio", "mean")).reset_index()
wcsv(ds, "s2_diagnostics_summary")

# ------------------------------------------------------------------ D3x10 dataset draws
s = pd.read_csv(X10DATA / "samples.csv", usecols=lambda c: c.startswith("x10_") or c in
                ("sample_id", "source_kind", "native_gsd_ge_1m", "restored_by_manual_review"))
OUT["x10_samples_by_split"] = s.x10_split.value_counts().to_dict()
OUT["x10_native_ge_1m_train"] = int(s[(s.x10_split == "train")].native_gsd_ge_1m.sum())
OUT["x10_kernel_type"] = s.x10_kernel_type.value_counts().to_dict()
OUT["x10_lr_vs_d3_psnr_min"] = float(s.x10_lr_vs_d3_psnr_db.min())
OUT["x10_lr_vs_d3_maxabs_max"] = int(s.x10_lr_vs_d3_max_abs_dn.max())
OUT["x10_noise_zero_share"] = float((s.x10_noise_sigma_dn == 0).mean())
wcsv(s[["x10_kernel_type", "x10_sigma_x_m", "x10_sigma_y_m", "x10_angle_radians", "x10_noise_sigma_dn",
        "x10_split", "x10_lr_vs_d3_psnr_db"]], "x10_degradation_draws")
OUT["x10_by_source_split"] = s.groupby(["x10_split", "source_kind"]).size().unstack(fill_value=0).to_dict()

json.dump(OUT, open(DATA / "summary.json", "w", encoding="utf-8"), indent=1, default=float)
print(json.dumps(OUT, indent=1, default=float))

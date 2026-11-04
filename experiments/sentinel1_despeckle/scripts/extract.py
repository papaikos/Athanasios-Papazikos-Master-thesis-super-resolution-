"""Extract Sentinel-1 GRD amplitude chips containing ships from Microsoft
Planetary Computer. For each busy maritime location we search S1 GRD (IW),
read a large VV window near the anchorage, then auto-select the sub-crop
richest in compact bright blobs over dark water (a simple ship detector)."""
import os, json, numpy as np
import planetary_computer, pystac_client
import rasterio
from rasterio.windows import Window
from rasterio.transform import GCPTransformer
from scipy import ndimage as ndi
from PIL import Image

CHIP = 512          # output chip size (px)
BIGWIN = 2048       # search window size (px)
STEP = 256          # slide step when scoring crops
OUTDIR = os.path.join(os.path.dirname(__file__), "..", "data", "chips")
os.makedirs(OUTDIR, exist_ok=True)

# Busy anchorages / shipping choke points: name, lon, lat
LOCATIONS = [
    ("singapore_east",     104.02,   1.20),
    ("singapore_strait",   103.90,   1.19),
    ("gibraltar_strait",    -5.45,  36.02),
    ("fujairah_anchorage",  56.46,  25.20),
    ("suez_gulf",           32.55,  29.85),
    ("port_said_suez",      32.33,  31.32),
    ("piraeus_saronic",     23.55,  37.88),
    ("hong_kong_anchorage",114.12,  22.22),
    ("panama_pacific",     -79.55,   8.78),
    ("istanbul_marmara",    28.70,  40.72),
    ("longbeach_la",      -118.22,  33.70),
    ("dover_channel",        1.48,  51.05),
    ("port_klang_malacca", 100.90,   2.95),
    ("bosphorus_north",     29.10,  41.22),
    ("jebel_ali_dubai",     55.02,  24.96),
    ("chittagong_anchor",   91.78,  22.18),
]

cat = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

def score_crop(crop):
    """Return (score, water_med). We want boats ON open water: a dark, water-
    dominant background with a handful of isolated compact bright targets."""
    if crop.size == 0 or not np.isfinite(crop).all():
        return -1, 0
    water_med = float(np.median(crop))
    p20 = float(np.percentile(crop, 20))
    if water_med <= 0 or p20 <= 0:
        return -1, water_med
    # fraction of pixels that look like homogeneous water (near the dark level)
    dark_frac = float(np.mean(crop < 2.5 * p20))
    if dark_frac < 0.55:                  # not water-dominant => land/urban, reject
        return -1, water_med
    # bright targets: well above the water background
    thr = max(water_med * 8.0, float(np.percentile(crop, 99.5)))
    mask = crop > thr
    frac = float(mask.mean())
    if frac == 0 or frac > 0.03:          # nothing, or too much bright (land)
        return -1, water_med
    lbl, n = ndi.label(mask)
    if n == 0:
        return -1, water_med
    sizes = ndi.sum(np.ones_like(lbl), lbl, index=np.arange(1, n + 1))
    ships = int(np.sum((sizes >= 1) & (sizes <= 200)))   # compact vessel-sized
    if ships < 2:
        return -1, water_med
    # reward ships, reward darker/cleaner water (better speckle demo)
    score = ships + dark_frac
    return score, water_med

def to_preview(amp):
    """Amplitude -> 8-bit dB-ish preview with percentile stretch."""
    a = amp.astype(np.float32)
    a = np.where(np.isfinite(a) & (a > 0), a, np.nan)
    db = 10.0 * np.log10(a + 1e-3)
    lo, hi = np.nanpercentile(db, 2), np.nanpercentile(db, 98)
    db = np.clip((db - lo) / max(hi - lo, 1e-6), 0, 1)
    db = np.nan_to_num(db, nan=0.0)
    return (db * 255).astype(np.uint8)

manifest = []
for name, lon, lat in LOCATIONS:
    try:
        search = cat.search(
            collections=["sentinel-1-grd"],
            intersects={"type": "Point", "coordinates": [lon, lat]},
            datetime="2023-06-01/2025-03-31",
            query={"sar:instrument_mode": {"eq": "IW"}},
            max_items=12,
        )
        items = list(search.items())
        if not items:
            print(f"[{name}] no items"); continue
        got = False
        for it in items:
            band = "vv" if "vv" in it.assets else ("hh" if "hh" in it.assets else None)
            if band is None:
                continue
            href = it.assets[band].href
            try:
                with rasterio.open(href) as ds:
                    gcps, _ = ds.gcps
                    with GCPTransformer(gcps) as tf:
                        row, col = tf.rowcol(lon, lat)
                    if not (0 <= row < ds.height and 0 <= col < ds.width):
                        continue
                    r0 = int(np.clip(row - BIGWIN // 2, 0, ds.height - BIGWIN))
                    c0 = int(np.clip(col - BIGWIN // 2, 0, ds.width - BIGWIN))
                    win = Window(c0, r0, BIGWIN, BIGWIN)
                    big = ds.read(1, window=win).astype(np.float32)
            except Exception as e:
                print(f"[{name}] read fail {it.id}: {e}"); continue
            # reject windows dominated by nodata
            if (big <= 0).mean() > 0.4:
                continue
            # slide to find the most ship-rich CHIP crop
            best = (-1, None, None)
            for rr in range(0, BIGWIN - CHIP + 1, STEP):
                for cc in range(0, BIGWIN - CHIP + 1, STEP):
                    crop = big[rr:rr + CHIP, cc:cc + CHIP]
                    if (crop <= 0).mean() > 0.02:
                        continue
                    s, wmed = score_crop(crop)
                    if s > best[0]:
                        best = (s, (rr, cc), crop.copy())
            if best[0] >= 2.5 and best[2] is not None:   # >= 2 vessels on water
                s, (rr, cc), crop = best
                np.save(os.path.join(OUTDIR, f"{name}.npy"), crop.astype(np.float32))
                Image.fromarray(to_preview(crop)).save(
                    os.path.join(OUTDIR, f"{name}_preview.png"))
                manifest.append({
                    "name": name, "n_ships": int(s), "scene": it.id, "band": band,
                    "datetime": str(it.datetime), "lon": lon, "lat": lat,
                    "crop_offset": [int(rr), int(cc)],
                })
                print(f"[{name}] OK ships~{s} scene={it.id}")
                got = True
                break
        if not got:
            print(f"[{name}] no ship-rich crop found")
    except Exception as e:
        print(f"[{name}] ERROR {e}")

with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print(f"\nTOTAL chips: {len(manifest)}")

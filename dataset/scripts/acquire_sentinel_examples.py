"""Download small real Sentinel-2 maritime inference patches; no export jobs/VMs."""

import hashlib
import io
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
from urllib.request import urlopen
from src.cloud_config import GCP_PROJECT

# Reuse authenticated project's Earth Engine environment alongside local NumPy.
sys.path.append(
    str(Path(__file__).resolve().parents[1] / ".venv-satellite/Lib/site-packages")
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageDraw
from scripts import skysat_acquire

SITES = [
    ("piraeus", "Piraeus ferry port", 23.6380, 37.9465),
    ("zea", "Zea Marina, Greece", 23.6477, 37.9362),
    ("flisvos", "Flisvos Marina, Greece", 23.6821, 37.9312),
    ("barcelona", "Port Vell, Barcelona", 2.1856, 41.3737),
    ("monaco", "Port Hercule, Monaco", 7.4275, 43.7349),
    ("marseille", "Vieux Port, Marseille", 5.3668, 43.2948),
    ("valencia", "Valencia marina", -0.3236, 39.4580),
    ("palma", "Palma harbour, Mallorca", 2.6374, 39.5609),
    ("genoa", "Genoa old harbour", 8.9230, 44.4110),
    ("livorno", "Livorno port", 10.2954, 43.5555),
    ("naples", "Naples ferry port", 14.2596, 40.8359),
    ("split", "Split harbour, Croatia", 16.4427, 43.5013),
    ("dubrovnik", "Gruz harbour, Dubrovnik", 18.0863, 42.6593),
    ("malta", "Grand Harbour, Malta", 14.5150, 35.8906),
    ("limassol", "Limassol Marina, Cyprus", 33.0412, 34.6693),
]
OUT = Path("outputs/sentinel-examples")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    skysat_acquire.GCLOUD = (
        "C:/Program Files (x86)/Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd"
    )
    ee = skysat_acquire.initialize(GCP_PROJECT, True)
    rows = []
    for key, title, lon, lat in SITES:
        saved = OUT / f"{key}.json"
        if saved.exists():
            rows.append(json.loads(saved.read_text()))
            continue
        point = ee.Geometry.Point([lon, lat])
        coll = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(point)
            .filterDate("2025-05-01", "2025-10-01")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 5))
            .sort("CLOUDY_PIXEL_PERCENTAGE")
            .limit(12)
        )
        ids = coll.aggregate_array("system:id").getInfo()
        for scene in ids:
            im = ee.Image(scene)
            projection = im.select("B4").projection().getInfo()
            crs = projection["crs"]
            x, y = point.transform(crs, 1).coordinates().getInfo()
            left = np.floor(x / 10) * 10 - 640
            top = np.ceil(y / 10) * 10 + 640
            transform = [10, 0, float(left), 0, -10, float(top)]
            bands = ["TCI_R", "TCI_G", "TCI_B", "B4", "B3", "B2", "SCL"]
            valid = (
                im.select(["B4", "B3", "B2"])
                .mask()
                .reduce(ee.Reducer.min())
                .gt(0)
                .rename("valid")
            )
            image = im.select(bands).addBands(valid).unmask(0)
            url = image.getDownloadURL(
                {
                    "format": "NPY",
                    "crs": crs,
                    "crs_transform": transform,
                    "dimensions": [128, 128],
                }
            )
            with urlopen(url, timeout=120) as response:
                raw = response.read()
            arr = np.load(io.BytesIO(raw), allow_pickle=False)
            if arr.shape != (128, 128):
                raise ValueError(arr.shape)
            cloud = np.isin(arr["SCL"], [0, 1, 3, 8, 9, 10, 11])
            if cloud.any() or np.mean(arr["valid"] > 0) < 0.999:
                print(
                    json.dumps(
                        {
                            "rejected": key,
                            "scene": scene,
                            "bad_pixels": int(cloud.sum()),
                            "valid_stats": np.unique(arr["valid"], return_counts=True)[
                                0
                            ].tolist(),
                            "tci_range": [
                                int(arr["TCI_R"].min()),
                                int(arr["TCI_R"].max()),
                            ],
                        }
                    ),
                    flush=True,
                )
                continue
            tci = np.stack([arr[b] for b in bands[:3]]).astype(np.uint8)
            lr = tci.astype(np.float32) / np.float32(255)
            reflectance = np.stack([arr[b] for b in bands[3:6]]).astype(
                np.float32
            ) / np.float32(10000)
            prov = {
                "collection": "COPERNICUS/S2_SR_HARMONIZED",
                "scene_id": scene,
                "date": scene.split("/")[-1][:8],
                "longitude": lon,
                "latitude": lat,
                "crs": crs,
                "transform": transform,
                "native_pixel_m": 10,
                "extent_m": [1280, 1280],
                "rgb_bands": bands[:3],
                "normalization": "Official Sentinel-2 uint8 true-color TCI_R/G/B divided by 255; no per-image stretch or enhancement.",
                "reflectance_bands": bands[3:6],
                "reflectance_scale": 0.0001,
                "invalid_cloud_shadow_snow_pixels": 0,
                "valid_fraction": float(np.mean(arr["valid"] > 0)),
                "screening": "No SCL cloud/shadow/snow/defective/nodata classes; at least 99.9% spectral validity. Original pixels retained without inpainting.",
                "ground_truth_available": False,
                "purpose": "inference illustration only; not training acquisition",
                "catalog": "https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED",
            }
            path = OUT / f"{key}.npz"
            np.savez_compressed(
                path,
                lr=lr,
                reflectance=reflectance,
                scl=arr["SCL"],
                provenance=np.array(json.dumps(prov)),
            )
            Image.fromarray(tci.transpose(1, 2, 0)).save(OUT / f"{key}-original.png")
            row = {
                "id": key,
                "title": title,
                "path": path.name,
                "provenance": prov,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            row["download_url"] = url
            row["download_sha256"] = hashlib.sha256(raw).hexdigest()
            saved.write_text(json.dumps(row, indent=2), encoding="utf-8")
            rows.append(row)
            print(
                json.dumps({"accepted": key, "count": len(rows), "scene": scene}),
                flush=True,
            )
            break
        else:
            raise RuntimeError(f"No clear valid candidate for {key}")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "examples": rows,
        "count": len(rows),
        "model_input": "lr float32 CHW RGB, official TCI / 255",
        "no_matching_hr_reference": True,
    }
    (OUT / "cloud-downloads.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    public = {
        **manifest,
        "examples": [
            {k: v for k, v in row.items() if k != "download_url"} for row in rows
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(public, indent=2), encoding="utf-8")
    canvas = Image.new("RGB", (5 * 276, 3 * 300), (15, 23, 35))
    draw = ImageDraw.Draw(canvas)
    for i, r in enumerate(rows):
        x = (i % 5) * 276
        y = (i // 5) * 300
        im = Image.open(OUT / f"{r['id']}-original.png").resize(
            (256, 256), Image.Resampling.NEAREST
        )
        canvas.paste(im, (x + 10, y + 30))
        draw.text((x + 10, y + 7), r["title"], fill="white")
    canvas.save(OUT / "originals-contact-sheet.png")
    print(json.dumps({"complete": len(rows)}), flush=True)


if __name__ == "__main__":
    main()

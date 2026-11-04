import planetary_computer, pystac_client, numpy as np, rasterio
from rasterio.transform import GCPTransformer
from rasterio.windows import Window
cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                modifier=planetary_computer.sign_inplace)
lon, lat = -5.40, 36.08
for it in cat.search(collections=["sentinel-1-grd"],
        intersects={"type":"Point","coordinates":[lon,lat]},
        datetime="2024-06-01/2025-03-31",
        query={"sar:instrument_mode":{"eq":"IW"}}, max_items=3).items():
    band = "vv" if "vv" in it.assets else ("hh" if "hh" in it.assets else None)
    print("scene", it.id, "band", band)
    if not band: continue
    with rasterio.open(it.assets[band].href) as ds:
        gcps, gcp_crs = ds.gcps
        with GCPTransformer(gcps) as t:
            row, col = t.rowcol(lon, lat)
        print("  size", ds.width, ds.height, "row,col", int(row), int(col),
              "inb", 0<=row<ds.height and 0<=col<ds.width)
        r0=int(np.clip(row-256,0,ds.height-512)); c0=int(np.clip(col-256,0,ds.width-512))
        a = ds.read(1, window=Window(c0,r0,512,512)).astype(np.float32)
        print("  win min/med/max", float(a.min()), float(np.median(a)), float(a.max()),
              "nodata", round((a<=0).mean(),3))
    break

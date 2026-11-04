import planetary_computer, pystac_client
cat = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)
# Gibraltar Strait bbox (lon/lat)
bbox = [-5.6, 35.85, -5.3, 36.05]
search = cat.search(
    collections=["sentinel-1-grd"],
    bbox=bbox,
    datetime="2024-01-01/2024-12-31",
    query={"sar:instrument_mode": {"eq": "IW"}},
    max_items=5,
)
items = list(search.items())
print("found", len(items), "items")
for it in items[:5]:
    print(it.id, it.datetime, "assets:", [k for k in it.assets.keys()])
if items:
    it = items[0]
    print("VV href sample:", it.assets.get("vv").href[:120] if "vv" in it.assets else "NO VV")

# Sentinel-1 despeckling (appendix)

Two despecklers on Sentinel-1 SAR chips of busy straits and anchorages, such as Piraeus/Saronic, Singapore Strait, Gibraltar, Suez, Dover, Bosphorus and Port Klang:
- **SAR2SAR**: deep, self-supervised, pretrained weights from `deepdespeckling`.
- **SAR-BM3D**: homomorphic BM3D.

Each output is scored by the ENL (equivalent number of looks) on the flattest water patch.

```bash
pip install -r requirements.txt
python scripts/extract.py          # Sentinel-1 GRD chips with ships, from Microsoft Planetary Computer -> data/chips/
python scripts/run_despeckle.py    # both models -> out/sar2sar/, out/merlin/ (SAR-BM3D), out/orig/, out/metrics.json
python scripts/verify_grid.py      # side-by-side check grid
```
`scripts/ddespeckle_stub.py` stubs `osgeo` and patches `torch.load`, so `deepdespeckling` runs without GDAL. The data and outputs are not in git.

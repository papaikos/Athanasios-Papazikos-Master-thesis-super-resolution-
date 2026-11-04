# Weights v1.0: SwinIR models of the maritime super-resolution thesis

Eight PyTorch `state_dict` files: the best (highest validation PSNR) and the final (100,000 updates) checkpoint of each run. Every model is classical SwinIR, trained from scratch on the maritime dataset.

| File | Scale | LR recipe | Step | Validation PSNR (bicubic) |
|---|---|---|---|---|
| `swinir_x4_D1_best_step84000.pth` | ×4 (10 m → 2.5 m) | D1 bicubic AA | 84,000 | 32.21 dB (31.39) |
| `swinir_x4_D1_final_step100000.pth` | ×4 | D1 | 100,000 | 32.20 dB |
| `swinir_x4_D2_best_step90000.pth` | ×4 | D2 Gaussian + area | 90,000 | 32.10 dB (30.68) |
| `swinir_x4_D2_final_step100000.pth` | ×4 | D2 | 100,000 | 32.09 dB |
| `swinir_x4_D3_best_step82000.pth` | ×4 | D3 variable PSF + noise | 82,000 | 31.77 dB (30.39) |
| `swinir_x4_D3_final_step100000.pth` | ×4 | D3 | 100,000 | 31.76 dB |
| `swinir_x10_D3x10_best_step72000.pth` | ×10 (10 m → 1 m) | D3 at 1 m | 72,000 | 28.11 dB (27.32) |
| `swinir_x10_D3x10_final_step100000.pth` | ×10 | D3 at 1 m | 100,000 | 28.08 dB |

Validation is 115 Maxar images from 6 port sites. Check the files with `sha256sum -c SHA256SUMS`.

Use them with `evaluation/predict.py`:

```bash
python evaluation/predict.py --weights swinir_x4_D3_best_step82000.pth --scale 4 --input s2.png --output sr.png
```

The models were trained on imagery whose own licences still apply (for example Maxar Open Data, CC BY-NC 4.0). Use them for research.

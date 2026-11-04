"""Model 2: SAR-BM3D style homomorphic BM3D despeckling.
Multiplicative speckle -> log domain (additive) -> BM3D -> exp back.
Operates on single-channel SAR amplitude (numpy float array)."""
import numpy as np
import bm3d

def _est_log_sigma(z):
    # Donoho-style MAD noise estimate on finest diagonal differences.
    # For iid noise Var(second-difference)=4*sigma^2, so divide by 2.
    dd = z[1:, 1:] - z[1:, :-1] - z[:-1, 1:] + z[:-1, :-1]
    sigma = 1.4826 * np.median(np.abs(dd - np.median(dd))) / 2.0
    return float(np.clip(sigma, 0.08, 0.8))

def sar_bm3d(amp, sigma_scale=2.4):
    """Homomorphic BM3D: work directly in log-amplitude (additive speckle).
    Sentinel-1 GRD speckle is spatially correlated, so the per-pixel MAD sigma
    is scaled up to clean it effectively; capped to avoid block artifacts."""
    a = np.clip(np.asarray(amp, dtype=np.float64), 1e-3, None)
    z = np.log(a)                                   # log-amplitude
    sigma = float(np.clip(_est_log_sigma(z) * sigma_scale, 0.18, 0.30))
    den = bm3d.bm3d(z, sigma_psd=sigma)             # additive denoising
    return np.exp(den).astype(np.float32)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    clean = np.ones((128, 128)) * 100
    clean[40:80, 40:80] = 400
    # multiplicative exponential speckle (1-look intensity -> amplitude sqrt)
    speck = clean * rng.gamma(shape=4.0, scale=1/4.0, size=clean.shape)
    out = sar_bm3d(speck)
    def enl(x, r=(5,40,5,40)):
        p = x[r[0]:r[1], r[2]:r[3]]; return (p.mean()/p.std())**2
    print("ENL noisy:", round(enl(speck),2), "-> denoised:", round(enl(out),2))
    print("mean preserved:", round(speck.mean(),1), "->", round(out.mean(),1))

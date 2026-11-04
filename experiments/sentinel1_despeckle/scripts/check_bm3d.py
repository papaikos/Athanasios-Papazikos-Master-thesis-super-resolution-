import numpy as np
from sar_bm3d import sar_bm3d, _est_log_sigma
from scipy.ndimage import uniform_filter
def flat(amp,k=48):
    a=amp.astype(np.float64); m=uniform_filter(a,k); m2=uniform_filter(a*a,k)
    cv=np.sqrt(np.clip(m2-m*m,0,None))/(m+1e-6); cv[m<np.percentile(m,20)]=np.inf
    h=k//2; cv[:h]=cv[-h:]=np.inf; cv[:,:h]=cv[:,-h:]=np.inf
    r,c=np.unravel_index(np.argmin(cv),cv.shape); return r-h,c-h,k
def enl(a,b): r,c,k=b; p=a[r:r+k,c:c+k].astype(float); return (p.mean()/(p.std()+1e-9))**2
for nm in ["dover_channel","chittagong_anchor","istanbul_marmara"]:
    amp=np.load(f"../data/chips/{nm}.npy").astype(np.float32)
    z=np.log(np.clip(amp,1e-3,None)); print(nm,"sigma~",round(_est_log_sigma(z),3))
    out=sar_bm3d(amp); b=flat(amp)
    print("   ENL orig=%.1f  bm3d=%.1f  mean %.1f->%.1f"%(enl(amp,b),enl(out,b),amp.mean(),out.mean()))

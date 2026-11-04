import numpy as np, bm3d
from scipy.ndimage import uniform_filter
from PIL import Image
def flat(amp,k=48):
    a=amp.astype(np.float64); m=uniform_filter(a,k); m2=uniform_filter(a*a,k)
    cv=np.sqrt(np.clip(m2-m*m,0,None))/(m+1e-6); cv[m<np.percentile(m,20)]=np.inf
    h=k//2; cv[:h]=cv[-h:]=np.inf; cv[:,:h]=cv[:,-h:]=np.inf
    r,c=np.unravel_index(np.argmin(cv),cv.shape); return r-h,c-h,k
def enl(a,b): r,c,k=b; p=a[r:r+k,c:c+k].astype(float); return (p.mean()/(p.std()+1e-9))**2
def db(amp,lo,hi):
    d=10*np.log10(np.clip(amp,1e-3,None)); return np.clip((d-lo)/(hi-lo),0,1)
amp=np.load("../data/chips/dover_channel.npy").astype(np.float32)
z=np.log(np.clip(amp,1e-3,None)); b=flat(amp)
d=10*np.log10(np.clip(amp,1e-3,None)); lo,hi=np.percentile(d,2),np.percentile(d,98)
tiles=[( "orig",(db(amp,lo,hi)*255).astype(np.uint8))]
print("orig ENL %.1f"%enl(amp,b))
for s in [0.15,0.25,0.35,0.5]:
    out=np.exp(bm3d.bm3d(z,sigma_psd=s)).astype(np.float32)
    print("sigma %.2f -> ENL %.1f  mean %.1f->%.1f"%(s,enl(out,b),amp.mean(),out.mean()))
    tiles.append((f"s{s}",(db(out,lo,hi)*255).astype(np.uint8)))
# side-by-side strip
strip=np.concatenate([t[1] for t in tiles],axis=1)
Image.fromarray(strip).save("../data/chips/_bm3d_sweep.png")
print("saved strip", [t[0] for t in tiles])

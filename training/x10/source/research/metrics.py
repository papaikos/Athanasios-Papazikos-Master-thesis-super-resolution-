"""RGB float64 metrics v1: valid Gaussian SSIM, population covariance."""
import math
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import convolve1d, uniform_filter

def score_image(prediction, target, mask):
    p=np.asarray(prediction,dtype=np.float64); t=np.asarray(target,dtype=np.float64); m=np.asarray(mask)
    if p.shape!=t.shape or p.ndim!=3 or p.shape[0]!=3 or m.shape!=p.shape[1:] or m.dtype!=bool or not m.any():
        raise ValueError('Matching CHW RGB and nonempty boolean support required')
    if not np.isfinite(p).all() or not np.isfinite(t).all(): raise ValueError('Nonfinite reconstruction')
    result={'prediction_min':float(p.min()),'prediction_max':float(p.max()),'out_of_range_fraction':float(np.mean((p<0)|(p>1))),'nonfinite_count':0}
    p=np.clip(p,0,1); d=(p-t)[:,m]; mse=float(np.mean(d*d)); l1=float(np.mean(np.abs(d)))
    g=np.exp(-np.arange(-5,6,dtype=np.float64)**2/(2*1.5**2));g/=g.sum();kernel=np.outer(g,g)
    centers=uniform_filter(m.astype(float),size=11)[5:-5,5:-5]>1-1e-12
    if not centers.any():raise ValueError('No full-support SSIM window')
    def conv(a):return convolve1d(convolve1d(a,g,axis=0),g,axis=1)[5:-5,5:-5]
    ss=[]
    for x,y in zip(p,t):
        ux,uy=conv(x),conv(y); vx=conv(x*x)-ux*ux;vy=conv(y*y)-uy*uy;cov=conv(x*y)-ux*uy
        value=((2*ux*uy+.01**2)*(2*cov+.03**2))/((ux*ux+uy*uy+.01**2)*(vx+vy+.03**2))
        ss.append(float(np.mean(value[centers])))
    return dict(result,l1=l1,mse=mse,psnr=None if mse==0 else -10*math.log10(mse),psnr_infinite=mse==0,ssim=float(np.mean(ss)))

def aggregate(rows):
    if not rows:raise ValueError('Empty evaluation')
    return {'count':len(rows),'mean_per_image_rgb_psnr':None if any(r['psnr_infinite'] for r in rows) else float(np.mean([r['psnr'] for r in rows])),
            'psnr_infinite':any(r['psnr_infinite'] for r in rows),**{k:float(np.mean([r[k] for r in rows])) for k in ('ssim','l1','mse')}}

import numpy as np
from PIL import Image, ImageDraw
scenes=["dover_channel","chittagong_anchor","port_klang_malacca","panama_pacific"]
cols=["orig","sar2sar","merlin"]; lbl=["Original","SAR2SAR","SAR-BM3D"]
cell=340; pad=6; head=22
W=3*cell+4*pad; H=len(scenes)*cell+(len(scenes)+1)*pad+head
sheet=Image.new("RGB",(W,H),(15,15,18)); dr=ImageDraw.Draw(sheet)
for j,l in enumerate(lbl):
    dr.text((pad+j*(cell+pad)+cell//2-24,4),l,fill=(240,240,150))
for i,s in enumerate(scenes):
    for j,c in enumerate(cols):
        im=Image.open(f"out/{c}/{s}.png").convert("L").resize((cell,cell))
        x=pad+j*(cell+pad); y=head+pad+i*(cell+pad)
        sheet.paste(im.convert("RGB"),(x,y))
    dr.text((pad+2,head+pad+i*(cell+pad)+2),s,fill=(120,220,255))
sheet.save("out/_verify_grid.png"); print("saved",W,H)

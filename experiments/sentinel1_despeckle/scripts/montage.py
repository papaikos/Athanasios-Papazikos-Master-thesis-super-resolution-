import os, glob, json, numpy as np
from PIL import Image, ImageDraw
CH = "data/chips"
files = sorted(glob.glob(os.path.join(CH, "*_preview.png")))
man = {m["name"]: m for m in json.load(open(os.path.join(CH,"manifest.json")))}
cols, cell, pad = 5, 300, 26
rows = (len(files)+cols-1)//cols
W, H = cols*(cell+pad)+pad, rows*(cell+pad+18)+pad
sheet = Image.new("RGB",(W,H),(20,20,24)); dr=ImageDraw.Draw(sheet)
for i,f in enumerate(files):
    name=os.path.basename(f).replace("_preview.png","")
    im=Image.open(f).convert("L").resize((cell,cell))
    r,c=divmod(i,cols); x=pad+c*(cell+pad); y=pad+r*(cell+pad+18)
    sheet.paste(im.convert("RGB"),(x,y))
    n=man.get(name,{}).get("n_ships","?")
    dr.text((x,y+cell+3),f"{i}: {name} (blobs~{n})",fill=(230,230,120))
sheet.save("data/chips/_contact_sheet.png")
print("saved", W, H, "n", len(files))

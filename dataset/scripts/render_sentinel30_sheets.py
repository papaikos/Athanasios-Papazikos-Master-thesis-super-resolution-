"""Render five faithful comparison sheets from thirty verified inference records."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def font(size, bold=False):
    name = 'arialbd.ttf' if bold else 'arial.ttf'
    return ImageFont.truetype('C:/Windows/Fonts/' + name, size)


def render(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    rows = manifest['examples']
    assert len(rows) == 30 and len({r['id'] for r in rows}) == 30
    panel, margin, gap = 512, 28, 16
    width = margin * 2 + panel * 3 + gap * 2
    height = 130 + 6 * 572 + 52
    paths=[]
    for page in range(5):
        sheet=Image.new('RGB',(width,height),'#101722')
        d=ImageDraw.Draw(sheet)
        d.text((margin,20),f'Sentinel-2 maritime RGB | Comparison {page+1} / 5',font=font(32,True),fill='white')
        d.text((margin,63),'Same input and colour scale | Fine-tuned models | 4x super-resolution',font=font(22),fill='#b6c2d0')
        for col,label in enumerate(['Original Sentinel-2 | 10 m','Model 1: SEN2SR | 2.5 m grid','Model 2: SwinIR | 2.5 m grid']):
            d.text((margin+col*(panel+gap),103),label,font=font(23,True),fill='#8edbd2' if col else '#ffffff')
        for index,row in enumerate(rows[page*6:(page+1)*6]):
            y=148+index*572
            date=row.get('provenance',{}).get('date','')
            if len(date)==8: date=f'{date[:4]}-{date[4:6]}-{date[6:]}'
            title=f"{page*6+index+1:02d}  {row['title']}  |  {date}"
            d.text((margin,y),title,font=font(24,True),fill='white')
            for col,key in enumerate(['original','sen2sr','swinir']):
                image=Image.open(root/row['images'][key]).convert('RGB')
                assert image.size == ((128,128) if key=='original' else (512,512)), (key,image.size)
                if key=='original': image=image.resize((panel,panel),Image.Resampling.NEAREST)
                sheet.paste(image,(margin+col*(panel+gap),y+38))
        d.text((margin,height-35),'Original enlarged with nearest-neighbour for display. SR detail is inferred; no matching 2.5 m ground truth.',font=font(21),fill='#b6c2d0')
        path=root/f'comparison-{page+1:02d}.png';sheet.save(path);paths.append(str(path))
    return paths

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    print(json.dumps(render(p.parse_args().root),indent=2))

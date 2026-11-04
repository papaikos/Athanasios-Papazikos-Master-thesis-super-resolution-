"""Arrange already-downloaded small catalog previews for geographic review."""
import json
from pathlib import Path
from PIL import Image, ImageDraw


def main():
    root = Path('reports/maritime-build-20261006/review')
    rows = json.loads((root/'catalog-rows.json').read_text())
    rows.sort(key=lambda r: (-r['priority'], r['group_id']))
    index = []
    for start in range(0, len(rows), 12):
        canvas = Image.new('RGB',(1600,1320),'white')
        draw = ImageDraw.Draw(canvas)
        batch = rows[start:start+12]
        for i,row in enumerate(batch):
            x,y = (i%4)*400,(i//4)*440
            path = root/'catalog'/(row['group_id']+'.png')
            if path.exists():
                img = Image.open(path).convert('RGB')
                img.thumbnail((400,400))
                canvas.paste(img,(x,y+40))
            countries = ','.join(row.get('screening',{}).get('countries',[]))[:48]
            draw.text((x+5,y+3),f'{start+i}: {row["group_id"]}',fill='black')
            draw.text((x+5,y+20),countries,fill='black')
        name = f'catalog-sheet-{start//12:02d}.png'
        canvas.save(root/name)
        index.append(dict(sheet=name, groups=[dict(position=i,group_id=r['group_id'],bbox=r['bbox'],
            priority=r['priority'], representative_asset_id=r['representative_asset_id']) for i,r in enumerate(batch)]))
    (root/'catalog-sheet-index.json').write_text(json.dumps(index,indent=2))
    print(json.dumps(dict(groups=len(rows),sheets=len(index))))


if __name__ == '__main__':
    main()

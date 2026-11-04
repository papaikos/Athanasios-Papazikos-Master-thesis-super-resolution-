"""Arrange pinned native previews for triage; sheets do not confirm vessels."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',required=True)
    parser.add_argument('--directory',required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('--asset-contains',default='')
    args=parser.parse_args()
    root=Path(args.directory)
    rows=[row for row in json.loads(Path(args.inventory).read_text())['candidates']
          if args.asset_contains in row['asset_id']]
    index=[]
    for start in range(0,len(rows),12):
        canvas=Image.new('RGB',(1600,1320),'white'); draw=ImageDraw.Draw(canvas)
        batch=rows[start:start+12]
        for i,row in enumerate(batch):
            x,y=i%4*400,i//4*440
            with Image.open(root/(row['candidate_id']+'.png')) as preview:
                preview=preview.convert('RGB'); preview.thumbnail((400,400))
                canvas.paste(preview,(x,y+40))
            draw.text((x+4,y+2),f"{start+i}: {row['candidate_id']}",fill='black')
            draw.text((x+4,y+20),str(row['country'])+' '+row['asset_id'][-20:],fill='black')
        name=f'{args.name}-sheet-{start//12:02d}.png'
        canvas.save(root/name)
        index.append(dict(sheet=name,candidates=[dict(index=start+i,candidate_id=row['candidate_id'],
            asset_id=row['asset_id'],country=row['country']) for i,row in enumerate(batch)]))
    (root/(args.name+'-sheet-index.json')).write_text(json.dumps(index,indent=2))
    (root/(args.name+'-rows.json')).write_text(json.dumps(rows))
    print(json.dumps(dict(candidates=len(rows),sheets=len(index))))


if __name__=='__main__':main()

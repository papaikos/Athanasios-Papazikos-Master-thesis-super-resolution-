"""Frozen CLIP scene-ranking pilot; assistant reviews are not ground truth."""
import argparse,hashlib,io,json,math,time
from pathlib import Path

def calibrate_threshold(rows, target_reject_coverage=.9):
 if not 0<target_reject_coverage<=1:raise ValueError('Invalid coverage target')
 scores=sorted(r['score'] for r in rows if r['review_reject'])
 if not scores:raise ValueError('Training reject observations required')
 return scores[len(scores)-math.ceil(target_reject_coverage*len(scores))]

def source_holdout_evaluation(rows):
 out={}
 for source in sorted({r['source_kind'] for r in rows}):
  train=[r for r in rows if r['source_kind']!=source];held=[r for r in rows if r['source_kind']==source];t=calibrate_threshold(train)
  out[source]=dict(threshold=t,training_count=len(train),heldout_count=len(held),heldout_reject_count=sum(r['review_reject'] for r in held),heldout_reject_missed=sum(r['review_reject'] and r['score']<t for r in held),heldout_pending_count=sum(not r['review_reject'] for r in held),heldout_pending_flagged=sum(not r['review_reject'] and r['score']>=t for r in held),independent_ground_truth=False,quality_accepted=False)
 return out

def main():
 from google.cloud import storage
 from scripts.maritime_cloud_inventory import require_cloud_vm,verify_output,json_bytes
 import torch,clip,numpy as np
 from PIL import Image
 require_cloud_vm();p=argparse.ArgumentParser();p.add_argument('--selection-pin',required=True);p.add_argument('--weights',required=True);p.add_argument('--output-prefix',required=True);p.add_argument('--scratch',required=True);a=p.parse_args()
 c=storage.Client();selection_pin=json.loads(Path(a.selection_pin).read_bytes());selection=json.loads(verify_output(c,selection_pin));weights=Path(a.weights)
 assert hashlib.sha256(weights.read_bytes()).hexdigest()==selection['model']['weight_sha256']
 torch.set_num_threads(4);torch.set_num_interop_threads(1);model,preprocess=clip.load(str(weights),device='cpu',jit=False);model.eval()
 prompts=selection['prompts'];texts=clip.tokenize([text for group in prompts for text in group['texts']]);start=time.monotonic()
 with torch.inference_mode():
  features=model.encode_text(texts).float();features/=features.norm(dim=-1,keepdim=True);centres=[];offset=0
  for group in prompts:
   v=features[offset:offset+len(group['texts'])].mean(dim=0);centres.append(v/v.norm());offset+=len(group['texts'])
  centres=torch.stack(centres);records=[]
  for row in selection['rows']:
   images=[];decoded=[]
   for ref in row['image_pins']:
    image=Image.open(io.BytesIO(verify_output(c,ref))).convert('RGB');assert image.size==(512,512)
    decoded.append(hashlib.sha256(np.asarray(image).tobytes()).hexdigest());images.append(preprocess(image))
   image_features=model.encode_image(torch.stack(images)).float();image_features/=image_features.norm(dim=-1,keepdim=True);similarity=(image_features@centres.T).cpu().numpy();bad=[i for i,g in enumerate(prompts) if g['quality_defect_prior']];clear=[i for i,g in enumerate(prompts) if not g['quality_defect_prior']]
   margins=similarity[:,bad].max(axis=1)-similarity[:,clear].max(axis=1)
   records.append(dict(sample_id=row['sample_id'],source_kind=row['source_kind'],score=float(margins.max()),review_reject=row['review_reject'],review_category=row['review_category'],image_pins=row['image_pins'],decoded_rgb_sha256=decoded,cosine_similarities=similarity.tolist(),model_predicted=True,final_eligible=False))
   print(json.dumps(dict(completed=len(records),source_count=len(selection['rows']))),flush=True)
 elapsed=time.monotonic()-start
 doc=dict(schema='clip-scene-ranking-development-pilot-v1',selection_pin=selection_pin,model=selection['model'],prompts=prompts,records=records,source_holdouts=source_holdout_evaluation(records),elapsed_seconds=elapsed,quality_accepted_count=0,final_eligible=False,caveat='Assistant reviewed development proxies, not independent cloud ground truth. Pending review is not a certified good negative. HR and saved native-overview derivatives are model-only resized to224; no source or training recipe changes.')
 raw=json_bytes(doc);bucket,prefix=a.output_prefix[5:].split('/',1);blob=c.bucket(bucket).blob(prefix.rstrip('/')+'/results.json');blob.upload_from_string(raw,if_generation_match=0,checksum='crc32c');blob.reload();assert blob.download_as_bytes(if_generation_match=blob.generation)==raw;print(json.dumps(dict(uri='gs://'+bucket+'/'+blob.name,generation=str(blob.generation),bytes=len(raw),crc32c=blob.crc32c,sha256=hashlib.sha256(raw).hexdigest())),flush=True)

if __name__=='__main__':main()

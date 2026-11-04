"""Licensed RGB cloud-model research scoring; no dataset qualification."""
import argparse,hashlib,io,json,time
from pathlib import Path
import numpy as np
from PIL import Image

def model_input(rgb):
 a=np.asarray(rgb)
 if a.shape!=(512,512,3) or a.dtype!=np.uint8:raise ValueError('Canonical uint8 RGB512 required')
 a=np.asarray(Image.fromarray(a).resize((256,256),Image.Resampling.BILINEAR))
 lo=float(a.min());hi=float(a.max())
 if hi==lo:raise ValueError('constant RGB input cannot use official minmax normalization')
 return ((a.astype(np.float32)-lo)/(hi-lo)).transpose(2,0,1)

def main():
 import torch
 import segmentation_models_pytorch as smp
 from google.cloud import storage
 from scripts.maritime_cloud_inventory import require_cloud_vm,verify_output,json_bytes
 from scripts.maritime_scene_triage import calibrate_threshold
 require_cloud_vm();p=argparse.ArgumentParser();p.add_argument('--protocol-pin',required=True);p.add_argument('--weights',required=True);p.add_argument('--output-prefix',required=True);a=p.parse_args();c=storage.Client();pin=json.loads(Path(a.protocol_pin).read_bytes());protocol=json.loads(verify_output(c,pin));selection=json.loads(verify_output(c,protocol['selection_pin']));cloudproto=json.loads(verify_output(c,protocol['cloud_partition_protocol_pin']));weights=Path(a.weights)
 assert hashlib.sha256(weights.read_bytes()).hexdigest()==protocol['model']['weight_sha256']
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 class Model(torch.nn.Module):
  def __init__(self):
   super().__init__();self.backbone=smp.Unet(encoder_name='mobilenet_v2',encoder_weights=None,in_channels=3,classes=2)
  def forward(self,x):return self.backbone(x)
 model=Model();model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True);model.eval();records=[];start=time.monotonic()
 with torch.inference_mode():
  for row in selection['rows']:
   assert len(row['image_pins'])==1;ref=row['image_pins'][0];image=Image.open(io.BytesIO(verify_output(c,ref))).convert('RGB');assert image.size==(512,512);rgb=np.asarray(image)
   try:x=model_input(rgb)
   except ValueError as e:records.append(dict(sample_id=row['sample_id'],source_kind=row['source_kind'],deferred=str(e),final_eligible=False));continue
   logits=model(torch.from_numpy(x).unsqueeze(0));assert tuple(logits.shape)==(1,2,256,256) and torch.isfinite(logits).all();mask=logits.argmax(dim=1);fraction=float((mask==1).float().mean());probmean=float(logits.softmax(dim=1)[0,1].mean())
   records.append(dict(sample_id=row['sample_id'],source_kind=row['source_kind'],image_pin=ref,decoded_hr_uint8_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),cloud_argmax_fraction=fraction,mean_cloud_model_probability=probmean,model_predicted=True,final_eligible=False));print(json.dumps(dict(completed=len(records),source_count=len(selection['rows']))),flush=True)
 elapsed=time.monotonic()-start;byid={r['sample_id']:r for r in records};evalrows=[]
 for r in cloudproto['rows']:
  pred=byid[r['sample_id']]
  if 'deferred' in pred:continue
  evalrows.append(dict(r,score=pred['cloud_argmax_fraction'],review_reject=r['apparent_cloud_proxy']))
 evaluations={}
 for source in sorted({r['source_kind'] for r in evalrows}):
  train=[r for r in evalrows if r['source_kind']==source and r['partition']=='calibration'];test=[r for r in evalrows if r['source_kind']==source and r['partition']=='heldout'];assert not({r['capture_group'] for r in train}&{r['capture_group'] for r in test})
  if len({r['review_reject'] for r in train})!=2 or len({r['review_reject'] for r in test})!=2:evaluations[source]=dict(status='INSUFFICIENT_CLASSES');continue
  threshold=calibrate_threshold(train,.9)
  def counts(rs):return dict(cloud_proxy_count=sum(r['review_reject'] for r in rs),cloud_proxy_missed=sum(r['review_reject'] and r['score']<threshold for r in rs),apparent_clear_proxy_count=sum(not r['review_reject'] for r in rs),apparent_clear_proxy_flagged=sum(not r['review_reject'] and r['score']>=threshold for r in rs))
  evaluations[source]=dict(threshold=threshold,calibration=counts(train),heldout=counts(test),quality_accepted=False,independent_accuracy_claim=False)
 doc=dict(schema='rgb-cloud-segmentation-development-pilot-v1',protocol_pin=pin,model=protocol['model'],records=records,source_evaluations=evaluations,elapsed_seconds=elapsed,quality_accepted_count=0,final_eligible=False,posthoc_followup_to_clip_pilot=True,limitations=protocol['limitations']);raw=json_bytes(doc);bucket,prefix=a.output_prefix[5:].split('/',1);blob=c.bucket(bucket).blob(prefix.rstrip('/')+'/results.json');blob.upload_from_string(raw,if_generation_match=0,checksum='crc32c');blob.reload();assert blob.download_as_bytes(if_generation_match=blob.generation)==raw;print(json.dumps(dict(uri='gs://'+bucket+'/'+blob.name,generation=str(blob.generation),bytes=len(raw),crc32c=blob.crc32c,sha256=hashlib.sha256(raw).hexdigest())),flush=True)

if __name__=='__main__':main()

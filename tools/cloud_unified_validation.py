"""Recompute COCO metrics on identical 525-frame validation truth, never test."""
import contextlib,csv,hashlib,importlib.util,io,json,os,sys,time
from pathlib import Path
import numpy as np
os.environ['YOLO_CONFIG_DIR']='/home/waas/GUIYU/runtime/ultralytics'

from faster_coco_eval import COCO,COCOeval_faster

ROOT=Path('/home/waas/GUIYU/runs')
OUT=ROOT/'formal100_b16_unified_validation_20261031'
TAG='formal100_b16_20261030_v1_'
keys=['pico','pico_b_mask','pico_a','pico_a_mask','pico_ab_cspatial_control','yolov8n','yolov10n','yolo11n','dfine_n','rtdetrv2_s']
def iou(a,b):
 x=max(0,min(a[2],b[2])-max(a[0],b[0]));y=max(0,min(a[3],b[3])-max(a[1],b[1]))
 inter=x*y;union=max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-inter
 return inter/union if union>0 else 0

def best_f1(records):
 events=[]; total=0
 for rec in records:
  gt=rec.get('reference_xyxy',[]); total+=len(gt);used=set()
  for pred in sorted(rec.get('predictions',[]),key=lambda x:-x['confidence']):
   choices=[(iou(pred['xyxy'],box),j) for j,box in enumerate(gt) if j not in used]
   score,j=max(choices,default=(0,-1));hit=score>=.5
   if hit: used.add(j)
   events.append((float(pred['confidence']),int(hit)))
 if not total or not events:return None
 events.sort(key=lambda x:-x[0]);tp=0;fp=0;best=None;i=0
 while i<len(events):
  threshold=events[i][0]
  while i<len(events) and events[i][0]==threshold:
   tp+=events[i][1];fp+=1-events[i][1];i+=1
  fn=total-tp;p=tp/(tp+fp) if tp+fp else 0;r=tp/total;f=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0
  if best is None or f>best['f1']:
   best={'threshold':threshold,'p':p,'r':r,'f1':f,'tp':tp,'fp':fp,'fn':fn}
 return best

def readjson(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def hashfile(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def metrics(rows):
 gt={'info':{},'images':[],'annotations':[],'categories':[{'id':1,'name':'surfacing'}]};pred=[];aid=0
 for imageid,r in enumerate(rows,1):
  gt['images'].append({'id':imageid,'width':r['width'],'height':r['height']})
  for b in r['reference_xyxy']:
   x,y,X,Y=b;w=max(0,X-x);h=max(0,Y-y);aid+=1;gt['annotations'].append({'id':aid,'image_id':imageid,'category_id':1,'bbox':[x,y,w,h],'area':w*h,'iscrowd':0})
  for item in r['predictions']:
   b=item['xyxy'];pred.append({'image_id':imageid,'category_id':1,'bbox':[b[0],b[1],max(0,b[2]-b[0]),max(0,b[3]-b[1])],'score':item['confidence']})
 if not pred:raise RuntimeError('No predictions')
 with contextlib.redirect_stdout(io.StringIO()):
  coco=COCO();coco.dataset=gt;coco.createIndex();dt=coco.loadRes(pred);ev=COCOeval_faster(coco,dt,'bbox')
  ev.params.iouThrs=np.linspace(.5,.95,10);ev.params.maxDets=[1,10,300];ev.evaluate();ev.accumulate()
  def ap(idx):
   x=ev.eval['precision'][idx,:,:,0,2];x=x[x>=0];return float(x.mean()) if len(x) else None
  ap50=ap(0);ap75=ap(5);ap5095=ap(slice(None))
 return {'ap50':ap50,'ap75':ap75,'ap50_95':ap5095,'best_f1_validation':best_f1(rows),'images':len(rows),'targets':aid,'predictions':len(pred)}

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 baseline=readjson(ROOT/(TAG+'pico_seed42')/'evaluation.json')['records']
 base={r['id']:r for r in baseline};assert len(base)==525 and sum(len(r['reference_xyxy']) for r in baseline)==2550
 index=readjson(Path('/home/waas/GUIYU/exports/reviewed5250_20261027_v1/paired_index.json'))
 src={r['id']:r for r in index['records'] if r['split']=='val'}
 assert set(src)==set(base)
 for r in src.values():
  p=Path(r['exported_rgb'])
  if hashfile(p)!=r['rgb_sha256']:raise RuntimeError('Image checksum mismatch: '+r['id'])
 print('VALIDATION_IMAGE_CHECKSUMS_OK 525',flush=True)
 results={}
 for key in keys:
  dest=ROOT/(TAG+key+'_seed42');stored=dest/'evaluation.json';cache=OUT/(key+'_predictions.json')
  if stored.exists():rows=readjson(stored)['records']
  elif cache.exists():rows=readjson(cache)
  elif key.startswith('yolo') and os.environ.get('ONLY_SAVED') == '1':
   print('MISSING_PREDICTIONS',key,flush=True);continue
  elif key.startswith('yolo'):
   from ultralytics import YOLO
   import torch
   if not torch.cuda.is_available():raise RuntimeError('Local CUDA not available')
   weights=dest/'weights/best.pt';model=YOLO(str(weights));paths=[str(Path(src[r['id']]['exported_rgb'])) for r in baseline]
   rows=[]
   for i,path in enumerate(paths,1):
    result=model.predict(source=path,stream=False,imgsz=960,batch=1,device=0,half=False,rect=False,conf=.001,iou=.5,max_det=300,verbose=False,save=False)[0]
    sid=Path(result.path).stem;r=base[sid];b=result.boxes
    pred=[{'xyxy':box,'confidence':float(conf)} for box,conf in zip(b.xyxy.detach().cpu().tolist(),b.conf.detach().cpu().tolist())]
    rows.append({'id':sid,'width':r['width'],'height':r['height'],'reference_xyxy':r['reference_xyxy'],'predictions':pred})
    if i%100==0:print('PREDICTIONS',key,i,'/525',flush=True)
   if len(rows)!=525 or set(r['id'] for r in rows)!=set(base):raise RuntimeError('Validation image coverage mismatch '+key)
   tmp=cache.with_suffix('.tmp');tmp.write_text(json.dumps(rows),encoding='utf-8');os.replace(tmp,cache)
   del model;torch.cuda.empty_cache()
  else:print('MISSING',key,flush=True);continue
  if {r['id'] for r in rows}!=set(base):raise RuntimeError('Image IDs differ '+key)
  for r in rows:
   if r['reference_xyxy']!=base[r['id']]['reference_xyxy']:raise RuntimeError('Reference boxes differ '+key+' '+r['id'])
  results[key]=metrics(rows)
  (OUT/'metrics.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
  print('MODEL_DONE',key,json.dumps(results[key]),flush=True)
 print('UNIFIED_VALIDATION_COMPLETE' if len(results)==10 else 'UNIFIED_VALIDATION_INCOMPLETE',len(results),flush=True)
if __name__=='__main__':main()

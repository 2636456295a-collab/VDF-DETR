import contextlib,hashlib,io,json,os,sys,traceback
from pathlib import Path
import numpy as np
ROOT=Path(os.environ.get('GUIYU_ROOT','/home/waas/GUIYU'))
RUNS=ROOT/'runs'; DATA=Path(os.environ.get('FORMAL_DATA',str(ROOT/'data/formal50_20261104_v1')))
OUT=Path(os.environ.get('FORMAL_EVAL_OUTPUT',str(RUNS/'formal50_unified_eval_20261106')));OUT.mkdir(parents=True,exist_ok=True)
TAG='formal50_20261104_v1_'
KEYS=['pico','pico_b_mask','pico_a','pico_a_mask','pico_ab_cspatial_control','yolov8n','yolov10n','yolo11n','yolo12n','dfine_n','rtdetrv2_s','faster_rcnn','retinanet','rtdetr_r18','deformable_detr']
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'runtime/ultralytics')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def atomic(p,obj):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8');os.replace(t,p)
def iou(a,b):
 x=max(0,min(a[2],b[2])-max(a[0],b[0]));y=max(0,min(a[3],b[3])-max(a[1],b[1]));inter=x*y
 u=max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-inter
 return inter/u if u else 0
def score(rows,threshold=.5):
 tp=fp=fn=0
 for r in rows:
  gt=r['reference_xyxy'];used=set()
  for p in sorted(r['predictions'],key=lambda x:-x['confidence']):
   if float(p['confidence'])<threshold:continue
   v,j=max(((iou(p['xyxy'],b),j) for j,b in enumerate(gt) if j not in used),default=(0,-1))
   if v>=.5:tp+=1;used.add(j)
   else:fp+=1
  fn+=len(gt)-len(used)
 p=tp/(tp+fp) if tp+fp else 0;r=tp/(tp+fn) if tp+fn else 0
 return {'threshold':threshold,'match_iou':.5,'P':p,'R':r,'F1':2*p*r/(p+r) if p+r else 0,'TP':tp,'FP':fp,'FN':fn}
def coco(rows):
 try:
  from faster_coco_eval import COCO,COCOeval_faster as COCOeval
 except ImportError:
  from pycocotools.coco import COCO
  from pycocotools.cocoeval import COCOeval
 gt={'info':{},'images':[],'annotations':[],'categories':[{'id':1,'name':'surfacing'}]};pred=[];aid=0
 for i,r in enumerate(rows,1):
  gt['images'].append({'id':i,'width':r['width'],'height':r['height']})
  for b in r['reference_xyxy']:
   x,y,X,Y=b;w=max(0,X-x);h=max(0,Y-y);aid+=1;gt['annotations'].append({'id':aid,'image_id':i,'category_id':1,'bbox':[x,y,w,h],'area':w*h,'iscrowd':0})
  for p in r['predictions']:
   x,y,X,Y=p['xyxy'];pred.append({'image_id':i,'category_id':1,'bbox':[x,y,max(0,X-x),max(0,Y-y)],'score':float(p['confidence'])})
 if not pred:raise RuntimeError('No predictions to evaluate')
 with contextlib.redirect_stdout(io.StringIO()):
  c=COCO();c.dataset=gt;c.createIndex();dt=c.loadRes(pred);e=COCOeval(c,dt,'bbox');e.params.maxDets=[1,10,300];e.evaluate();e.accumulate()
  def ap(i):
   z=e.eval['precision'][i,:,:,0,2];z=z[z>=0];return float(z.mean()) if len(z) else None
  return {'AP50':ap(0),'AP75':ap(5),'AP50_95':ap(slice(None)),'max_detections':300,'prediction_count':len(pred)}
def main():
 baseline=read(RUNS/(TAG+'pico_seed42')/'evaluation.json')['records'];base={r['id']:r for r in baseline}
 assert len(base)==525 and sum(len(x['reference_xyxy']) for x in baseline)==2550
 idx=read(DATA/'paired_index.json');vr=[r for r in idx['records'] if r['split']=='val'];assert len(vr)==525 and {r['id'] for r in vr}==set(base)
 for r in vr:
  p=DATA/'images'/'val'/(r['id']+'.png');assert p.is_file(),str(p)
  if r.get('rgb_sha256'):
   h=hashlib.sha256(p.read_bytes()).hexdigest()
   assert h==r['rgb_sha256'],'Image SHA256 mismatch '+r['id']
 print('VALIDATION_IMAGES_VERIFIED 525 targets 2550',flush=True)
 for key in os.environ.get('FORMAL_MODELS',','.join(KEYS)).split(','):
  output=OUT/(key+'_metrics.json')
  if output.exists():print('MODEL_ALREADY_DONE',key,flush=True);continue
  try:
   dest=RUNS/(TAG+key+'_seed42');stored=dest/'evaluation.json';cache=OUT/(key+'_predictions.json')
   if stored.exists():rows=read(stored)['records']
   elif cache.exists():rows=read(cache)
   elif key.startswith('yolo'):
    from ultralytics import YOLO
    import torch
    assert torch.cuda.is_available(),'CUDA unavailable'
    model=YOLO(str(dest/'weights/best.pt'));rows=[]
    for n,r in enumerate(baseline,1):
     z=model.predict(source=str(DATA/'images'/'val'/(r['id']+'.png')),imgsz=960,batch=1,device=0,half=False,rect=False,conf=.001,iou=.5,max_det=300,verbose=False,save=False)[0]
     rows.append({'id':r['id'],'width':r['width'],'height':r['height'],'reference_xyxy':r['reference_xyxy'],'predictions':[{'xyxy':b,'confidence':float(c)} for b,c in zip(z.boxes.xyxy.cpu().tolist(),z.boxes.conf.cpu().tolist())]})
     if n%100==0:print('PREDICT',key,n,flush=True)
    atomic(cache,rows);del model;torch.cuda.empty_cache()
   else:raise FileNotFoundError(str(stored))
   assert len(rows)==525 and {r['id'] for r in rows}==set(base)
   for r in rows:assert r['reference_xyxy']==base[r['id']]['reference_xyxy'],'GT mismatch '+key+' '+r['id']
   result={'model':key,'dataset':'formal50_20261104_v1 validation','images':len(rows),'targets':2550,'fixed_050':score(rows),**coco(rows),'weights':'best','note':'Recomputed from stored predictions; upstream candidate and postprocessing may differ by architecture.'}
   atomic(output,result);print('MODEL_DONE',key,json.dumps(result),flush=True)
  except Exception as e:print('MODEL_FAILED',key,repr(e),flush=True);traceback.print_exc()
 print('VALIDATION_DONE',sum((OUT/(k+'_metrics.json')).exists() for k in KEYS),'of',len(KEYS),flush=True)
if __name__=='__main__':main()

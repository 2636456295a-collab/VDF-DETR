"""DEIMv2 RGB fine-tuning adapter; official model/criterion, frozen fish split.

New accepted5000_v4 dataset: 4200 training frames and 400 validation frames.
Selection is validation mAP50-95; no hard depth gate is used in this RGB queue. This is not
the authors' full COCO training recipe. No held-out 10.19 test inference.
"""
import argparse, copy, csv, datetime, hashlib, importlib.util, json, math, os
from pathlib import Path
import random, sys, time, traceback, types
ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / 'comparisons/short20_v4_seed42'
INDEX = ROOT / 'exports/accepted5000_v4/paired_index.json'
PROVENANCE = json.loads((ROOT / 'runtime/external/deimv2_source_provenance.json').read_text(encoding='utf-8'))
SOURCE = Path(PROVENANCE['source_directory'])
DFINE_SOURCE = ROOT / 'runtime/external/D-FINE-956d1709314c'
MODELS = ['n', 'pico']
os.environ.update(YOLO_CONFIG_DIR=str(ROOT/'runtime/ultralytics'), TORCH_HOME=str(ROOT/'runtime/torch'),
                 HF_HOME=str(ROOT/'runtime/huggingface'), HF_HUB_OFFLINE='1', PYTHONUNBUFFERED='1')
sys.path[:0] = ['F:/du/depth_rule_ai2000_train_v1/runtime/dfine_deps', str(SOURCE), str(DFINE_SOURCE)]
sys.modules.setdefault('tensorboard.compat.notf', types.ModuleType('tensorboard.compat.notf'))

def run_dir(size):
    return ROOT / 'runs' / ('short20_v4_' + size + '_rgb_seed42')

def weights(size):
    return ROOT / 'runtime/weights' / ('deimv2_' + size + '_coco.safetensors')

def make_config(size):
    from engine.core import YAMLConfig
    cfg = YAMLConfig(str(SOURCE/'configs/deimv2'/('deimv2_hgnetv2_'+size+'_coco.yml')))
    cfg.yaml_cfg.update(num_classes=1, remap_mscoco_category=False, eval_spatial_size=[960,960],
                        num_top_queries=300 if size=='n' else 200)
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    cfg.yaml_cfg['PostProcessor']['num_top_queries'] = 300 if size=='n' else 200
    cfg.yaml_cfg['DEIMCriterion']['matcher']['matcher_change_epoch'] = 90
    return cfg

def make_model(size, pretrained=True):
    from safetensors.torch import load_file
    cfg = make_config(size)
    model = cfg.model
    metadata = {'parameters':sum(p.numel() for p in model.parameters()), 'source_commit':PROVENANCE['commit'],
                'native_queries':300 if size=='n' else 200, 'input_channels':3}
    if pretrained:
        state = load_file(str(weights(size)))
        # Hub serialization removes aliases for the shared Pico bbox heads.
        if size == 'pico':
            assert model.decoder.dec_bbox_head[0] is model.decoder.dec_bbox_head[1]
            assert model.decoder.dec_bbox_head[0] is model.decoder.dec_bbox_head[2]
            for key, value in list(state.items()):
                if key.startswith('decoder.dec_bbox_head.0.'):
                    for layer in [1,2]:
                        state.setdefault(key.replace('.0.', '.'+str(layer)+'.', 1), value)
        current = model.state_dict()
        mismatched = [k for k,v in state.items() if k in current and current[k].shape != v.shape]
        allowed_geometry = {'decoder.anchors','decoder.valid_mask'}
        assert all('score_head' in k or 'denoising_class_embed' in k or k in allowed_geometry for k in mismatched), mismatched
        assert not set(state)-set(current), 'Unexpected checkpoint keys'
        loaded = {k:v for k,v in state.items() if k in current and k not in mismatched}
        missing = model.load_state_dict(loaded, strict=False)
        # These fixed native constants are omitted in the Hub serialization.
        constants = {'decoder.up','decoder.reg_scale'}
        assert set(missing.missing_keys) == set(mismatched)|constants, missing
        assert not missing.unexpected_keys
        metadata.update(pretrained_weights_sha256=sha(weights(size)), loaded_tensors=len(loaded),
                        reinitialized_class_heads=[k for k in mismatched if k not in allowed_geometry],
                        regenerated_geometry=sorted(set(mismatched)&allowed_geometry),
                        native_initialized_constants=sorted(constants))
    return model,cfg,metadata

def decode(output, postprocessor, imgsz=960, conf=.001):
    import torch
    sizes = torch.full((len(output['pred_logits']),2),imgsz,device=output['pred_logits'].device,dtype=torch.int64)
    result=[]
    for p in postprocessor(output,sizes):
        assert len(p['scores']) in [200,300]
        mask=p['scores']>conf
        result.append({'bboxes':p['boxes'][mask],'conf':p['scores'][mask],'cls':p['labels'][mask].float()})
    return result


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def now():return datetime.datetime.now().isoformat(timespec='seconds')

def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def seed_all(seed=42):
    import numpy as np,torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False

def datasets(batch=1):
    import yaml
    from ultralytics.cfg import get_cfg
    e=load_module('dfine_paired_evaluation_helpers',ROOT/'tools/evaluate_paired_baselines.py')
    data=yaml.safe_load((ROOT/'exports/accepted5000_v4/rgb/data.yaml').read_text(encoding='utf-8'))
    args=get_cfg(overrides={'imgsz':960,'fliplr':.5,'flipud':0.,'fraction':1.,'task':'detect'})
    out=[]
    for split,mode in [('train','train'),('val','val')]:
        ds=e.paired.PairedDataset(img_path=str(Path(data['path'])/data[split]),imgsz=960,batch_size=batch,
            augment=mode=='train',hyp=args,rect=False,cache=False,single_cls=False,stride=32,pad=0.0,
            prefix='DEIMv2 '+mode+': ',task='detect',classes=None,data=data,fraction=1.0)
        out.append(ds)
    assert len(out[0])==4200 and len(out[1])==400
    return e,data,*out

def targets_from_batch(batch,device):
    import torch
    targets=[]
    for i in range(len(batch['img'])):
        selected=batch['batch_idx']==i
        targets.append({'labels':batch['cls'][selected].view(-1).to(device,dtype=torch.int64),
                        'boxes':batch['bboxes'][selected].to(device,dtype=torch.float32)})
    return targets

def move_batch(batch,device):
    import torch
    for key in ['img','batch_idx','cls','bboxes']:
        batch[key]=batch[key].to(device)
    batch['img']=batch['img'].float()/255.
    return batch

def new_validator(e,data,dest,device):
    import torch
    dest.mkdir(parents=True,exist_ok=True)
    v=e.ReviewValidator(save_dir=dest,args={'imgsz':960,'batch':1,'conf':.001,'iou':.5,'plots':False,
        'save_json':False,'save_txt':False,'max_det':300,'verbose':False})
    v.data=data;v.device=torch.device(device);v.training=False;v.iouv=v.iouv.to(device)
    v.init_metrics(types.SimpleNamespace(names={0:'surfacing'},end2end=True))
    return v

def finish_metrics(v):
    import numpy as np
    metrics=v.get_stats();rows=sorted(v.review_rows,key=lambda r:r['id'])
    for row in rows:
        row['count_conf025']=sum(p['confidence']>=.25 for p in row['predictions'])
        row['absolute_count_error_conf025']=abs(row['count_conf025']-row['reference_count'])
    mae=float(np.mean([r['absolute_count_error_conf025'] for r in rows]))
    assert abs(mae-metrics['metrics/count_MAE_conf025'])<1e-6
    return {'metrics':{k:float(x) for k,x in metrics.items()},'rows':rows,'count_MAE_conf025':mae}

def validate(model,cfg,e,data,valset,dest,device='cuda',gated=False):
    import torch,numpy as np,cv2
    from torch.utils.data import DataLoader
    from ultralytics.utils import ops
    loader=DataLoader(valset,batch_size=1,shuffle=False,num_workers=0,collate_fn=valset.collate_fn)
    v=new_validator(e,data,dest,device);model.eval();post=cfg.postprocessor.to(device).eval()
    lookup={r['id']:r for r in read(INDEX)['records'] if r['split']=='val'};audits=[]
    with torch.no_grad():
        for i,batch in enumerate(loader):
            batch=move_batch(batch,device);out=model(batch['img']);preds=decode(out,post)
            if gated:
                for j,p in enumerate(preds):
                    sid=Path(batch['im_file'][j]).stem;r=lookup[sid]
                    depth=cv2.imdecode(np.fromfile(r['exported_depth'],np.uint8),-1)
                    assert depth.dtype==np.uint16 and depth.shape==(720,1280)
                    boxes=ops.scale_boxes(batch['img'].shape[2:],p['bboxes'].clone(),batch['ori_shape'][j],ratio_pad=batch['ratio_pad'][j]).cpu().numpy()
                    keep=[];decisions=[]
                    for box,conf in zip(boxes,p['conf'].cpu().tolist()):
                        x1,y1,x2,y2=box
                        roi=depth[max(0,math.floor(y1)):min(720,math.ceil(y2)),max(0,math.floor(x1)):min(1280,math.ceil(x2))]
                        nz=roi[roi>0];median=float(np.median(nz)*r['depth_scale_m_per_unit']) if nz.size else None
                        ok=median is not None and median<1.17;keep.append(ok)
                        decisions.append({'xyxy':list(map(float,box)),'confidence':conf,'raw_nonzero_median_m':median,'keep':ok})
                    mask=torch.tensor(keep,device=device,dtype=torch.bool)
                    preds[j]={k:x[mask] for k,x in p.items()};audits.append({'id':sid,'decisions':decisions})
            v.update_metrics(preds,batch)
            if (i+1)%100==0:print('validation',i+1,len(valset),flush=True)
    result=finish_metrics(v)
    assert len(result['rows'])==400 and {r['id'] for r in result['rows']}==set(lookup)
    assert sum(r['reference_count'] for r in result['rows'])==1065
    if gated:save(dest.parent/'depth_decisions.json',audits)
    return result

def gpu_preflight(size):
    import torch
    from src.optim.ema import ModelEMA
    seed_all();free,total=torch.cuda.mem_get_info();budget=min(total*.85,free-256*1024**2)
    model,cfg,meta=make_model(size);model=model.cuda().train();criterion=cfg.criterion.cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0001)
    ema=ModelEMA(model,decay=.9999,warmups=2000)
    e,data,trainset,valset=datasets()
    i=max(range(len(trainset.labels)),key=lambda j:len(trainset.labels[j]['cls']))
    batch=move_batch(trainset.collate_fn([trainset[i]]),'cuda');targets=targets_from_batch(batch,'cuda')
    losses=criterion(model(batch['img'],targets),targets);loss=sum(losses.values())
    loss.backward();assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step();ema.update(model);model.eval()
    with torch.no_grad():out=model(batch['img']);decoded=decode(out,cfg.postprocessor.cuda())
    assert len(decoded)==1
    torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()
    result={'size':size,'batch':1,'imgsz':960,'peak_allocated_MiB':peak/1024**2,
        'physical_memory_MiB':total/1024**2,'safe_budget_MiB':budget/1024**2,
        'within_physical_VRAM_budget':peak<=budget,'finite_training_and_inference':True,
        'parameters':meta['parameters'],'dense_reference_count':len(targets[0]['labels']),'timestamp':now()}
    save(JOB/size/'gpu_preflight.json',result);print(json.dumps(result),flush=True)
    if peak>budget:raise RuntimeError('DEIMv2 GPU preflight exceeds physical memory budget; refuse shared memory spill.')

def checkpoint(path,model,ema,optimizer,epoch,best,best_epoch,elapsed):
    import torch,numpy as np
    payload={'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':optimizer.state_dict(),
        'epoch':epoch,'best_fitness':best,'best_epoch':best_epoch,'elapsed':elapsed,
        'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
        'cuda_rng':torch.cuda.get_rng_state_all(),'configuration_sha256':sha(JOB/'configuration.json')}
    tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)

def train(size):
    import torch,numpy as np
    from torch.utils.data import DataLoader
    from src.optim.ema import ModelEMA
    seed_all();settings=read(JOB/'configuration.json');dest=run_dir(size);dest.mkdir(exist_ok=True)
    (dest/'weights').mkdir(exist_ok=True)
    last=dest/'weights/last.pth';best_path=dest/'weights/best.pth';history=dest/'results.csv'
    model,cfg,meta=make_model(size);model=model.cuda();criterion=cfg.criterion.cuda()
    backbone=[p for n,p in model.named_parameters() if n.startswith('backbone.') and p.requires_grad]
    rest=[p for n,p in model.named_parameters() if not n.startswith('backbone.') and p.requires_grad]
    optimizer=torch.optim.AdamW([{'params':backbone,'lr_scale':.5},{'params':rest,'lr_scale':1.}],lr=settings['lr0'],betas=tuple(settings['betas']),weight_decay=settings['weight_decay'])
    ema=ModelEMA(model,decay=.9999,warmups=2000);start=0;best=-1.;best_epoch=-1;elapsed_before=0.
    if last.exists():
        state=torch.load(last,map_location='cpu',weights_only=False)
        assert state['configuration_sha256']==sha(JOB/'configuration.json')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);ema.load_state_dict(state['ema'])
        start=state['epoch']+1;best=state['best_fitness'];best_epoch=state['best_epoch'];elapsed_before=state['elapsed']
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng']);del state
        # An interrupted epoch may already have written its CSV row. Keep only checkpointed rows.
        if history.exists():
            with history.open(encoding='utf-8-sig',newline='') as f:old=list(csv.DictReader(f))
            kept=[r for r in old if int(r['epoch'])<=start]
            with history.open('w',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(old[0]));w.writeheader();w.writerows(kept)
    elif history.exists():raise RuntimeError('History exists without a resumable checkpoint; refuse to overwrite.')
    save(dest/'configuration.json',{'settings':settings,'model':meta,'resume_epoch':start})
    if start>=settings['epochs']:
        save(dest/'training_complete.json',{'completed':now(),'epochs_finished':start,'best_epoch':best_epoch+1,'best_fitness':best})
        return
    e,data,trainset,valset=datasets();loader=DataLoader(trainset,batch_size=1,shuffle=True,num_workers=0,collate_fn=trainset.collate_fn)
    session_started=time.perf_counter();columns=['epoch','time','train/total_loss','metrics/precision(B)','metrics/recall(B)',
        'metrics/mAP50(B)','metrics/mAP50-95(B)','metrics/count_MAE_conf025','lr/pg0']
    if not history.exists():
        with history.open('w',encoding='utf-8',newline='') as f:csv.DictWriter(f,fieldnames=columns).writeheader()
    accumulation=settings['nominal_effective_batch'];epochs=settings['epochs']
    for epoch in range(start,epochs):
        model.train();criterion.train();optimizer.zero_grad(set_to_none=True)
        epoch_started=time.perf_counter();loss_total=0.
        for i,batch in enumerate(loader):
            lr=settings['lr0']*((1-epoch/epochs)*(1-settings['lrf'])+settings['lrf'])
            warmup=min(1.,(epoch+(i+1)/len(loader))/settings['warmup_epochs'])
            for group in optimizer.param_groups:group['lr']=lr*warmup*group['lr_scale']
            batch=move_batch(batch,'cuda');targets=targets_from_batch(batch,'cuda')
            outputs=model(batch['img'],targets);losses=criterion(outputs,targets,epoch=epoch);loss=sum(losses.values())
            if not torch.isfinite(loss):raise RuntimeError('Non-finite DEIMv2 loss')
            # Average each accumulated group; final incomplete group uses its actual size.
            group_size=min(accumulation,len(loader)-(i//accumulation)*accumulation)
            (loss/group_size).backward();loss_total+=float(loss.detach())
            if (i+1)%accumulation==0 or i+1==len(loader):
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),settings['clip_grad_norm'])
                if not torch.isfinite(norm):raise RuntimeError('Non-finite DEIMv2 gradient')
                optimizer.step();optimizer.zero_grad(set_to_none=True);ema.update(model)
            if i==0 or (i+1)%10==0:
                elapsed=time.perf_counter()-epoch_started
                progress={'model':'DEIMv2-'+size.upper(),'epoch':epoch+1,'epochs':epochs,'batches_completed':i+1,
                    'batches_per_epoch':len(loader),'elapsed_epoch_seconds':elapsed,
                    'estimated_epoch_seconds':elapsed/(i+1)*len(loader),'train_loss_mean':loss_total/(i+1),'updated':now()}
                save(JOB/size/'batch_progress.json',progress)
                if i==0 or (i+1)%100==0:print(json.dumps(progress),flush=True)
        val=validate(ema.module,cfg,e,data,valset,dest/'epoch_validation','cuda')
        m=val['metrics'];fitness=m['metrics/mAP50-95(B)']
        improved=fitness>best
        if improved:best=fitness;best_epoch=epoch
        elapsed=elapsed_before+time.perf_counter()-session_started
        row={'epoch':epoch+1,'time':round(elapsed,3),'train/total_loss':loss_total/len(loader),
            **{k:m[k] for k in columns if k.startswith('metrics/')},'lr/pg0':optimizer.param_groups[0]['lr']}
        with history.open('a',encoding='utf-8',newline='') as f:csv.DictWriter(f,fieldnames=columns).writerow(row)
        checkpoint(last,model,ema,optimizer,epoch,best,best_epoch,elapsed)
        if improved:checkpoint(best_path,model,ema,optimizer,epoch,best,best_epoch,elapsed)
        print('EPOCH_COMPLETE',json.dumps(row),flush=True)
        save(JOB/size/'training_progress.json',{'epochs_finished':epoch+1,'best_epoch':best_epoch+1,'latest':row,'timestamp':now()})
        if epoch-best_epoch>=settings['patience']:break
    save(dest/'training_complete.json',{'completed':now(),'epochs_finished':epoch+1,'best_epoch':best_epoch+1,'best_fitness':best})

from short20_common import save, read, retry_checkpoint
checkpoint=retry_checkpoint(checkpoint)

"""Shared paths and integrity checks for the new, isolated 5000-frame experiment."""
import hashlib,json,os,sys
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'exports/accepted5000_v4'
JOB=ROOT/'comparisons/short20_v4_seed42'
MODELS=['pico','n','dfine_n']
os.environ.update(YOLO_CONFIG_DIR=str(JOB/'runtime/ultralytics'),TORCH_HOME=str(ROOT/'runtime/torch'),
                  HF_HOME=str(ROOT/'runtime/huggingface'),HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def now():return datetime.now().astimezone().isoformat(timespec='seconds')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');t.replace(p)
def run_dir(model):return ROOT/'runs'/('short20_v4_'+model+'_rgb_seed42')
def assert_frozen():
    cfg=read(JOB/'configuration.json')
    for path,digest in cfg['frozen_files'].items():assert sha(path)==digest,'Frozen dependency changed: '+path
    return cfg
def module(name,path):
    import importlib.util
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def gpu_lock():
    sys.path.insert(0,str(ROOT))
    from guiyu_runner import gpu_lock as lock
    return lock()
def ensure_queue_authorized():
    import psutil
    state=read(JOB/'state.json');pid=state.get('pid')
    if pid and pid!=os.getpid() and psutil.pid_exists(pid):
        try:cmd=' '.join(psutil.Process(pid).cmdline())
        except psutil.AccessDenied:cmd=''
        if 'short20_queue.py' in cmd and os.getppid()!=pid:
            raise RuntimeError('The serial queue is active. Do not launch a second GPU experiment.')

def result_payload(model,weights,metrics,rows,parameters):
    import numpy as np
    index=read(DATA/'paired_index.json');truth={r['id']:r for r in index['records'] if r['split']=='val'}
    assert set(r['id'] for r in rows)==set(truth) and len(rows)==400
    assert sum(r['reference_count'] for r in rows)==1065
    for r in rows:
        r['count_conf025']=sum(p['confidence']>=.25 for p in r['predictions'])
        r['absolute_count_error_conf025']=abs(r['count_conf025']-r['reference_count'])
    errors=[r['count_conf025']-r['reference_count'] for r in rows]
    mae=float(np.mean(np.abs(errors)));negative=[r for r in rows if not r['reference_count']]
    out={'architecture':{'pico':'DEIMv2-Pico','n':'DEIMv2-N','dfine_n':'D-FINE-N'}[model],'method':'rgb',
         'role':'development validation on the reviewed v4 label set; not independent test',
         'images':400,'reference_boxes':1065,'metrics':{k:float(v) for k,v in metrics.items()},
         'count_MAE_conf025':mae,'count_RMSE_conf025':float(np.sqrt(np.mean(np.square(errors)))),
         'negative_images':len(negative),'negative_images_with_false_positive':sum(r['count_conf025']>0 for r in negative),
         'parameters':parameters,'weights':str(weights),'test_used':False,'records':rows,'strata':{},
         'provenance':{'weights_sha256':sha(weights),'dataset_index_sha256':sha(DATA/'paired_index.json'),
                       'configuration_sha256':sha(JOB/'configuration.json'),'seed':42,'imgsz':960,
                       'count_confidence':.25,'AP_confidence':.001,'label_version':'accepted5000_v4',
                       'label_origin':'reviewed','depth_used_as_input':False,'DO_used_as_input':False}}
    for name,rule in [('zero',lambda n:n==0),('one_to_five',lambda n:1<=n<=5),('six_plus',lambda n:n>=6)]:
        group=[r for r in rows if rule(r['reference_count'])]
        out['strata'][name]={'images':len(group),'count_MAE_conf025':float(np.mean([r['absolute_count_error_conf025'] for r in group])) if group else None}
    return out

def efficiency(model,net):
    import time,torch,numpy as np
    net=net.cuda().float().eval();x=torch.zeros(1,3,960,960,device='cuda');timings=[]
    with torch.no_grad():
        for _ in range(30):net(x)
        torch.cuda.synchronize()
        for _ in range(100):
            t=time.perf_counter();net(x);torch.cuda.synchronize();timings.append((time.perf_counter()-t)*1000)
    d={'parameters':sum(p.numel() for p in net.parameters()),'input_shape':[1,3,960,960],'precision':'FP32',
       'device':torch.cuda.get_device_name(0),'scope':'model forward only; no file read or external postprocessing',
       'warmup':30,'iterations':100,'mean_latency_ms':float(np.mean(timings)),'FPS_from_mean_latency':float(1000/np.mean(timings)),
       'measured_at':now()}
    try:
        from calflops import calculate_flops
        with torch.no_grad():flops,macs,params=calculate_flops(net,args=[x],print_results=False,print_detailed=False,output_as_string=False)
        d.update(profiled_FLOPs=float(flops),profiled_MACs=float(macs),FLOPs_are_partial_estimate=True,
                 profiler_limitations='functional and grid_sample operations may not all be counted')
    except Exception as e:d['profiler_error']=str(e)
    save(JOB/model/'efficiency.json',d);return d

# Retain the proven Windows atomic-write retry fix in the new job.
import accepted5000_recovery as io_retry
io_retry.JOB=JOB
save=io_retry.resilient_save
read=io_retry.resilient_read

def retry_checkpoint(native):
    import time
    def wrapped(*args,**kwargs):
        for attempt in range(12):
            try:return native(*args,**kwargs)
            except PermissionError:
                if attempt==11:raise
                time.sleep(min(.1*1.5**attempt,.75))
    return wrapped

"""New ablation architectures; historical A+B and training sources remain immutable."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import short20_deim_core as core
native_make=core.make_model
SPECS={
 'pico':('rgb',False,'none'),
 'pico_a':('rgb',True,'none'),
 'pico_b':('rgbd_mask',False,'quality'),
 'pico_ab':('rgbd_mask',True,'quality'),
 'pico_a_depth':('rgbd',True,'direct'),
 'pico_a_mask':('rgbd_mask',True,'mask'),
 'yolov8n':('rgb',False,'none'),
}
def build_model(key,pretrained=True):
 import torch
 from torch import nn
 from torch.nn import functional as F
 mode,detail,depth=SPECS[key]
 base,cfg,meta=native_make('pico',pretrained)
 if key=='pico':return base,cfg,meta
 original={k:v.clone() for k,v in base.state_dict().items()}
 class Encoder(nn.Module):
  def __init__(self,enc):
   super().__init__()
   for name,module in enc.named_children():self.add_module(name,module)
   self.use_detail=detail;self.use_depth=depth!='none';self.depth_mode=depth
   if detail:
    self.detail=nn.Sequential(nn.Conv2d(256,112,1,bias=False),nn.BatchNorm2d(112),nn.SiLU(),nn.Conv2d(112,112,3,2,1,groups=112,bias=False),nn.BatchNorm2d(112),nn.SiLU(),nn.Conv2d(112,112,1,bias=False))
    self.detail_scale=nn.Parameter(torch.tensor(.01))
   if depth!='none':
    layers=[];c=2
    for out in [8,16,32,112]:
     layers.extend([nn.Conv2d(c,c,3,2,1,groups=c,bias=False),nn.Conv2d(c,out,1,bias=False),nn.BatchNorm2d(out),nn.SiLU()]);c=out
    self.depth_branch=nn.Sequential(*layers);self.depth_scale=nn.Parameter(torch.tensor(.01))
    if depth=='quality':self.quality_gate=nn.Sequential(nn.Conv2d(227,16,1),nn.SiLU(),nn.Conv2d(16,1,1),nn.Sigmoid())
  def fuse(self,feats,dm=None):
   x=self.input_proj[0](feats[-1])
   if self.use_detail:x=x+self.detail_scale*self.detail(feats[0])
   if self.use_depth:
    d=dm[:,:1]
    m=torch.ones_like(d) if depth=='direct' else (dm[:,1:2]>.5).to(d.dtype)
    d=d*m;z=self.depth_branch(torch.cat([d,m],1))
    if depth=='direct':gate=1.
    else:
     q=F.avg_pool2d(m,16,16);gate=q
     if depth=='quality':
      mean=F.avg_pool2d(d,16,16)/q.clamp_min(1e-6)
      var=(F.avg_pool2d(d.square(),16,16)/q.clamp_min(1e-6)-mean.square()).clamp_min(0)
      gate=q*self.quality_gate(torch.cat([x,z,q,mean,var],1))
    x=x+self.depth_scale*gate*z
   return x
  def forward(self,feats,dm=None):
   x=self.fuse(feats,dm);small=self.bi_fusion(self.down_sample1(x))
   out16=self.fpn_block(x+F.interpolate(small,scale_factor=2.,mode='nearest'))
   return [out16,self.pan_block(small+self.down_sample2(out16))]
 class Model(nn.Module):
  def __init__(self):
   super().__init__();self.backbone=base.backbone
   if detail:self.backbone.return_idx=[1,2]
   self.encoder=Encoder(base.encoder);self.decoder=base.decoder
  def forward(self,x,targets=None):
   assert x.shape[1]=={'rgb':3,'rgbd':4,'rgbd_mask':5}[mode]
   return self.decoder(self.encoder(self.backbone(x[:,:3]),x[:,3:]),targets)
 net=Model();state=net.state_dict()
 assert all(k in state and torch.equal(v,state[k]) for k,v in original.items())
 meta.update(architecture=key,input_channels={'rgb':3,'rgbd':4,'rgbd_mask':5}[mode],base_parameters=meta['parameters'],parameters=sum(p.numel() for p in net.parameters()),all_original_tensors_preserved=True,detail_enabled=detail,depth_mode=depth,new_state_keys=sorted(set(state)-set(original)))
 return net,cfg,meta

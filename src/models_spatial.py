"""Historical 99-parameter spatial C on today's unchanged A+B, EMA-safe implementation."""
import copy
import torch
from torch import nn
import wall20_models_v1 as historical
NATIVE_BUILD=historical.build_model
KEY='pico_ab_cspatial_control'
ORDER=[KEY]
NAMES={KEY:'Pico + A + B + C_spatial'}
SPECS={KEY:('rgbd_mask',True,'mask'),'pico_ab_mask_control':('rgbd_mask',True,'mask')}
historical.SPECS.update(SPECS)
class ResidualSpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial=nn.Conv2d(2,1,7,padding=3,bias=False)
        self.gamma=nn.Parameter(torch.zeros(()))
    def forward(self,x):
        pooled=torch.cat([x.mean(1,keepdim=True),x.amax(1,keepdim=True)],dim=1)
        s=torch.sigmoid(self.spatial(pooled))
        return x*(1+torch.tanh(self.gamma)*(2*s-1))
def build_model(key,pretrained=True):
    assert key==KEY
    net,cfg,meta=NATIVE_BUILD(key,pretrained)
    before={k:v.clone() for k,v in net.state_dict().items()}
    class SpatialEncoder(type(net.encoder)):
        def fuse(self,feats,dm=None):
            return self.spatial_attention(super().fuse(feats,dm))
    net.encoder.__class__=SpatialEncoder
    # Keep today's A+B/data-loader RNG stream unchanged by extra module initialization.
    rng=torch.get_rng_state()
    net.encoder.spatial_attention=ResidualSpatialAttention()
    torch.set_rng_state(rng)
    assert all(torch.equal(v,net.state_dict()[k]) for k,v in before.items())
    meta.update(architecture=key,parameters=sum(p.numel() for p in net.parameters()),reference_AB_parameters=1542755,added_C_parameters=99,depth_rule_postfilter=False,depth_normalization_max_m=1.7,
      C={'type':'residual_spatial_attention','kernel':7,'channels':112,'gamma_initial':0.,'formula':'F*(1+tanh(gamma)*(2*sigmoid(conv7([mean_c(F),max_c(F)]))-1))','location':'after_AB_fuse_before_multiscale_encoder','EMA_safe_encoder_subclass':True,'backbone_AB_unchanged':True,'preserve_AB_random_stream':True})
    assert meta['parameters']==1542854 and not hasattr(net.encoder,'quality_gate')
    return net,cfg,meta
def self_checks():
    torch.set_num_threads(2);torch.manual_seed(42)
    m=ResidualSpatialAttention();x=torch.randn(2,112,19,17)
    assert torch.equal(m(x),x)
    opt=torch.optim.SGD(m.parameters(),lr=.1)
    m(x).square().mean().backward();assert m.gamma.grad.abs()>0;opt.step();opt.zero_grad()
    m(x).square().mean().backward();assert m.spatial.weight.grad.abs().sum()>0
    torch.manual_seed(42);base,_,bm=NATIVE_BUILD('pico_ab_mask_control',True);base_rng=torch.get_rng_state().clone()
    torch.manual_seed(42);net,_,meta=build_model(KEY,True);new_rng=torch.get_rng_state().clone()
    assert torch.equal(base_rng,new_rng)
    assert all(torch.equal(v,net.state_dict()[k]) for k,v in base.state_dict().items())
    base.eval();net.eval();z=torch.rand(1,5,320,320);z[:,4]=1
    with torch.no_grad():
        a=base.encoder(base.backbone(z[:,:3]),z[:,3:]);b=net.encoder(net.backbone(z[:,:3]),z[:,3:])
    assert all(torch.equal(v,w) for v,w in zip(a,b))
    with torch.no_grad():net.encoder.spatial_attention.gamma.fill_(.5)
    duplicate=copy.deepcopy(net);duplicate.eval()
    with torch.no_grad():
        a=[v.clone() for v in duplicate.encoder(duplicate.backbone(z[:,:3]),z[:,3:])]
        net.encoder.depth_scale.add_(2.)
        for p in net.encoder.spatial_attention.parameters():p.add_(1.)
        b=duplicate.encoder(duplicate.backbone(z[:,:3]),z[:,3:])
    assert all(torch.equal(v,w) for v,w in zip(a,b))
    return {'passed':True,'C_zero_identity':True,'C_gamma_and_spatial_gradients':True,'same_AB_initial_tensors':True,'same_AB_post_model_RNG':True,'same_AB_initial_encoder_outputs':True,'EMA_copy_independent':True,'models':{KEY:meta}}

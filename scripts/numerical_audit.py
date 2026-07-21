from __future__ import annotations
import json,sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import numpy as np
from lagrangian_ellipsoid.core import FlowConfig,SpectralOUFlowFFT,load_npz,khachiyan_mee,shape_obs
RES=ROOT/'data'/'results';RES.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(20260714);cfg=FlowConfig(k_max=192,zeta=1/3,ngrid=768);flow=SpectralOUFlowFFT(cfg,rng);errs=[]
for q in range(5):
    x=rng.uniform(0,2*np.pi,size=(200,2));vi=flow.evaluate_v(x);phase=x@flow.k.astype(float).T;scal=2*(np.cos(phase)*flow.a[None,:]-np.sin(phase)*flow.b[None,:]);vd=scal@flow.perp;errs.append(np.sqrt(np.mean((vi-vd)**2))/np.sqrt(np.mean(vd**2)))
    for _ in range(20):flow.update(rng)
files=sorted((ROOT/'data/raw/decisive_k192').glob('*.npz'))[:4];diff=[]
for p in files:
    d=load_npz(p)
    for pts in d['snapshot_positions']:
        if not np.isfinite(pts).all():continue
        c1,g1,_=khachiyan_mee(pts,tol=1e-5,max_iter=100);c2,g2,_=khachiyan_mee(pts,tol=1e-8,max_iter=3000);o1=shape_obs(g1,np.zeros((2,2)));o2=shape_obs(g2,np.zeros((2,2)));diff.append([abs(o1['sigma']-o2['sigma']),abs(o1['r']/o2['r']-1),np.linalg.norm(c1-c2)])
a=np.asarray(diff);out={'Kmax':192,'ngrid':768,'interpolation_relative_rms_mean':float(np.mean(errs)),'interpolation_relative_rms_max':float(np.max(errs)),'mee_comparisons':len(diff),'mee_sigma_abs_mean':float(np.mean(a[:,0])),'mee_sigma_abs_max':float(np.max(a[:,0])),'mee_r_rel_mean':float(np.mean(a[:,1])),'mee_r_rel_max':float(np.max(a[:,1])),'mee_center_abs_max':float(np.max(a[:,2]))}
with open(RES/'numerical_audit_b.json','w') as f:json.dump(out,f,indent=2)
print(json.dumps(out,indent=2),flush=True)

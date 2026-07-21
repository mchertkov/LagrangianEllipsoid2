from __future__ import annotations
import argparse,json,os,sys
from dataclasses import asdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import numpy as np
from lagrangian_ellipsoid.core import FlowConfig,SpectralOUFlowFFT,seed_ball,khachiyan_mee,covariance_shape,TWOPI
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--outdir',required=True);p.add_argument('--zeta',type=float,default=1/3);p.add_argument('--kmax',type=int,default=128);p.add_argument('--ngrid',type=int,default=512);p.add_argument('--dt',type=float,default=.01);p.add_argument('--T',type=float,default=15);p.add_argument('--N',type=int,default=1000);p.add_argument('--subsets',default='1000');p.add_argument('--r0-uv',type=float,default=1);p.add_argument('--stride',type=int,default=10);p.add_argument('--tau-model',default='turnover');p.add_argument('--mee-tol',type=float,default=1e-4);p.add_argument('--s2ref',type=float,default=.17);p.add_argument('--s2scale',type=float,default=.5)
a=p.parse_args();subs=tuple(int(x) for x in a.subsets.split(','));rng=np.random.default_rng(a.seed);fc=FlowConfig(k_max=a.kmax,zeta=a.zeta,dt=a.dt,tau_model=a.tau_model,ngrid=a.ngrid,s2_ref_scale=a.s2scale,s2_ref_value=a.s2ref);flow=SpectralOUFlowFFT(fc,rng);uv=TWOPI/a.kmax;r0=a.r0_uv*uv;x=seed_ball(a.N,r0,rng);nsteps=int(round(a.T/a.dt));nmax=nsteps//a.stride+2;ns=len(subs);time=np.zeros(nmax);mg=np.zeros((ns,nmax,2,2));cg=np.zeros_like(mg);mc=np.zeros((ns,nmax,2));cc=np.zeros_like(mc);j=0;stop='max_time'
for step in range(nsteps+1):
    rec=step%a.stride==0 or step==nsteps;vel=None
    if rec:
        time[j]=step*a.dt;fullr=fulla=None
        for qi,n in enumerate(subs):
            c,g,h=khachiyan_mee(x[:n],tol=a.mee_tol);c2,g2,C=covariance_shape(x[:n]);mc[qi,j]=c;mg[qi,j]=g;cc[qi,j]=c2;cg[qi,j]=g2
            if n==max(subs):
                ev=np.linalg.eigvalsh(g);fullr=(ev[0]*ev[1])**.25;fulla=np.sqrt(ev[1])
        j+=1
        if fullr>=TWOPI*.125:stop='r_threshold';break
        if fulla>=TWOPI*.25:stop='aplus_threshold';break
    if step<nsteps:
        if vel is None:vel=flow.evaluate_v(x)
        x+=a.dt*vel;flow.update(rng)
out={'config_json':np.asarray(json.dumps({'shape_only':True,'flow':asdict(fc),'T':a.T,'N':a.N,'subsets':subs,'r0_over_uv':a.r0_uv})),'seed':np.asarray(a.seed),'time':time[:j],'subsets':np.asarray(subs),'mee_g':mg[:,:j],'cov_g':cg[:,:j],'mee_center':mc[:,:j],'cov_center':cc[:,:j],'mee_M':np.zeros_like(mg[:,:j]),'cov_M':np.zeros_like(cg[:,:j]),'M_ls':np.zeros_like(mg[:,:j]),'nonaffinity':np.full((ns,j),np.nan),'stop_reason':np.asarray(stop),'uv_scale':np.asarray(uv),'r0':np.asarray(r0)}
outdir=Path(a.outdir);outdir.mkdir(parents=True,exist_ok=True);path=outdir/f'{a.name}_seed{a.seed:02d}.npz';np.savez_compressed(path,**out);print(path,flush=True);sys.stdout.flush();os._exit(0)

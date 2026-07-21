from __future__ import annotations
import argparse
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from lagrangian_ellipsoid.core import FlowConfig,SimConfig,save_run
p=argparse.ArgumentParser()
p.add_argument('--name',required=True); p.add_argument('--seed',type=int,required=True); p.add_argument('--outdir',required=True)
p.add_argument('--zeta',type=float,default=1/3); p.add_argument('--kmax',type=int,default=128); p.add_argument('--ngrid',type=int,default=512)
p.add_argument('--dt',type=float,default=.01); p.add_argument('--T',type=float,default=15); p.add_argument('--N',type=int,default=1000); p.add_argument('--subsets',default='1000')
p.add_argument('--r0-uv',type=float,default=1.); p.add_argument('--stride',type=int,default=5); p.add_argument('--tau-model',default='turnover',choices=['turnover','constant','short'])
p.add_argument('--mee-tol',type=float,default=1e-5); p.add_argument('--s2ref',type=float,default=.17); p.add_argument('--s2scale',type=float,default=.5)
a=p.parse_args(); subs=tuple(int(x) for x in a.subsets.split(','))
flow=FlowConfig(k_max=a.kmax,zeta=a.zeta,dt=a.dt,tau_model=a.tau_model,ngrid=a.ngrid,s2_ref_scale=a.s2scale,s2_ref_value=a.s2ref)
cfg=SimConfig(flow=flow,max_time=a.T,n_particles=a.N,subsets=subs,r0_over_uv=a.r0_uv,record_stride=a.stride,mee_tol=a.mee_tol)
out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True); path=out/f'{a.name}_seed{a.seed:02d}.npz'; save_run(path,cfg,a.seed); print(path,flush=True); sys.stdout.flush(); sys.stderr.flush(); os._exit(0)

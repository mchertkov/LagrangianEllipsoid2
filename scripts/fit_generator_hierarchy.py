from __future__ import annotations
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 14.0,
    "axes.titlesize": 15.0,
    "axes.labelsize": 15.0,
    "xtick.labelsize": 13.0,
    "ytick.labelsize": 13.0,
    "legend.fontsize": 12.0,
    "lines.linewidth": 2.0,
    "lines.markersize": 7.0,
})
from lagrangian_ellipsoid.core import load_npz,derive

FIG=ROOT/'figures'/'controls'; RES=ROOT/'data'/'results'; FIG.mkdir(parents=True,exist_ok=True); RES.mkdir(parents=True,exist_ok=True)
files=sorted((ROOT/'data/raw/decisive_k192').glob('*.npz')); runs=[load_npz(p) for p in files]
if len(runs)<8: raise RuntimeError('Need at least 8 baseline runs')
train=runs[:16]; test=runs[16:24]


def state(d):
    o=derive(d,'mee'); r0=float(d['r0'])
    return {'v':np.log(o['r']/r0),'sigma':o['sigma'],'z':np.column_stack([o['q'],o['p'],o['omega']]),'time':o['time']}
tr=[state(d) for d in train]; te=[state(d) for d in test]
allv=np.concatenate([x['v'][:-1] for x in tr]); edges=np.quantile(allv,np.linspace(0,1,6)); edges[0]-=1e-9; edges[-1]+=1e-9
nb=len(edges)-1


def ridge(X,Y,lam=1e-3):
    X=np.asarray(X,float); Y=np.asarray(Y,float)
    scale=np.std(X,axis=0); scale[0]=1.; scale[scale<1e-10]=1.
    Xs=X/scale
    R=np.eye(X.shape[1]); R[0,0]=0
    l=lam*np.trace(Xs.T@Xs)/max(X.shape[1],1)
    B=np.linalg.solve(Xs.T@Xs+l*R,Xs.T@Y)
    return B/scale[:,None]


def collect(data,lag):
    out=[]
    for s in data:
        dt=float(s['time'][lag]-s['time'][0]); n=len(s['time'])-lag
        qavg=np.zeros(n)
        w=np.ones(lag+1); w[[0,-1]]=0.5; h=s['time'][1]-s['time'][0]
        for j,ww in enumerate(w): qavg+=ww*s['z'][j:j+n,0]
        qavg*=h/dt
        out.append({'v':s['v'][:-lag],'sig':s['sigma'][:-lag],'z':s['z'][:-lag],
                    'rsig':(s['sigma'][lag:]-s['sigma'][:-lag])/dt-qavg,
                    'rz':(s['z'][lag:]-s['z'][:-lag])/dt,'dt':dt,
                    'dsig':s['sigma'][lag:]-s['sigma'][:-lag],'dz':s['z'][lag:]-s['z'][:-lag]})
    return out


def fit_model(data,kind='M2',lag=1):
    c=collect(data,lag); dt=c[0]['dt']
    params=[]
    for b in range(nb):
        rows=[]
        for x in c:
            sel=(x['v']>=edges[b])&(x['v']<edges[b+1])
            for key in ('v','sig','z','rsig','rz'): pass
            if np.any(sel): rows.append({k:(v[sel] if isinstance(v,np.ndarray) and len(v)==len(sel) else v) for k,v in x.items()})
        if not rows:
            params.append(None); continue
        sig=np.concatenate([r['sig'] for r in rows]); z=np.concatenate([r['z'] for r in rows]); rs=np.concatenate([r['rsig'] for r in rows]); rz=np.concatenate([r['rz'] for r in rows])
        if kind in ('M0','M1'):
            ZB=np.zeros((5,3)); zpred=np.zeros_like(rz)
            for j in range(3):
                cols=[0,2+j]
                Xj=np.column_stack([np.ones(len(z)),z[:,j]])
                B=ridge(Xj,rz[:,j,None]).ravel()
                if kind=='M0' and j==0: B[0]=0.0
                ZB[cols,j]=B; zpred[:,j]=Xj@B
            SB=np.array([np.mean(rs),0.0])
            rspred=np.full_like(rs,SB[0])
        elif kind=='M2':
            # Sparse physical coupling: q responds to sigma; transverse strain and
            # vorticity form a coupled rotation block. The residual is affine in sigma.
            ZB=np.zeros((5,3)); zpred=np.zeros_like(rz)
            selections=([0,1,2],[0,3,4],[0,3,4])
            Xfull=np.column_stack([np.ones(len(z)),sig,z])
            for j,cols in enumerate(selections):
                B=ridge(Xfull[:,cols],rz[:,j,None]).ravel(); ZB[list(cols),j]=B; zpred[:,j]=Xfull[:,cols]@B
            SX=np.column_stack([np.ones(len(sig)),sig]); SB=ridge(SX,rs[:,None]).ravel(); rspred=SX@SB
        else:
            X=np.column_stack([np.ones(len(z)),sig,z]); ZB=ridge(X,rz,lam=1e-2); zpred=X@ZB
            SX=np.column_stack([np.ones(len(sig)),sig]); SB=ridge(SX,rs[:,None],lam=1e-2).ravel(); rspred=SX@SB
        ez=rz-zpred; es=rs-rspred
        Qz=np.cov(ez,rowvar=False,bias=False)*dt; Qs=float(np.var(es,ddof=1)*dt)
        # regularize diffusion
        wv,V=np.linalg.eigh(0.5*(Qz+Qz.T)); wv=np.maximum(wv,1e-8*np.max(wv) if np.max(wv)>0 else 1e-10); Qz=(V*wv)@V.T
        params.append({'ZB':ZB,'SB':SB,'Qz':Qz,'Qs':max(Qs,1e-10),'n':len(sig)})
    # fill any empty bins from nearest
    for b in range(nb):
        if params[b] is None:
            j=min((j for j in range(nb) if params[j] is not None),key=lambda j:abs(j-b)); params[b]=params[j]
    return {'kind':kind,'lag':lag,'dt':dt,'params':params}


def ibin(v): return int(np.clip(np.searchsorted(edges,v,side='right')-1,0,nb-1))

def drift(model,v,sig,z):
    p=model['params'][ibin(v)]
    x=np.r_[1.,sig,z]
    dz=x@p['ZB']
    rs=p['SB'][0]+p['SB'][1]*sig
    return rs,dz,p


def nll(model,data):
    vals=[]; rm=[]; innov=[]
    lag=model['lag']; dt=model['dt']
    for s in data:
        for t in range(len(s['time'])-lag):
            v=s['v'][t]; sig=s['sigma'][t]; z=s['z'][t]
            # Predict from the current state only; do not use future q in held-out scoring.
            rs,dz,p=drift(model,v,sig,z)
            mean=np.r_[(z[0]+rs)*dt,dz*dt]; obs=np.r_[s['sigma'][t+lag]-sig,s['z'][t+lag]-z]; e=obs-mean
            C=np.zeros((4,4)); C[0,0]=p['Qs']*dt; C[1:,1:]=p['Qz']*dt; C+=1e-10*np.eye(4)
            inv=np.linalg.inv(C); sign,ld=np.linalg.slogdet(C); vals.append(.5*(ld+e@inv@e+4*np.log(2*np.pi))); rm.append(e)
            innov.append(e/np.sqrt(np.maximum(np.diag(C),1e-20)))
    return float(np.mean(vals)),float(np.sqrt(np.mean(np.asarray(rm)**2))),np.asarray(innov)


def rollout(model,s,nsim=300,seed=123):
    """Vectorized Monte Carlo rollout for a shared empirical scale path."""
    rng=np.random.default_rng(seed); dt=float(s['time'][1]-s['time'][0]); nt=len(s['time'])
    sig=np.full(nsim,s['sigma'][0]); z=np.tile(s['z'][0],(nsim,1)); sh=np.zeros((nsim,nt)); zh=np.zeros((nsim,nt,3)); sh[:,0]=sig; zh[:,0]=z
    for t in range(nt-1):
        p=model['params'][ibin(s['v'][t])]
        X=np.column_stack([np.ones(nsim),sig,z])
        dz=X@p['ZB']
        rs=p['SB'][0]+p['SB'][1]*sig
        wz=rng.multivariate_normal(np.zeros(3),p['Qz']*dt,size=nsim)
        ws=np.sqrt(p['Qs']*dt)*rng.standard_normal(nsim)
        sig=np.abs(sig+(z[:,0]+rs)*dt+ws)
        z=z+dz*dt+wz
        sh[:,t+1]=sig; zh[:,t+1]=z
    return sh,zh

models={k:fit_model(tr,k,1) for k in ('M0','M1','M2','M3')}
scores={}; roll={}
for k,m in models.items():
    nn,rr,inn=nll(m,te); scores[k]={'one_step_nll':nn,'one_step_rmse':rr}
    # innovation lag-one autocorrelation by component
    scores[k]['innovation_lag1_corr']=[float(np.corrcoef(inn[:-1,j],inn[1:,j])[0,1]) for j in range(4)]
    simsig=[]; simz=[]
    for i,s in enumerate(te):
        sh,zh=rollout(m,s,nsim=250,seed=1000+i); simsig.append(sh); simz.append(zh)
    roll[k]=(simsig,simz)
    # path mean sigma RMSE
    er=[]
    for s,sh in zip(te,simsig): er.extend((np.mean(sh,0)-s['sigma'])**2)
    scores[k]['rollout_mean_sigma_rmse']=float(np.sqrt(np.mean(er)))

# Lag sensitivity for M2
lag_scores={}
for lag in (1,2,4):
    m=fit_model(tr,'M2',lag); nn,rr,inn=nll(m,te); lag_scores[str(lag)]={'delta':m['dt'],'nll':nn,'rmse':rr,'innovation_lag1_corr':[float(np.corrcoef(inn[:-1,j],inn[1:,j])[0,1]) for j in range(4)]}

# Conditional validation curves on test data and rollouts, common v bins.
cent=.5*(edges[:-1]+edges[1:]); empirical=np.full(nb,np.nan); empirical_q=np.full(nb,np.nan); count=np.zeros(nb,int)
for b in range(nb):
    sv=[]; qv=[]
    for s in te:
        q=(s['v']>=edges[b])&(s['v']<edges[b+1]); sv.extend(s['sigma'][q]); qv.extend(s['z'][q,0])
    if sv: empirical[b]=np.mean(sv); empirical_q[b]=np.mean(qv); count[b]=len(sv)
model_curves={}
for k,(ss,zz) in roll.items():
    ms=np.full(nb,np.nan); mq=np.full(nb,np.nan)
    for b in range(nb):
        av=[]; aq=[]
        for s,sh,zh in zip(te,ss,zz):
            q=(s['v']>=edges[b])&(s['v']<edges[b+1]); av.extend(sh[:,q].ravel()); aq.extend(zh[:,q,0].ravel())
        if av: ms[b]=np.mean(av); mq[b]=np.mean(aq)
    model_curves[k]={'sigma':ms,'q':mq}

summary={'n_train':len(tr),'n_test':len(te),'train_files':[p.name for p in files[:len(tr)]],'test_files':[p.name for p in files[len(tr):]],'v_edges':edges.tolist(),'scores':scores,'lag_sensitivity':lag_scores}
with open(RES/'intrinsic_model_b_summary.json','w') as f: json.dump(summary,f,indent=2)
# coefficient table
rows=[]
for k,m in models.items():
    for b,p in enumerate(m['params']): rows.append([k,b,edges[b],edges[b+1],p['n'],*np.ravel(p['SB']),*np.ravel(p['ZB'])])
maxlen=max(map(len,rows)); arr=np.full((len(rows),maxlen),np.nan,dtype=object)
for i,row in enumerate(rows): arr[i,:len(row)]=row
np.savez_compressed(RES/'intrinsic_model_b_coefficients.npz',edges=edges,rows=arr)

# Plots
fig,ax=plt.subplots(figsize=(7.2,5.1)); ax.plot(np.exp(cent),empirical,marker='o',label='held-out empirical')
for k in ('M0','M1','M2','M3'): ax.plot(np.exp(cent),model_curves[k]['sigma'],marker='o',label=k)
ax.set_xscale('log'); ax.set_xlabel(r'$r_{\rm MEE}/r_0$'); ax.set_ylabel(r'$\langle\sigma\mid r\rangle$'); ax.legend(); fig.tight_layout(); fig.savefig(FIG/'B4_intrinsic_hierarchy_sigma.pdf'); fig.savefig(FIG/'B4_intrinsic_hierarchy_sigma.png',dpi=220); plt.close(fig)
fig,ax=plt.subplots(figsize=(7.2,5.1)); ax.plot(np.exp(cent),empirical_q,marker='o',label='held-out empirical')
for k in ('M0','M1','M2','M3'): ax.plot(np.exp(cent),model_curves[k]['q'],marker='o',label=k)
ax.set_xscale('log'); ax.set_xlabel(r'$r_{\rm MEE}/r_0$'); ax.set_ylabel(r'$\langle q\mid r\rangle$'); ax.legend(); fig.tight_layout(); fig.savefig(FIG/'B4_intrinsic_hierarchy_source.pdf'); fig.savefig(FIG/'B4_intrinsic_hierarchy_source.png',dpi=220); plt.close(fig)
fig,ax=plt.subplots(figsize=(7.0,5.0)); xx=np.arange(4); vals=[scores[k]['one_step_nll'] for k in ('M0','M1','M2','M3')]; ax.bar(xx,vals); ax.set_xticks(xx,['M0','M1','M2','M3']); ax.set_ylabel('held-out one-step negative log score'); fig.tight_layout(); fig.savefig(FIG/'B4_intrinsic_hierarchy_scores.pdf'); fig.savefig(FIG/'B4_intrinsic_hierarchy_scores.png',dpi=220); plt.close(fig)
fig,ax=plt.subplots(figsize=(7.0,5.0)); xx=np.arange(4); vals=[scores[k]['rollout_mean_sigma_rmse'] for k in ('M0','M1','M2','M3')]; ax.bar(xx,vals); ax.set_xticks(xx,['M0','M1','M2','M3']); ax.set_ylabel(r'held-out rollout RMSE of mean $\sigma(t)$'); fig.tight_layout(); fig.savefig(FIG/'B4_intrinsic_hierarchy_rollout.pdf'); fig.savefig(FIG/'B4_intrinsic_hierarchy_rollout.png',dpi=220); plt.close(fig)

print(json.dumps(summary,indent=2),flush=True); sys.stdout.flush(); os._exit(0)

from __future__ import annotations
import json,csv,sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import numpy as np
import matplotlib.pyplot as plt
from lagrangian_ellipsoid.core import load_npz,derive
RES=ROOT/'data'/'results';FIG=ROOT/'figures'/'controls';RES.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
files=sorted((ROOT/'data/raw/decisive_k192').glob('*.npz'));runs=[load_npz(p) for p in files]
if len(runs)<20: raise RuntimeError(f'Need >=20 runs, found {len(runs)}')
train=runs[:16];test=runs[16:24]

def state(d):
    E=derive(d,'mee');L=derive(d,'mee',-1,'M_ls');r0=float(d['r0'])
    e=np.column_stack([E['q'],E['p'],E['omega']]);ls=np.column_stack([L['q'],L['p'],L['omega']])
    return {'v':np.log(E['r']/r0),'sig':E['sigma'],'e':e,'d':ls-e,'ls':ls,'time':E['time']}
tr=[state(d) for d in train];te=[state(d) for d in test]
allv=np.concatenate([s['v'][:-1] for s in tr]);edges=np.quantile(allv,np.linspace(0,1,7));edges[0]-=1e-9;edges[-1]+=1e-9;nb=len(edges)-1

def ibin(v):return int(np.clip(np.searchsorted(edges,v,side='right')-1,0,nb-1))
def ridge(X,Y,lam=2e-3):
    X=np.asarray(X,float);Y=np.asarray(Y,float);sc=np.std(X,0);sc[0]=1.;sc[sc<1e-10]=1.;Xs=X/sc;R=np.eye(X.shape[1]);R[0,0]=0;l=lam*np.trace(Xs.T@Xs)/max(X.shape[1],1);B=np.linalg.solve(Xs.T@Xs+l*R,Xs.T@Y);return B/sc[:,None]
def poscov(C):
    C=np.atleast_2d(C);w,V=np.linalg.eigh(.5*(C+C.T));mx=max(float(np.max(w)),1e-12);w=np.maximum(w,1e-7*mx);return (V*w)@V.T

def collect(data,lag):
    rows=[]
    for s in data:
        dt=s['time'][lag]-s['time'][0];n=len(s['time'])-lag
        rows.append({'v':s['v'][:-lag],'sig':s['sig'][:-lag],'e':s['e'][:-lag],'d':s['d'][:-lag],'ls':s['ls'][:-lag],
                     'de':(s['e'][lag:]-s['e'][:-lag])/dt,'dd':(s['d'][lag:]-s['d'][:-lag])/dt,'dls':(s['ls'][lag:]-s['ls'][:-lag])/dt,
                     'ds':(s['sig'][lag:]-s['sig'][:-lag])/dt,'dt':float(dt)})
    return rows

def fit(kind,lag=1):
    rows=collect(tr,lag);dt=rows[0]['dt'];pars=[]
    for b in range(nb):
        chunks=[]
        for r in rows:
            q=(r['v']>=edges[b])&(r['v']<edges[b+1])
            if q.any():chunks.append({k:(v[q] if isinstance(v,np.ndarray) and len(v)==len(q) else v) for k,v in r.items()})
        sig=np.concatenate([r['sig'] for r in chunks]);e=np.concatenate([r['e'] for r in chunks]);d=np.concatenate([r['d'] for r in chunks]);ls=e+d
        de=np.concatenate([r['de'] for r in chunks]);dd=np.concatenate([r['dd'] for r in chunks]);dls=de+dd;ds=np.concatenate([r['ds'] for r in chunks])
        if kind=='E':
            state=e;resp=de;X=np.column_stack([np.ones(len(e)),sig,e]);sels=([0,1,2],[0,3,4],[0,3,4]);B=np.zeros((5,3));pred=np.zeros_like(resp)
            for j,cols in enumerate(sels):bb=ridge(X[:,cols],resp[:,j,None]).ravel();B[list(cols),j]=bb;pred[:,j]=X[:,cols]@bb
            source=e[:,0];Sx=np.column_stack([np.ones(len(sig)),sig]);SB=ridge(Sx,(ds-source)[:,None]).ravel();sp=Sx@SB;extra={}
        elif kind=='LS':
            state=ls;resp=dls;X=np.column_stack([np.ones(len(ls)),sig,ls]);sels=([0,1,2],[0,3,4],[0,3,4]);B=np.zeros((5,3));pred=np.zeros_like(resp)
            for j,cols in enumerate(sels):bb=ridge(X[:,cols],resp[:,j,None]).ravel();B[list(cols),j]=bb;pred[:,j]=X[:,cols]@bb
            source=ls[:,0];Sx=np.column_stack([np.ones(len(sig)),sig]);SB=ridge(Sx,(ds-source)[:,None]).ravel();sp=Sx@SB;extra={}
        elif kind=='TC_static':
            # Dynamic M_E plus an algebraic conditional tensor corrector.
            X=np.column_stack([np.ones(len(e)),sig,e]);sels=([0,1,2],[0,3,4],[0,3,4]);B=np.zeros((5,3));pred=np.zeros_like(de)
            for j,cols in enumerate(sels):bb=ridge(X[:,cols],de[:,j,None]).ravel();B[list(cols),j]=bb;pred[:,j]=X[:,cols]@bb
            DB=ridge(X,d,lam=5e-3);dhat=X@DB;Dc=poscov(np.cov(d-dhat,rowvar=False))
            source=e[:,0]+dhat[:,0];Sx=np.column_stack([np.ones(len(sig)),sig]);SB=ridge(Sx,(ds-source)[:,None]).ravel();sp=Sx@SB;state=e;resp=de;extra={'DB':DB,'Dc':Dc}
        else:
            # Joint dynamics of the averaged gradient and tensor corrector.
            state=np.column_stack([e,d]);resp=np.column_stack([de,dd]);X=np.column_stack([np.ones(len(e)),sig,e,d]);B=np.zeros((8,6));pred=np.zeros_like(resp)
            if kind=='TC_joint_sparse':
                sels=([0,1,2,5],[0,3,4,6,7],[0,3,4,6,7],[0,1,2,5],[0,3,4,6,7],[0,3,4,6,7])
                for j,cols in enumerate(sels):bb=ridge(X[:,cols],resp[:,j,None],lam=5e-3).ravel();B[list(cols),j]=bb;pred[:,j]=X[:,cols]@bb
            else:
                B=ridge(X,resp,lam=2e-2);pred=X@B
            source=e[:,0]+d[:,0];Sx=np.column_stack([np.ones(len(sig)),sig]);SB=ridge(Sx,(ds-source)[:,None]).ravel();sp=Sx@SB;extra={}
        er=resp-pred;Q=poscov(np.cov(er,rowvar=False)*dt);es=ds-source-sp;Qs=max(float(np.var(es,ddof=1)*dt),1e-10)
        pars.append({'B':B,'Q':Q,'SB':SB,'Qs':Qs,'n':len(sig),**extra})
    return {'kind':kind,'lag':lag,'dt':dt,'pars':pars}

def mean_corrector(p,sig,e):
    x=np.r_[1.,sig,e];return x@p['DB']
def drift(m,v,sig,x):
    p=m['pars'][ibin(v)];kind=m['kind']
    if kind in ('E','LS','TC_static'):xx=np.r_[1.,sig,x];dx=xx@p['B']
    else:xx=np.r_[1.,sig,x];dx=xx@p['B']
    rs=p['SB'][0]+p['SB'][1]*sig
    if kind=='E':src=x[0]
    elif kind=='LS':src=x[0]
    elif kind=='TC_static':src=x[0]+mean_corrector(p,sig,x)[0]
    else:src=x[0]+x[3]
    return dx,src+rs,p

def initial_x(s,kind):
    if kind=='E' or kind=='TC_static':return s['e'][0].copy()
    if kind=='LS':return s['ls'][0].copy()
    return np.r_[s['e'][0],s['d'][0]]
def empirical_x(s,kind):
    if kind in ('E','TC_static'):return s['e']
    if kind=='LS':return s['ls']
    return np.column_stack([s['e'],s['d']])

def score_sigma(m,data):
    vals=[];errs=[];innov=[];xerrs=[];correrrs=[];lag=m['lag'];dt=m['dt']
    for s in data:
        xx=empirical_x(s,m['kind'])
        for t in range(len(s['time'])-lag):
            dx,ds,p=drift(m,s['v'][t],s['sig'][t],xx[t]);obs=s['sig'][t+lag]-s['sig'][t];mu=ds*dt;var=p['Qs']*dt;e=obs-mu
            vals.append(.5*(np.log(2*np.pi*var)+e*e/var));errs.append(e);innov.append(e/np.sqrt(var))
            xobs=xx[t+lag]-xx[t];xerrs.extend((xobs-dx*dt).ravel())
            if m['kind']=='TC_static':correrrs.extend((s['d'][t]-mean_corrector(p,s['sig'][t],s['e'][t])).ravel())
            elif m['kind'].startswith('TC_joint'):correrrs.extend((xobs[3:]-dx[3:]*dt).ravel())
    inn=np.asarray(innov);return {'sigma_nll':float(np.mean(vals)),'sigma_one_step_rmse':float(np.sqrt(np.mean(np.asarray(errs)**2))),
                                  'state_increment_rmse':float(np.sqrt(np.mean(np.asarray(xerrs)**2))),
                                  'sigma_innovation_lag1':float(np.corrcoef(inn[:-1],inn[1:])[0,1]),
                                  'corrector_rmse':float(np.sqrt(np.mean(np.asarray(correrrs)**2))) if correrrs else np.nan}

def rollout(m,s,nsim=250,seed=0):
    # v2: vectorized over nsim (all simulations share the empirical v(t), hence the
    # same scale bin at each t, so the per-bin linear drift applies as one matmul).
    rng=np.random.default_rng(seed);dt=s['time'][1]-s['time'][0];nt=len(s['time']);kind=m['kind']
    x=np.tile(initial_x(s,kind),(nsim,1));dim=x.shape[1];sig=np.full(nsim,s['sig'][0]);sh=np.zeros((nsim,nt));sh[:,0]=sig
    for t in range(nt-1):
        p=m['pars'][ibin(s['v'][t])]
        X=np.column_stack([np.ones(nsim),sig,x]);dx=X@p['B']
        if kind=='TC_static':src=x[:,0]+(X@p['DB'])[:,0]
        elif kind.startswith('TC_joint'):src=x[:,0]+x[:,3]
        else:src=x[:,0]
        ds=src+p['SB'][0]+p['SB'][1]*sig
        Lc=np.linalg.cholesky(p['Q']*dt+1e-14*np.eye(dim))
        x=x+dx*dt+rng.standard_normal((nsim,dim))@Lc.T
        sig=np.abs(sig+ds*dt+np.sqrt(p['Qs']*dt)*rng.standard_normal(nsim))
        sh[:,t+1]=sig
    return sh

def eval_roll(m):
    ss=[];er=[]
    for i,s in enumerate(te):
        sh=rollout(m,s,250,4000+i);ss.append(sh);er.extend((np.mean(sh,0)-s['sig'])**2)
    return float(np.sqrt(np.mean(er))),ss

def ck_test(kind):
    m1=fit(kind,1);m2=fit(kind,2);diff=[];err=[];scale=[]
    for s in te:
        xx=empirical_x(s,kind);dt=s['time'][1]-s['time'][0]
        for t in range(len(s['time'])-2):
            x0=xx[t];sg=s['sig'][t];dx1,ds1,p=drift(m1,s['v'][t],sg,x0);x1=x0+dx1*dt;sg1=abs(sg+ds1*dt);dx2,ds2,p=drift(m1,s['v'][t+1],sg1,x1);mcomp=sg1+ds2*dt
            dxd,dsd,p=drift(m2,s['v'][t],sg,x0);mdir=sg+dsd*(2*dt);obs=s['sig'][t+2]
            diff.append(mcomp-mdir);err.append(mcomp-obs);scale.append(obs-sg)
    return {'composed_vs_direct_rmse':float(np.sqrt(np.mean(np.asarray(diff)**2))),'composed_vs_empirical_rmse':float(np.sqrt(np.mean(np.asarray(err)**2))),
            'normalized_CK_discrepancy':float(np.sqrt(np.mean(np.asarray(diff)**2))/np.std(scale))}

kinds=['E','LS','TC_static','TC_joint_sparse','TC_joint_full'];models={k:fit(k,1) for k in kinds};summary={'n_train':len(tr),'n_test':len(te),'train_files':[p.name for p in files[:16]],'test_files':[p.name for p in files[16:24]],'v_edges':edges.tolist(),'models':{}}
rolls={}
for k,m in models.items():
    sc=score_sigma(m,te);rm,ss=eval_roll(m);sc['rollout_mean_sigma_rmse']=rm;sc['CK']=ck_test(k);summary['models'][k]=sc;rolls[k]=ss
# Lag sensitivity for preferred joint sparse
summary['lag_sensitivity']={}
for lag in (1,2,4):summary['lag_sensitivity'][str(lag)]=score_sigma(fit('TC_joint_sparse',lag),te)
with open(RES/'tensor_corrector_model_summary.json','w') as f:json.dump(summary,f,indent=2)
# coefficient archive
np.savez_compressed(RES/'tensor_corrector_coefficients.npz',edges=edges,**{f'{k}_B':np.stack([p['B'] for p in m['pars']]) for k,m in models.items()},**{f'{k}_Q':np.stack([p['Q'] for p in m['pars']]) for k,m in models.items()},**{f'{k}_SB':np.stack([p['SB'] for p in m['pars']]) for k,m in models.items()})
with open(RES/'tensor_corrector_model_scores.csv','w',newline='') as f:
    w=csv.writer(f);w.writerow(['model','sigma_nll','sigma_one_step_rmse','state_increment_rmse','innovation_lag1','corrector_rmse','rollout_sigma_rmse','CK_normalized'])
    for k in kinds:
        s=summary['models'][k];w.writerow([k,s['sigma_nll'],s['sigma_one_step_rmse'],s['state_increment_rmse'],s['sigma_innovation_lag1'],s['corrector_rmse'],s['rollout_mean_sigma_rmse'],s['CK']['normalized_CK_discrepancy']])
# heldout conditional curves
cent=.5*(edges[:-1]+edges[1:]);emp=np.full(nb,np.nan);cur={k:np.full(nb,np.nan) for k in kinds}
for b in range(nb):
    ee=[];vals={k:[] for k in kinds}
    for i,s in enumerate(te):
        q=(s['v']>=edges[b])&(s['v']<edges[b+1]);ee.extend(s['sig'][q]);
        for k in kinds:vals[k].extend(rolls[k][i][:,q].ravel())
    if ee:
        emp[b]=np.mean(ee)
        for k in kinds:cur[k][b]=np.mean(vals[k])
fig,ax=plt.subplots(figsize=(6.7,4.6));ax.plot(np.exp(cent),emp,marker='o',label='held-out empirical')
for k in kinds:ax.plot(np.exp(cent),cur[k],marker='o',label=k)
ax.set_xscale('log');ax.set_xlabel(r'$r_{MEE}/r_0$');ax.set_ylabel(r'$\langle\sigma\mid r\rangle$');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(FIG/'B4_tensor_corrector_sigma.pdf');fig.savefig(FIG/'B4_tensor_corrector_sigma.png',dpi=200);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4.4));x=np.arange(len(kinds));v=[summary['models'][k]['rollout_mean_sigma_rmse'] for k in kinds];ax.bar(x,v);ax.set_xticks(x,kinds,rotation=20);ax.set_ylabel(r'held-out rollout RMSE of mean $\sigma(t)$');fig.tight_layout();fig.savefig(FIG/'B4_tensor_corrector_scores.pdf');fig.savefig(FIG/'B4_tensor_corrector_scores.png',dpi=200);plt.close(fig)
print(json.dumps(summary,indent=2),flush=True)

from __future__ import annotations
import json, csv, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, spearmanr
from lagrangian_ellipsoid.core import load_npz, derive, finite_lag_components

RES=ROOT/'data'/'results'; FIG=ROOT/'figures'/'controls'; RES.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)


def loadglob(path): return [load_npz(p) for p in sorted((ROOT/'data').glob(path))]
def sem(x):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    return float(np.std(x,ddof=1)/np.sqrt(len(x))) if len(x)>1 else np.nan

def ci95(x):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if len(x)<2:return (np.nan,np.nan)
    m=np.mean(x); h=1.96*np.std(x,ddof=1)/np.sqrt(len(x)); return float(m-h),float(m+h)

def seed_metric(d, lo, hi, subset=-1):
    me=derive(d,'mee',subset); ma=derive(d,'cov',subset); rr=me['r']/float(d['r0']); q=(rr>=lo)&(rr<=hi)
    if q.sum()<5:
        return dict(beta_mee=np.nan,beta_mass=np.nan,mee=np.nan,mass=np.nan,delta_sigma=np.nan,delta_rho=np.nan,n=int(q.sum()),max_ratio=float(rr.max()))
    x=np.log(rr[q])
    return dict(beta_mee=float(np.polyfit(x,me['sigma'][q],1)[0]),beta_mass=float(np.polyfit(x,ma['sigma'][q],1)[0]),
                mee=float(np.mean(me['sigma'][q])),mass=float(np.mean(ma['sigma'][q])),
                delta_sigma=float(np.mean(ma['sigma'][q]-me['sigma'][q])),delta_rho=float(np.mean(np.log(me['r'][q]/ma['r'][q]))),
                n=int(q.sum()),max_ratio=float(rr.max()))

def aggregate_metrics(runs,lo,hi,subset=-1):
    vals=[seed_metric(d,lo,hi,subset) for d in runs]
    out={'n_runs':len(runs),'range':[lo,hi],'n_covered':int(sum(np.isfinite(v['beta_mee']) for v in vals)),'per_seed':vals}
    for k in ['beta_mee','beta_mass','mee','mass','delta_sigma','delta_rho','max_ratio']:
        a=[v[k] for v in vals]; out[k+'_mean']=float(np.nanmean(a)); out[k+'_sem']=sem(a); out[k+'_ci95']=ci95(a)
    return out

def binned_curves(runs,edges,subset=-1):
    nb=len(edges)-1; centers=np.sqrt(edges[:-1]*edges[1:]); keys=['mee','mass','delta_sigma','delta_rho']
    arr={k:np.full((len(runs),nb),np.nan) for k in keys}
    samples={k:[[] for _ in range(nb)] for k in ('mee','mass')}
    for i,d in enumerate(runs):
        me=derive(d,'mee',subset); ma=derive(d,'cov',subset); rr=me['r']/float(d['r0']); ib=np.digitize(rr,edges)-1
        for b in range(nb):
            q=ib==b
            if q.sum()>=2:
                arr['mee'][i,b]=np.mean(me['sigma'][q]);arr['mass'][i,b]=np.mean(ma['sigma'][q])
                arr['delta_sigma'][i,b]=np.mean(ma['sigma'][q]-me['sigma'][q]);arr['delta_rho'][i,b]=np.mean(np.log(me['r'][q]/ma['r'][q]))
                samples['mee'][b].extend(me['sigma'][q].tolist());samples['mass'][b].extend(ma['sigma'][q].tolist())
    out={'centers':centers,'edges':edges,'per_seed':arr,'samples':samples}
    for k,a in arr.items():
        out[k+'_mean']=np.nanmean(a,axis=0); out[k+'_sem']=np.nanstd(a,axis=0,ddof=1)/np.sqrt(np.sum(np.isfinite(a),axis=0));out[k+'_n']=np.sum(np.isfinite(a),axis=0)
    return out

def paired_blocks(runs,a=(4,8),b=(8,12)):
    vals=[]
    for d in runs:
        me=derive(d,'mee');ma=derive(d,'cov');rr=me['r']/float(d['r0']);qa=(rr>=a[0])&(rr<a[1]);qb=(rr>=b[0])&(rr<=b[1])
        vals.append({'mee_a':np.mean(me['sigma'][qa]) if qa.sum()>=3 else np.nan,'mee_b':np.mean(me['sigma'][qb]) if qb.sum()>=3 else np.nan,
                     'mass_a':np.mean(ma['sigma'][qa]) if qa.sum()>=3 else np.nan,'mass_b':np.mean(ma['sigma'][qb]) if qb.sum()>=3 else np.nan})
    out={'blocks':[list(a),list(b)],'per_seed':vals}
    for kind in ('mee','mass'):
        diff=np.asarray([v[kind+'_b']-v[kind+'_a'] for v in vals]);out[kind+'_late_minus_early_mean']=float(np.nanmean(diff));out[kind+'_late_minus_early_sem']=sem(diff);out[kind+'_n']=int(np.sum(np.isfinite(diff)))
    return out

def distribution_stability(curves, bins=(3,4,5)):
    out={}
    for kind in ('mee','mass'):
        ss=[np.asarray(curves['samples'][kind][b],float) for b in bins]
        pooled=np.concatenate([x for x in ss if len(x)]);scale=np.std(pooled) if len(pooled)>1 else np.nan
        pairs=[]
        for i in range(len(ss)):
            for j in range(i+1,len(ss)):
                if len(ss[i]) and len(ss[j]):pairs.append(float(wasserstein_distance(ss[i],ss[j])/scale))
        out[kind+'_normalized_W1_pairwise']=pairs;out[kind+'_normalized_W1_mean']=float(np.mean(pairs)) if pairs else np.nan
    return out

# Data groups
b1=loadglob('raw/decisive_k192/*.npz')
rough={
    .25:loadglob('raw/roughness_k128/rough025*.npz'),
    1/3:loadglob('raw/roughness_k128/rough13_k128*.npz'),
    .5:loadglob('raw/roughness_k128/rough050*.npz'),
    2/3:loadglob('raw/roughness_k128/rough067*.npz'),
}
k64=loadglob('raw/controls_k64/*.npz')
r0half=loadglob('raw/controls_r0/*r0half*.npz'); r0double=loadglob('raw/controls_r0/*r0double*.npz')
memconst=loadglob('raw/controls_memory/*memconst*.npz'); memshort=loadglob('raw/controls_memory/*memshort*.npz')
nested=loadglob('raw/controls_N/*.npz')

summary={'inventory':{'B1_k192':len(b1),'roughness':{str(k):len(v) for k,v in rough.items()},'k64':len(k64),'r0half':len(r0half),'r0double':len(r0double),'memconst':len(memconst),'memshort':len(memshort),'nestedN':len(nested)}}

# B1 decisive metrics
summary['decisive_k192']={}
for rg in ((3,8),(4,12),(6,16)):
    summary['decisive_k192'][f'slopes_{rg[0]}_{rg[1]}']=aggregate_metrics(b1,*rg)
edges=np.geomspace(1,20,13); c1=binned_curves(b1,edges);summary['decisive_k192']['paired_plateau']=paired_blocks(b1);summary['decisive_k192']['distribution_stability']=distribution_stability(c1,bins=(5,6,7))

# Residual decomposition for B1 at multiple lags and central range 3-12
summary['residual_decomposition']={}
for lag in (1,2,4):
    rows=[]
    for d in b1:
        f=finite_lag_components(d,lag=lag); rr=f['r']/float(d['r0']);q=(rr>=3)&(rr<=12)
        if q.sum()<5:continue
        rows.append([np.mean(f[k][q]) for k in ['q_E','q_LS','R_total','R_gradient','R_envelope','R_mass_E','R_mass_LS']])
    a=np.asarray(rows); names=['q_E','q_LS','R_total','R_gradient','R_envelope','R_mass_E','R_mass_LS']; x={'n':len(rows),'delta':float(f['dt_lag']) if rows else np.nan}
    for j,k in enumerate(names):x[k+'_mean']=float(np.mean(a[:,j])) if len(a) else np.nan;x[k+'_sem']=sem(a[:,j]) if len(a) else np.nan
    summary['residual_decomposition'][str(lag)]=x

# Balanced roughness
summary['roughness']={}; rough_curves={}; redges=np.geomspace(1,8,9)
for z,rr in rough.items():
    summary['roughness'][str(z)]=aggregate_metrics(rr,2,6);rough_curves[z]=binned_curves(rr,redges)
# realization-level trend in delta_rho and delta_sigma: group means and Spearman
zs=[]; dr=[]; ds=[]
for z in sorted(rough):
    m=summary['roughness'][str(z)];zs.append(z);dr.append(m['delta_rho_mean']);ds.append(m['delta_sigma_mean'])
summary['roughness_trend']={'zeta':zs,'delta_rho_group_means':dr,'delta_sigma_group_means':ds,
                            'delta_rho_spearman_rho':float(spearmanr(zs,dr).statistic),'delta_rho_spearman_p':float(spearmanr(zs,dr).pvalue),
                            'delta_sigma_spearman_rho':float(spearmanr(zs,ds).statistic),'delta_sigma_spearman_p':float(spearmanr(zs,ds).pvalue),
                            'delta_rho_linear_slope':float(np.polyfit(zs,dr,1)[0]),'delta_sigma_linear_slope':float(np.polyfit(zs,ds,1)[0])}

# Controls
summary['controls']={}
base128=rough[1/3]
summary['controls']['cutoff']={'K64':aggregate_metrics(k64,2,6),'K128':aggregate_metrics(base128,2,6),'K192':aggregate_metrics(b1,2,6)}
# r0 compare at common scale r/ell_uv =4..8; implement using modified temporary metric
def metric_uv(runs,lo=4,hi=8):
    vals=[]
    for d in runs:
        me=derive(d,'mee');ma=derive(d,'cov');rr=me['r']/float(d['uv_scale']);q=(rr>=lo)&(rr<=hi)
        if q.sum()<5: vals.append({'beta_mee':np.nan,'beta_mass':np.nan,'mee':np.nan,'mass':np.nan});continue
        x=np.log(rr[q]);vals.append({'beta_mee':np.polyfit(x,me['sigma'][q],1)[0],'beta_mass':np.polyfit(x,ma['sigma'][q],1)[0],'mee':np.mean(me['sigma'][q]),'mass':np.mean(ma['sigma'][q])})
    out={'n_runs':len(runs),'n_covered':int(sum(np.isfinite(v['beta_mee']) for v in vals)),'range_r_over_uv':[lo,hi]}
    for k in ('beta_mee','beta_mass','mee','mass'):
        a=[v[k] for v in vals];out[k+'_mean']=float(np.nanmean(a));out[k+'_sem']=sem(a)
    return out
summary['controls']['initial_radius']={'r0_half_uv':metric_uv(r0half),'r0_one_uv':metric_uv(base128),'r0_two_uv':metric_uv(r0double)}
summary['controls']['memory']={'turnover':aggregate_metrics(base128,2,6),'constant':aggregate_metrics(memconst,2,6),'short':aggregate_metrics(memshort,2,6)}
summary['controls']['particle_number']={}
for n,idx in [(100,0),(300,1),(1000,2)]: summary['controls']['particle_number'][str(n)]=aggregate_metrics(nested,2,6,subset=idx)

# support dynamics B1
sup=[]; switch_means=[]; persistent_means=[]; corr_turn=[]; corr_nonaff=[]
for d in b1:
    ratio=np.nanmean(d['support_error_rms']/np.maximum(d['bulk_error_rms'],1e-30)); turnover=[]
    ids=d['support_ids']
    for aa,bb in zip(ids[:-1],ids[1:]):
        A=set(int(x) for x in aa if x>=0);B=set(int(x) for x in bb if x>=0);turnover.append(1-len(A&B)/max(len(A|B),1))
    turnover=np.asarray(turnover,float)
    f=finite_lag_components(d,lag=1); n=min(len(turnover),len(f['R_envelope']))
    # v2 fix: restrict to the same post-transient window as the balance analysis
    # (3<=r/r0<=12); this is the range behind the switch/persistent numbers
    # quoted in the paper (Fig. 4d).
    rr=np.asarray(f['r'][:n],float)/float(d['r0']); mask=(rr>=3)&(rr<=12)
    turn=turnover[:n][mask]; re=np.asarray(f['R_envelope'][:n],float)[mask]
    na=(.5*(d['nonaffinity'][-1,:-1]+d['nonaffinity'][-1,1:])[:n])[mask]
    hi=turn>=2/3; lo=turn<=1/3
    switch_means.append(np.mean(re[hi]) if hi.any() else np.nan)
    persistent_means.append(np.mean(re[lo]) if lo.any() else np.nan)
    def cc(x,y): return np.corrcoef(x,y)[0,1] if len(x)>3 and np.std(x)>0 and np.std(y)>0 else np.nan
    corr_turn.append(cc(re,turn)); corr_nonaff.append(cc(re,na))
    sup.append([ratio,np.mean(turnover),np.mean(d['support_boundary_count']),np.mean(d['nonaffinity'][-1])])
a=np.asarray(sup);summary['support']={}
for j,k in enumerate(['support_to_bulk_error_ratio','turnover','boundary_count','nonaffinity']):summary['support'][k+'_mean']=float(np.mean(a[:,j]));summary['support'][k+'_sem']=sem(a[:,j])
summary['support'].update({
    'Renv_switch_mean':float(np.nanmean(switch_means)),'Renv_switch_sem':sem(switch_means),
    'Renv_persistent_mean':float(np.nanmean(persistent_means)),'Renv_persistent_sem':sem(persistent_means),
    'corr_Renv_turnover_mean':float(np.nanmean(corr_turn)),'corr_Renv_turnover_sem':sem(corr_turn),
    'corr_Renv_nonaffinity_mean':float(np.nanmean(corr_nonaff)),'corr_Renv_nonaffinity_sem':sem(corr_nonaff),
})

# Save JSON before figures
with open(RES/'phase3B_analysis_summary.json','w') as f:json.dump(summary,f,indent=2)

# CSV concise table
with open(RES/'phase3B_key_metrics.csv','w',newline='') as f:
    w=csv.writer(f);w.writerow(['group','n','beta_MEE','sem_beta_MEE','beta_mass','sem_beta_mass','mean_sigma_MEE','mean_sigma_mass','delta_sigma','delta_rho'])
    for label,m in [('K192_4_12',summary['decisive_k192']['slopes_4_12'])]+[(f'zeta_{z}',summary['roughness'][str(z)]) for z in sorted(rough)]:
        w.writerow([label,m['n_covered'],m['beta_mee_mean'],m['beta_mee_sem'],m['beta_mass_mean'],m['beta_mass_sem'],m['mee_mean'],m['mass_mean'],m['delta_sigma_mean'],m['delta_rho_mean']])

# Figures
# 1 extended saturation curves
fig,ax=plt.subplots(figsize=(6.5,4.5))
ok=c1['mee_n']>=max(4,len(b1)//3);ax.errorbar(c1['centers'][ok],c1['mee_mean'][ok],yerr=c1['mee_sem'][ok],marker='o',label='outer MEE')
ok2=c1['mass_n']>=max(4,len(b1)//3);ax.errorbar(c1['centers'][ok2],c1['mass_mean'][ok2],yerr=c1['mass_sem'][ok2],marker='s',label='mass/covariance ellipse')
ax.set_xscale('log');ax.set_xlabel(r'$r_{\rm MEE}/r_0$');ax.set_ylabel(r'$\langle\sigma\mid r\rangle$');ax.legend();fig.tight_layout();fig.savefig(FIG/'B1_extended_aspect_ratios.pdf');fig.savefig(FIG/'B1_extended_aspect_ratios.png',dpi=200);plt.close(fig)
# 2 difference curves B1
fig,ax=plt.subplots(figsize=(6.5,4.5));ok=c1['delta_sigma_n']>=max(4,len(b1)//3);ax.errorbar(c1['centers'][ok],c1['delta_sigma_mean'][ok],yerr=c1['delta_sigma_sem'][ok],marker='o',label=r'$\sigma_{mass}-\sigma_{MEE}$');ax.set_xscale('log');ax.set_xlabel(r'$r_{\rm MEE}/r_0$');ax.set_ylabel('bulk–envelope anisotropy gap');fig.tight_layout();fig.savefig(FIG/'B1_delta_sigma.pdf');fig.savefig(FIG/'B1_delta_sigma.png',dpi=200);plt.close(fig)
# 3 roughness group means
fig,ax=plt.subplots(figsize=(6.5,4.5));yy=[summary['roughness'][str(z)]['delta_rho_mean'] for z in sorted(rough)];ee=[summary['roughness'][str(z)]['delta_rho_sem'] for z in sorted(rough)];ax.errorbar(sorted(rough),yy,yerr=ee,marker='o');ax.set_xlabel(r'Hölder exponent $\zeta$');ax.set_ylabel(r'$\langle\log(r_{MEE}/r_{mass})\rangle$ on $2\leq r/r_0\leq6$');fig.tight_layout();fig.savefig(FIG/'B2_roughness_delta_rho.pdf');fig.savefig(FIG/'B2_roughness_delta_rho.png',dpi=200);plt.close(fig)
# 4 residual decomposition conditional curves B1 lag1
edges2=np.geomspace(2,14,8);cent2=np.sqrt(edges2[:-1]*edges2[1:]);names=['q_E','q_LS','R_total','R_gradient','R_envelope'];per={k:np.full((len(b1),len(cent2)),np.nan) for k in names}
for i,d in enumerate(b1):
    f=finite_lag_components(d,1);rr=f['r']/float(d['r0']);ib=np.digitize(rr,edges2)-1
    for b in range(len(cent2)):
        q=ib==b
        if q.sum()>=2:
            for k in names:per[k][i,b]=np.mean(f[k][q])
fig,ax=plt.subplots(figsize=(6.5,4.5))
for k in names:
    m=np.nanmean(per[k],0);e=np.nanstd(per[k],0,ddof=1)/np.sqrt(np.sum(np.isfinite(per[k]),0));ok=np.sum(np.isfinite(per[k]),0)>=max(4,len(b1)//3);ax.errorbar(cent2[ok],m[ok],yerr=e[ok],marker='o',label=k.replace('_',' '))
ax.axhline(0,linewidth=.8);ax.set_xscale('log');ax.set_xlabel(r'$r_{MEE}/r_0$');ax.set_ylabel('conditional rate');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(FIG/'B1_residual_decomposition.pdf');fig.savefig(FIG/'B1_residual_decomposition.png',dpi=200);plt.close(fig)
# 5 robustness summary beta MEE
labels=[];means=[];errs=[]
for lab,m in [('K64',summary['controls']['cutoff']['K64']),('K128',summary['controls']['cutoff']['K128']),('K192',summary['controls']['cutoff']['K192']),('r0/uv=.5',summary['controls']['initial_radius']['r0_half_uv']),('r0/uv=1',summary['controls']['initial_radius']['r0_one_uv']),('r0/uv=2',summary['controls']['initial_radius']['r0_two_uv']),('turnover',summary['controls']['memory']['turnover']),('constant memory',summary['controls']['memory']['constant']),('short memory',summary['controls']['memory']['short'])]:
    labels.append(lab);means.append(m['beta_mee_mean']);errs.append(m['beta_mee_sem'])
fig,ax=plt.subplots(figsize=(8.4,4.6));x=np.arange(len(labels));ax.errorbar(x,means,yerr=errs,fmt='o');ax.axhline(0,linewidth=.8);ax.set_xticks(x,labels,rotation=35,ha='right');ax.set_ylabel(r'MEE slope $d\langle\sigma\rangle/d\log r$');fig.tight_layout();fig.savefig(FIG/'B3_MEE_robustness.pdf');fig.savefig(FIG/'B3_MEE_robustness.png',dpi=200);plt.close(fig)

print(json.dumps(summary,indent=2),flush=True)

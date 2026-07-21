from __future__ import annotations

import json
import os, time as _time, resource
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.ndimage import map_coordinates, spline_filter
from scipy.special import j0, j1
from scipy.spatial import ConvexHull, QhullError

TWOPI = 2.0 * np.pi


def make_wavevectors(k_max: int):
    ks=[]
    for kx in range(-k_max,k_max+1):
        for ky in range(-k_max,k_max+1):
            if kx==0 and ky==0: continue
            if kx*kx+ky*ky>k_max*k_max: continue
            if kx>0 or (kx==0 and ky>0): ks.append((kx,ky))
    k=np.asarray(ks,dtype=int)
    kmag=np.sqrt(np.sum(k.astype(float)**2,axis=1))
    perp=np.column_stack([-k[:,1],k[:,0]])/kmag[:,None]
    return k,kmag,perp


@dataclass(frozen=True)
class FlowConfig:
    k_max:int=128
    zeta:float=1/3
    dt:float=0.01
    tau0:float=1.0
    tau_model:Literal['turnover','constant','short']='turnover'
    ngrid:int=512
    interpolation_order:int=3
    # Match all roughness cases by S2 at a common reference scale.
    normalization:Literal['s2_ref','grad_rms']='s2_ref'
    s2_ref_scale:float=0.5
    s2_ref_value:float=0.17
    grad_rms:float=5.0


class SpectralOUFlowFFT:
    """Homogeneous full sine-cosine incompressible Gaussian-Hölder flow."""
    def __init__(self,cfg:FlowConfig,rng:np.random.Generator):
        if cfg.ngrid<=2*cfg.k_max:
            raise ValueError('ngrid must exceed 2*k_max')
        self.cfg=cfg
        self.k,self.kmag,self.perp=make_wavevectors(cfg.k_max)
        shape=self.kmag**(-(2*cfg.zeta+2))
        if cfg.normalization=='grad_rms':
            norm_sum=np.sum(self.kmag**(-2*cfg.zeta))
            c=cfg.grad_rms**2/(2*norm_sum)
        elif cfg.normalization=='s2_ref':
            denom=np.sum(8*shape*(1-j0(self.kmag*cfg.s2_ref_scale)))
            c=cfg.s2_ref_value/denom
        else:
            raise ValueError(cfg.normalization)
        self.E=c*shape
        if cfg.tau_model=='turnover':
            self.tau=cfg.tau0/self.kmag**(1-cfg.zeta)
        elif cfg.tau_model=='constant':
            self.tau=np.full_like(self.kmag,cfg.tau0)
        elif cfg.tau_model=='short':
            self.tau=0.1*cfg.tau0/self.kmag**(1-cfg.zeta)
        else:
            raise ValueError(cfg.tau_model)
        self.decay=np.exp(-cfg.dt/self.tau)
        self.noise=np.sqrt(self.E*(1-self.decay**2))
        self.a=np.sqrt(self.E)*rng.standard_normal(self.kmag.size)
        self.b=np.sqrt(self.E)*rng.standard_normal(self.kmag.size)
        n=cfg.ngrid
        self._pos=(self.k[:,0]%n,self.k[:,1]%n)
        self._neg=((-self.k[:,0])%n,(-self.k[:,1])%n)
        self._grid=None; self._spline=None

    def update(self,rng):
        self.a=self.decay*self.a+self.noise*rng.standard_normal(self.a.size)
        self.b=self.decay*self.b+self.noise*rng.standard_normal(self.b.size)
        self._grid=None; self._spline=None

    def velocity_grid(self):
        if self._grid is None:
            n=self.cfg.ngrid
            coeff=np.zeros((2,n,n),dtype=np.complex128)
            c=self.a+1j*self.b
            for q in range(2):
                val=c*self.perp[:,q]
                coeff[q,self._pos[0],self._pos[1]]=val
                coeff[q,self._neg[0],self._neg[1]]=np.conjugate(val)
            self._grid=np.fft.ifft2(coeff,axes=(-2,-1)).real*n*n
        return self._grid

    def evaluate_v(self,x):
        grid=self.velocity_grid(); n=self.cfg.ngrid
        uv=(np.mod(x,TWOPI)/TWOPI)*n
        ix=np.floor(uv[:,0]).astype(np.int64); iy=np.floor(uv[:,1]).astype(np.int64)
        tx=uv[:,0]-ix; ty=uv[:,1]-iy
        order=int(self.cfg.interpolation_order)
        if order==1:
            wx=np.column_stack([1-tx,tx]); wy=np.column_stack([1-ty,ty]); offs=(0,1)
        elif order==3:
            def cw(t):
                t2=t*t; t3=t2*t
                return np.column_stack([-0.5*t+t2-0.5*t3, 1-2.5*t2+1.5*t3, 0.5*t+2*t2-1.5*t3, -0.5*t2+0.5*t3])
            wx=cw(tx); wy=cw(ty); offs=(-1,0,1,2)
        else:
            raise ValueError('interpolation_order must be 1 or 3')
        out=np.zeros((len(x),2),float)
        for a,ox in enumerate(offs):
            xx=(ix+ox)%n
            for b,oy in enumerate(offs):
                yy=(iy+oy)%n; ww=wx[:,a]*wy[:,b]
                out += ww[:,None]*grid[:,xx,yy].T
        return out

    def averaged_gradient(self,center,g):
        kf=self.k.astype(float)
        rho=np.sqrt(np.maximum(np.sum((kf@g)*kf,axis=1),1e-30))
        wb=np.where(rho<1e-4,1-rho*rho/8,2*j1(rho)/rho)
        phase=kf@center
        scalar=2*(-self.a*np.sin(phase)-self.b*np.cos(phase))*wb
        # (M_L)_ij = e_perp_i k_j : the du = M y convention of the manuscript's
        # exact-averaged-gradient equation, matching best_affine_map.
        # (v2 fix: previous releases returned the transpose.)
        m=np.einsum('k,ki,kj->ij',scalar,self.perp,kf)
        return m-0.5*np.trace(m)*np.eye(2)


def safe_sym_inv(a, rel_floor=1e-12):
    a=0.5*(np.asarray(a,float)+np.asarray(a,float).T)
    w,v=np.linalg.eigh(a)
    floor=max(float(np.max(np.abs(w)))*rel_floor,1e-30)
    w=np.maximum(w,floor)
    return (v*(1.0/w))@v.T


def seed_ball(n,r0,rng):
    rr=r0*np.sqrt(rng.uniform(size=n)); th=rng.uniform(0,TWOPI,size=n)
    return np.column_stack([rr*np.cos(th),rr*np.sin(th)])


def khachiyan_mee(points,tol=1e-5,max_iter=100):
    pts=np.asarray(points,float)
    try:
        hull=ConvexHull(pts,qhull_options='QJ'); work=pts[hull.vertices]; hcount=len(work)
    except QhullError:
        work=pts; hcount=len(pts)
    n,d=work.shape; q=np.vstack([work.T,np.ones((1,n))]); u=np.full(n,1/n)
    for _ in range(max_iter):
        xdx=(q*u)@q.T; inv=safe_sym_inv(xdx,1e-14)
        lev=np.sum(q*(inv@q),axis=0); idx=int(np.argmax(lev)); mm=lev[idx]
        if mm-d-1<=tol: break
        step=(mm-d-1)/((d+1)*(mm-1)); u*=1-step; u[idx]+=step
    c=work.T@u; centered=work.T-c[:,None]
    g=d*(centered*u)@centered.T; g=0.5*(g+g.T)
    yy=pts-c; d2=np.einsum('ni,ij,nj->n',yy,safe_sym_inv(g),yy)
    g*=max(1.0,float(np.max(d2)))*(1.0+1e-10)
    return c,g,hcount


def covariance_shape(points):
    c=points.mean(axis=0); y=points-c
    C=(y.T@y)/len(points)
    return c,4*C,C


def best_affine_map(points,velocities):
    c=points.mean(axis=0); y=points-c
    vm=velocities.mean(axis=0); du=velocities-vm
    C=(y.T@y)/len(points); B=(du.T@y)/len(points)
    M=B@safe_sym_inv(C)
    pred=y@M.T; err=du-pred
    den=np.sum(du*du)
    eta=float(np.sum(err*err)/den) if den>0 else np.nan
    return M,eta,err,y


def shape_obs(g,M):
    ev,evec=np.linalg.eigh(0.5*(g+g.T)); ev=np.maximum(ev,1e-30)
    axis=evec[:,1]; theta=np.arctan2(axis[1],axis[0])
    sigma=0.5*np.log(ev[1]/ev[0]); r=(ev[0]*ev[1])**0.25; aplus=np.sqrt(ev[1])
    # v2 fix: deviatoric strain components, valid also for M with nonzero trace
    # (e.g. M_LS, whose least-squares fit does not enforce incompressibility
    # exactly; the isotropic part contributes to the area rate, not to sigma_dot).
    # omega matches the manuscript definition (M21-M12)/2.
    sp=0.5*(M[0,0]-M[1,1]); sx=0.5*(M[0,1]+M[1,0]); om=0.5*(M[1,0]-M[0,1])
    q=2*(sp*np.cos(2*theta)+sx*np.sin(2*theta))
    p=2*(-sp*np.sin(2*theta)+sx*np.cos(2*theta))
    return dict(r=r,aplus=aplus,sigma=sigma,theta=theta,q=q,p=p,omega=om,A=np.hypot(sp,sx))


@dataclass(frozen=True)
class SimConfig:
    flow:FlowConfig=FlowConfig()
    max_time:float=15.0
    n_particles:int=1000
    subsets:tuple[int,...]=(1000,)
    r0_over_uv:float=1.0
    record_stride:int=5
    mee_tol:float=1e-5
    stop_r_fraction:float=0.125
    stop_aplus_fraction:float=0.25
    support_topk:int=3
    support_boundary_tol:float=2e-3
    snapshot_scale_ratios:tuple[float,...]=(1.0,2.0,4.0,8.0,12.0)


def simulate(cfg:SimConfig,seed:int):
    rng=np.random.default_rng(seed); flow=SpectralOUFlowFFT(cfg.flow,rng)
    uv=TWOPI/cfg.flow.k_max; r0=cfg.r0_over_uv*uv
    x=seed_ball(cfg.n_particles,r0,rng)
    nsteps=int(round(cfg.max_time/cfg.flow.dt)); rec_stride=cfg.record_stride
    nmax=nsteps//rec_stride+2; nsb=len(cfg.subsets)
    result={'time':np.zeros(nmax),'subsets':np.asarray(cfg.subsets,int)}
    for kind in ('mee','cov'):
        result[f'{kind}_g']=np.zeros((nsb,nmax,2,2)); result[f'{kind}_M']=np.zeros((nsb,nmax,2,2)); result[f'{kind}_center']=np.zeros((nsb,nmax,2))
    result['M_ls']=np.zeros((nsb,nmax,2,2)); result['nonaffinity']=np.zeros((nsb,nmax))
    ksup=cfg.support_topk
    result['support_ids']=np.full((nmax,ksup),-1,int); result['support_boundary_count']=np.zeros(nmax,int); result['hull_count']=np.zeros(nmax,int)
    result['support_error_rms']=np.zeros(nmax); result['bulk_error_rms']=np.zeros(nmax); result['support_cov_radius']=np.zeros(nmax)
    ntarget=len(cfg.snapshot_scale_ratios)
    result['snapshot_positions']=np.full((ntarget,cfg.n_particles,2),np.nan); result['snapshot_time']=np.full(ntarget,np.nan); result['snapshot_ratio']=np.asarray(cfg.snapshot_scale_ratios,float)
    hit=np.zeros(ntarget,bool); jrec=0; stop_reason='max_time'; _twall=_time.time()
    stop_r=TWOPI*cfg.stop_r_fraction; stop_a=TWOPI*cfg.stop_aplus_fraction
    for step in range(nsteps+1):
        if os.environ.get('PHASE3_TRACE') and step>=910:
            print(f'trace pre step={step} wall={_time.time()-_twall:.2f} xmax={np.nanmax(np.abs(x)):.3g}', flush=True)
        isrec=(step%rec_stride==0 or step==nsteps)
        velocities=None
        if isrec:
            if os.environ.get('PHASE3_TRACE') and step>=910: print(f'rec start {step} wall={_time.time()-_twall:.2f}',flush=True)
            velocities=flow.evaluate_v(x)
            if os.environ.get('PHASE3_TRACE') and step>=910: print(f'rec vel {step} wall={_time.time()-_twall:.2f}',flush=True)
            result['time'][jrec]=step*cfg.flow.dt
            full_r=None; full_a=None
            for qs,nsub in enumerate(cfg.subsets):
                pts=x[:nsub]; vv=velocities[:nsub]
                if os.environ.get('PHASE3_TRACE') and step>=910: print(f'rec premee {step} n={nsub} wall={_time.time()-_twall:.2f}',flush=True)
                cm,gm,hc=khachiyan_mee(pts,tol=cfg.mee_tol)
                if os.environ.get('PHASE3_TRACE') and step>=910: print(f'rec postmee {step} wall={_time.time()-_twall:.2f}',flush=True)
                cc,gc,C=covariance_shape(pts)
                if os.environ.get('PHASE3_TRACE') and step>=910: print(f'rec precalc {step} wall={_time.time()-_twall:.2f}',flush=True)
                ME=flow.averaged_gradient(cm,gm); MC=flow.averaged_gradient(cc,gc)
                if os.environ.get('PHASE3_TRACE') and step>=910: print(f'rec postgrad {step} wall={_time.time()-_twall:.2f}',flush=True)
                MLS,eta,err,y=best_affine_map(pts,vv)
                result['mee_center'][qs,jrec]=cm; result['mee_g'][qs,jrec]=gm; result['mee_M'][qs,jrec]=ME
                result['cov_center'][qs,jrec]=cc; result['cov_g'][qs,jrec]=gc; result['cov_M'][qs,jrec]=MC
                result['M_ls'][qs,jrec]=MLS; result['nonaffinity'][qs,jrec]=eta
                if nsub==max(cfg.subsets):
                    o=shape_obs(gm,ME); full_r=o['r']; full_a=o['aplus']; result['hull_count'][jrec]=hc
                    yy=pts-cm; invg=safe_sym_inv(gm); d2=np.einsum('ni,ij,nj->n',yy,invg,yy)
                    ids=np.argsort(d2)[-ksup:][::-1]; result['support_ids'][jrec]=ids
                    result['support_boundary_count'][jrec]=int(np.sum(d2>=1-cfg.support_boundary_tol))
                    es=np.sqrt(np.mean(np.sum(err[ids]**2,axis=1))); mask=np.ones(nsub,bool); mask[ids]=False
                    eb=np.sqrt(np.mean(np.sum(err[mask]**2,axis=1))) if mask.any() else np.nan
                    result['support_error_rms'][jrec]=es; result['bulk_error_rms'][jrec]=eb
                    yc=pts[ids]-cc; invgc=safe_sym_inv(gc); dc=np.sqrt(np.maximum(np.einsum('ni,ij,nj->n',yc,invgc,yc),0))
                    result['support_cov_radius'][jrec]=float(np.mean(dc))
            if full_r is None: raise RuntimeError('largest subset absent')
            ratio=full_r/r0
            for it,target in enumerate(cfg.snapshot_scale_ratios):
                if (not hit[it]) and ratio>=target:
                    result['snapshot_positions'][it]=x; result['snapshot_time'][it]=step*cfg.flow.dt; hit[it]=True
            jrec+=1
            if os.environ.get('PHASE3_PROGRESS') and jrec%10==0:
                print(f'progress seed={seed} t={step*cfg.flow.dt:.2f} r={full_r:.3g} a={full_a:.3g} hull={result["hull_count"][jrec-1]} wall={_time.time()-_twall:.1f}s rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f}MB', flush=True)
            if full_r>=stop_r:
                stop_reason='r_threshold'; break
            if full_a>=stop_a:
                stop_reason='aplus_threshold'; break
        if step<nsteps:
            if velocities is None: velocities=flow.evaluate_v(x)
            if os.environ.get('PHASE3_TRACE') and step>=910:
                print(f'trace vel step={step} wall={_time.time()-_twall:.2f} vmax={np.nanmax(np.abs(velocities)):.3g}', flush=True)
            x += cfg.flow.dt*velocities
            flow.update(rng)
    # trim time-dependent arrays
    for key,val in list(result.items()):
        if key in ('time',): result[key]=val[:jrec]
        elif key in ('mee_g','mee_M','mee_center','cov_g','cov_M','cov_center','M_ls'):
            result[key]=val[:,:jrec]
        elif key=='nonaffinity': result[key]=val[:,:jrec]
        elif key in ('support_ids','support_boundary_count','hull_count','support_error_rms','bulk_error_rms','support_cov_radius'):
            result[key]=val[:jrec]
    result['stop_reason']=np.asarray(stop_reason); result['uv_scale']=np.asarray(uv); result['r0']=np.asarray(r0)
    return result


CODE_VERSION=2  # v2: averaged_gradient convention fix; deviatoric q/p; omega=(M21-M12)/2


def save_run(path:Path,cfg:SimConfig,seed:int):
    out={'config_json':np.asarray(json.dumps({'sim':asdict(cfg),'flow':asdict(cfg.flow)})),'seed':np.asarray(seed),
         'code_version':np.asarray(CODE_VERSION)}
    out.update(simulate(cfg,seed)); np.savez_compressed(path,**out)


def load_npz(path):
    with np.load(path,allow_pickle=True) as z: out={k:z[k] for k in z.files}
    if 'code_version' not in out:
        # Legacy (v1) files store the transposed spatially averaged gradients.
        # Transposing on load makes sigma, q, p, and omega from derive()/shape_obs()
        # follow the manuscript conventions for old and new data alike.
        for key in ('mee_M','cov_M'):
            if key in out: out[key]=np.swapaxes(np.asarray(out[key]),-1,-2)
    return out


def derive(data,kind='mee',subset=-1,Mkey=None):
    g=data[f'{kind}_g'][subset]
    M=data[f'{kind}_M'][subset] if Mkey is None else data[Mkey][subset]
    keys=['r','aplus','sigma','theta','q','p','omega','A']; out={k:np.zeros(len(data['time'])) for k in keys}
    for t in range(len(data['time'])):
        o=shape_obs(g[t],M[t])
        for k in keys: out[k][t]=o[k]
    out['time']=data['time']; return out


def finite_lag_components(data,lag=2,subset=-1):
    om=derive(data,'mee',subset); ols=derive(data,'mee',subset,'M_ls'); oc=derive(data,'cov',subset); ocls=derive(data,'cov',subset,'M_ls')
    dt=float(data['time'][lag]-data['time'][0]); h=float(data['time'][1]-data['time'][0])
    w=np.ones(lag+1); w[[0,-1]]=0.5
    def avg(arr):
        y=np.zeros(len(arr)-lag)
        for j,ww in enumerate(w): y+=ww*arr[j:j+len(y)]
        return y*h/dt
    dsm=(om['sigma'][lag:]-om['sigma'][:-lag])/dt
    dsc=(oc['sigma'][lag:]-oc['sigma'][:-lag])/dt
    qE=avg(om['q']); qLS=avg(ols['q']); qcE=avg(oc['q']); qcLS=avg(ocls['q'])
    return {
        'time':0.5*(data['time'][lag:]+data['time'][:-lag]),
        'r':np.sqrt(om['r'][lag:]*om['r'][:-lag]),
        'sigma':0.5*(om['sigma'][lag:]+om['sigma'][:-lag]),
        'sigma_mass':0.5*(oc['sigma'][lag:]+oc['sigma'][:-lag]),
        'sigma_dot':dsm,'sigma_mass_dot':dsc,
        'q_E':qE,'q_LS':qLS,
        'R_total':dsm-qE,
        'R_envelope':dsm-qLS,
        'R_gradient':qLS-qE,
        'R_mass_E':dsc-qcE,
        'R_mass_LS':dsc-qcLS,
        'dt_lag':np.asarray(dt)
    }

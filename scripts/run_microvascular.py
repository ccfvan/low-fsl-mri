"""Synthetic straight-cylinder diffusion--susceptibility decays (no image data).

Defaults follow the paper geometry; smoke runs are NOT converged paper results.
CPMG uses ideal instantaneous rotations, not the finite-pulse mapping kernel.
"""
import sys
from pathlib import Path
import argparse,csv,math
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lowfsl.microvascular import Params,make_fft_cylinder_field,Evolution

def fit_decay(t,y):
    t,y=np.asarray(t),np.asarray(y)
    if len(t)<4 or not np.all(np.isfinite(y)) or np.min(y)<1e-12:
        return dict(rate_s_1=float('nan'),R2=float('nan'),valid=False)
    result=least_squares(lambda p:p[0]*np.exp(-p[1]*t)-y,[y[0],12.5],bounds=([0,0],[np.inf,np.inf]))
    predicted=result.x[0]*np.exp(-result.x[1]*t)
    den=np.sum((y-y.mean())**2)
    r2=1-np.sum((predicted-y)**2)/den if den>0 else float('nan')
    return dict(rate_s_1=float(result.x[1]),R2=float(r2),valid=bool(result.success and r2>=.95))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid',type=int,default=64)
    parser.add_argument('--dt-us',type=float,default=12.5)
    parser.add_argument('--frequencies',default='20,50,100')
    parser.add_argument('--output',default='microvascular_output')
    args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if args.grid<4 or args.dt_us<=0: raise ValueError('Positive step and grid >=4 required')
    if ':' in args.frequencies:
        a,b=map(int,args.frequencies.split(':'));frequencies=list(range(a,b+1))
    else: frequencies=[float(f) for f in args.frequencies.split(',')]
    decays=[];fits=[]
    for f in frequencies:
        if not np.isfinite(f) or f<=0: raise ValueError('Frequencies must be positive')
        # Multiples of four ensure ideal echo/pulse positions fall on step boundaries.
        n=max(4,4*math.ceil(1/(4*f*args.dt_us*1e-6)))
        p=Params(nx=args.grid,ny=args.grid,nz=args.grid,dt_s=1/(f*n))
        field=make_fft_cylinder_field(p)
        evolution=Evolution(p,field,axial=True)
        for prep in ('ordinary','composite','balanced','cpmg'):
            if prep=='cpmg':
                cycles=3 if f<=11 else min(5,math.floor(.25*f))
                ts=np.arange(cycles+1)/f
            else:
                ts=np.unique(np.round(np.linspace(0,min(.2,4/f),13)/(4*p.dt_s)))*4*p.dt_s
            if prep in ('ordinary','cpmg'):
                yy=[y for _,y in evolution.samples(f,prep,ts)]
            else: yy=[evolution.independent(f,prep,float(t)) for t in ts]
            for t,y in zip(ts,yy): decays.append(dict(target_Hz=f,preparation=prep,time_s=float(t),signed_mean_Mx=y))
            row=dict(target_Hz=f,preparation=prep,grid=args.grid,dt_us=p.dt_s*1e6,**fit_decay(ts,yy))
            print(row,flush=True);fits.append(row)
    for name,rows in [('decays.csv',decays),('fits.csv',fits)]:
        with (out/name).open('w',newline='') as h:
            w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__':main()

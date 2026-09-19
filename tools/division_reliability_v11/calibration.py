"""Registered source-calibration occurrence fit and graph safety selection."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def fit_occurrence(logits,labels,distinct_positive_events):
    x=np.asarray(logits,dtype=np.float64);y=np.asarray(labels,dtype=int)
    supported=y>=0;x=x[supported];y=y[supported]
    if distinct_positive_events<2 or int((y==0).sum())<100:
        return dict(temperature=1.,intercept=0.,insufficient_support=True,margins=[6])
    def objective(v):
        scaled=x*np.exp(-v[0]);z=scaled+v[1];d=expit(z)-y
        return np.mean(np.logaddexp(0,z)-y*z)+.01*np.dot(v,v),np.array([np.mean(-d*scaled),np.mean(d)])+.02*v
    result=minimize(objective,np.zeros(2),method='L-BFGS-B',jac=True,bounds=[(-2,2),(-12,12)])
    if not result.success:raise ValueError('Occurrence calibration did not converge')
    return dict(temperature=float(np.exp(result.x[0])),intercept=float(result.x[1]),
                insufficient_support=False,margins=[2,4,6,8],objective=float(result.fun))


def select_margin(rows):
    eligible=[r for r in rows if r['combined_delta']>=0 and r['adjusted_edge_delta']>=-.001
              and r['introduced_fp']<=max(2,2*r['newly_recovered_tp'])]
    if not eligible:return dict(disabled_policy=True,margin=None)
    best=max(r['combined_delta'] for r in eligible)
    selected=max((r for r in eligible if best-r['combined_delta']<=1e-9),key=lambda r:r['margin'])
    return dict(disabled_policy=False,margin=selected['margin'])

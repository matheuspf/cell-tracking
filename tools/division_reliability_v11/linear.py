"""CPU weighted factorized C01 objectives with fixed L2 and initialization."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp


def fit(parent_features, action_features, risks, weights, complete):
    parent_features=np.asarray(parent_features,dtype=np.float64)
    weights=np.asarray(weights,dtype=np.float64)
    targets=np.asarray([1 if c and 1 in y else 0 if c and len(y) and all(z==0 for z in y) else -1
                        for y,c in zip(risks,complete)],dtype=int)
    eligible=targets>=0
    if not eligible.any() or not (targets==1).any():
        raise ValueError('C01 head training blocked: no supported positive occurrence group')
    def occurrence(theta):
        x=parent_features[eligible];y=targets[eligible];w=weights[eligible]
        z=x@theta[:-1]+theta[-1];norm=w.sum()
        loss=np.sum(w*(np.logaddexp(0,z)-y*z))/norm+.5*np.dot(theta[:-1],theta[:-1])
        dz=w*(expit(z)-y)/norm
        return loss,np.r_[x.T@dz+theta[:-1],dz.sum()]
    width=action_features[0].shape[1]
    positive=[i for i,y in enumerate(risks) if complete[i] and 1 in y]
    def action(theta):
        loss=.5*np.dot(theta[:-1],theta[:-1]);grad=np.r_[theta[:-1].copy(),0.]
        norm=weights[positive].sum()
        for i in positive:
            y=np.asarray(risks[i]);x=np.asarray(action_features[i],dtype=np.float64)
            known=y>=0;good=y==1;z=x@theta[:-1]+theta[-1]
            amount=weights[i]/norm
            loss+=amount*(logsumexp(z[known])-logsumexp(z[good]))
            dz=np.zeros(len(z));dz[known]=np.exp(z[known]-logsumexp(z[known]));dz[good]-=np.exp(z[good]-logsumexp(z[good]))
            grad[:-1]+=amount*(x.T@dz);grad[-1]+=amount*dz.sum()
        return loss,grad
    a=minimize(occurrence,np.zeros(parent_features.shape[1]+1),method='L-BFGS-B',jac=True,options=dict(maxiter=1000))
    b=minimize(action,np.zeros(width+1),method='L-BFGS-B',jac=True,options=dict(maxiter=1000))
    return dict(occurrence=a.x.tolist(),action=b.x.tolist(),l2=1.,intercept_penalized=False,
                fit_census=True,diagnostics=[dict(converged=bool(r.success),iterations=int(r.nit),
                                                objective=float(r.fun),gradient_max=float(np.abs(r.jac).max()),
                                                message=str(r.message)) for r in (a,b)])

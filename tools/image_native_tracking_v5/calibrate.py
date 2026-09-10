"""Small source-only score calibration on supported incoming choices; no target labels."""
from scipy.optimize import minimize
from scipy.special import expit
from .common import *

def logistic(X,y,residual=False,fixed=None):
    X=np.asarray(X,np.float64).reshape(len(y),-1);y=np.asarray(y,np.float64)
    prior=np.r_[np.ones(X.shape[1]),0.];bounds=[(.01,10.)]*X.shape[1]+[(-15.,15.)]
    if residual:
        # H2 uses an unchanged native logit plus a fitted HOCT residual and offset.
        prior=np.array([.25,1.,0.]);bounds=[(0.,2.),(1.,1.),(-15.,15.)]
    def objective(w):
        z=X@w[:-1]+w[-1]+(0 if fixed is None else fixed);diff=expit(z)-y
        penalty=.001*np.sum((w-prior)**2)
        value=np.mean(np.logaddexp(0,z)-y*z)+penalty
        grad=np.r_[X.T@diff/len(y),diff.mean()]+.002*(w-prior)
        return value,grad
    fit=minimize(objective,prior,method='L-BFGS-B',jac=True,bounds=bounds,options=dict(maxiter=500,ftol=1e-12))
    assert np.isfinite(fit.fun)
    return dict(coefficients=fit.x[:-1].tolist(),offset=float(fit.x[-1]),source_loss=float(fit.fun),rows=len(y),
        positives=int(y.sum()),converged=bool(fit.success),iterations=int(fit.nit),
        provenance='conditional supported incoming-parent labels; source resubstitution, not calibrated division or birth prevalence')

def apply(cal,*values):
    out=np.zeros_like(values[0],dtype=np.float32)+cal['offset']
    for coefficient,value in zip(cal['coefficients'],values):out+=coefficient*value
    return out

def hoct(source,seed):
    from .joint_calibration import fit as fit_joint
    joint=fit_joint(source)
    c=arrays(OUT/'source_calibration'/f'{source}_H_{seed}.npz');y=c['labels']
    fixed=joint['config']['incumbent_edge_bonus']*c['c0']
    result={key:logistic(c[key][:,None],y,fixed=fixed) for key in ['H0','H1','N0']}
    result['H2']=logistic(np.column_stack([c['H1'],c['N0']]),y,residual=True,fixed=fixed)
    write(OUT/'calibration'/f'{source}_H_{seed}.json',dict(source=source,seed=seed,models=result,
        input_sha256=sha(OUT/'source_calibration'/f'{source}_H_{seed}.npz'),created=now()))
    return result

if __name__=='__main__':
    import sys
    hoct(sys.argv[1],int(sys.argv[2]) if len(sys.argv)>2 else 20260910)

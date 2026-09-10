"""Source-only calibration of native evidence and the observable C0 no-op prior."""
import time
from scipy.optimize import minimize
from scipy.special import expit
from .common import *
from .supervision import supported_pairs
from .temporal_decode import DEFAULT

def fit(source):
    dest=OUT/'calibration'/f'{source}_J.json'
    if dest.exists():return read(dest)
    # Registered deterministic source-development subset; labels never cross direction.
    rows=sorted([r for r in inventory() if r['embryo']==source],key=lambda r:r['dataset'])[:10]
    X=[];Y=[];shards=[]
    for row in rows:
        name=row['dataset'];path=OUT/'banks'/source/'P0'/f'{name}.npz'
        while not path.exists():time.sleep(10)
        c=arrays(path);g=graph(name);labels,cov=supported_pairs(name,g['nodes'],c['pairs']);known=labels>=0
        old=set(map(tuple,g['edges']));x=np.column_stack([c['native_logits'],[tuple(e) in old for e in c['pairs']],np.ones(len(labels))])
        X.append(x[known]);Y.append(labels[known]);shards.append(dict(dataset=name,bank_sha256=sha(path),labels_sha256=digest(labels.tolist())))
    X=np.concatenate(X);Y=np.concatenate(Y);prior=np.array([1.,.75,0.])
    def objective(w):
        z=X@w;d=expit(z)-Y
        return float(np.mean(np.logaddexp(0,z)-Y*z)+.001*np.sum((w-prior)**2)),X.T@d/len(Y)+.002*(w-prior)
    opt=minimize(objective,prior,jac=True,method='L-BFGS-B',bounds=[(.01,10.),(0.,8.),(-15.,15.)])
    config={**DEFAULT,'incumbent_edge_bonus':float(opt.x[1])}
    result=dict(source=source,created=now(),native_scale=float(opt.x[0]),native_offset=float(opt.x[2]),config=config,
        source_loss=float(opt.fun),source_shards=shards,rows=len(Y),positives=int(Y.sum()),
        provenance='first ten lexicographic training-source clips, dependent source development; C0 membership is predicted structure',
        repair='source N0 pilot lost two TP; fit registered score scale/offset plus no-op prior before outer scoring; no target feedback')
    write(dest,result);return result

if __name__=='__main__':
    import sys
    print(fit(sys.argv[1]))

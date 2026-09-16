"""Source-only proof that frame indexing preserves exact MILP arrays and ordering."""
import time
from types import SimpleNamespace
from .common import *
from . import temporal_decode as decoder
from .event_index import WindowEvents


def fingerprint(cost,integrality,bounds,constraints,options):
    h=hashlib.sha256()
    for value in [cost,integrality,bounds.lb,bounds.ub,constraints.lb,constraints.ub,
            constraints.A.data,constraints.A.indices,constraints.A.indptr]:
        a=np.ascontiguousarray(value);h.update(str((a.shape,str(a.dtype))).encode());h.update(a.tobytes())
    h.update(str(constraints.A.shape).encode());h.update(json.dumps(options,sort_keys=True).encode())
    return h.hexdigest()


def run():
    capture=[]
    def observe(cost,*,integrality,bounds,constraints,options):
        capture.append(fingerprint(cost,integrality,bounds,constraints,options))
        return SimpleNamespace(x=None,status=99)
    original=decoder.milp;decoder.milp=observe;records=[]
    try:
        for item in read(OUT/'density_selection.json')['pilots']:
            name=item['dataset'];source=item['embryo'];base=graph(name);c=arrays(OUT/'observations'/f'{name}.npz')
            for population in ['P0','P1']:
                nodes=c['nodes'][c['oldmask']] if population=='P0' else c['nodes']
                bank=arrays(OUT/'banks'/source/population/f'{name}.npz');joint=read(OUT/'calibration'/f'{source}_J.json');cfg=joint['config']
                pairs=bank['pairs'];scores=joint['native_scale']*bank['native_logits']+joint['native_offset']
                pairs,scores=decoder.unique_edges(pairs,scores);old=set(map(tuple,base['edges']));old_ids=set(map(int,base['nodes'][:,0]))
                enabled=np.isfinite(scores)&(scores>-(cfg['birth_cost']+cfg['termination_cost']+cfg['incumbent_edge_bonus']+4))
                enabled|=np.array([tuple(p) in old for p in pairs]);pairs=pairs[enabled];scores=scores[enabled]
                times={int(n[0]):int(n[1]) for n in nodes};classes=decoder.equivalence_classes(nodes,pairs,scores,base['edges'],old_ids)
                index=WindowEvents(classes,times);pt=np.array([times[int(a)] for a,b in pairs]);protected,_=decoder.protected_context(base['edges'])
                for t in [0,30,50,97]:
                    end=min(99,t+4);nm=(nodes[:,1]>=t)&(nodes[:,1]<=end);em=(pt>=t)&(pt<end)
                    active=index.window(t,end);committed=set(list(classes)[::3])
                    timings=[];capture.clear()
                    for events in [classes,active]:
                        start=time.monotonic()
                        decoder.solve_window(nodes[nm],pairs[em],scores[em],old_ids,old,np.ones(nm.sum()),[],
                            None,{},99,protected,cfg,events,committed)
                        timings.append(time.monotonic()-start)
                    assert capture[0]==capture[1],(name,population,t)
                    records.append(dict(dataset=name,source_only=True,population=population,first_frame=t,
                        global_classes=len(classes),active_classes=len(active),all_MILP_arrays_exact=True,sha256=capture[0],
                        original_assembly_seconds=timings[0],indexed_assembly_seconds=timings[1]))
                    print('MILP arrays identical',name,population,t,[round(s,4) for s in timings],flush=True)
    finally:decoder.milp=original
    write(OUT/'event_index_equivalence.json',dict(at=now(),passed=True,source_only=True,fixtures=records,
        compared='cost, integrality, lower/upper bounds, CSC matrix data/indices/indptr/shape, all row bounds, solver options',
        numerical_decoder_changed=False,optimization='avoid scanning alternatives at unobserved frames; preserve variable/row ordering',
        original_decoder_sha256=sha(Path(decoder.__file__))))


if __name__=='__main__':run()

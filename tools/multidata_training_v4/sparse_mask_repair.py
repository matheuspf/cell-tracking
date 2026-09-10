"""Preserve pre-outcome failed fits; repair sparse labels and repeat full budgets."""
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from .common import *

FAILED=OUT/'failed_fits/sparse_edge_mask'

def prepare():
    proof=OUT/'sparse_edge_mask_correction.json'
    if proof.exists():return
    assert not (OUT/'inference_config.json').exists()
    assert not (OUT/'calibration.json').exists()
    FAILED.mkdir(parents=True,exist_ok=True)
    unaffected={'G_synthetic','I_synthetic','I_randomized_44b6','I_randomized_6bba'}
    moved=[]
    for p in sorted((OUT/'models').glob('*')):
        if p.suffix not in ['.pt','.json'] or p.stem in unaffected or not p.stem.startswith(('G_','I_')):continue
        dest=FAILED/'models'/p.name;dest.parent.mkdir(exist_ok=True)
        assert not dest.exists(),dest
        p.rename(dest)
    moved=[dict(path=str(p.relative_to(FAILED)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted((FAILED/'models').glob('*'))]
    for p in (OUT/'logs').glob('*.log'):
        dest=FAILED/'logs'/p.name;dest.parent.mkdir(exist_ok=True)
        if not dest.exists():shutil.copyfile(p,dest)
    changes=[]
    from .index import supervision,zoo
    from .proposals import build
    for row in inputs():
        name=row['dataset'];path=OUT/'cache/real'/f'{name}.npz'
        dest=FAILED/'cache/real'/path.name;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copyfile(path,dest)
        old=arrays(dest)
        graph=arrays(V3/'selected_predictions'/f'{name}.npz');truth=arrays(V1/'evaluation/gt'/f'{name}.npz')
        matches=dict(arrays(OUT/'evaluation_matches/C0'/f'{name}.npz')['matches'])
        gtindex={int(n):i for i,n in enumerate(truth['nodes'][:,0])}
        mapping=np.array([gtindex.get(int(matches.get(int(i),-1)),-1) for i in graph['nodes'][:,0]])
        reverse={g:i for i,g in enumerate(mapping) if g>=0};anchors=np.array([reverse[int(g)] for g in old['group']])
        a,c,x=build(graph['nodes'][:,1],graph['nodes'][:,2:].astype(float),anchors)
        edges=np.array([[gtindex[int(a)],gtindex[int(b)]] for a,b in truth['edges']],np.int64).reshape(-1,2)
        y,ly,w=supervision(a,c,mapping,edges,truth['nodes'][:,1])
        np.testing.assert_array_equal(old['x'],x);np.testing.assert_array_equal(old['target'],y)
        changed=ly!=old['link_y'];assert ((old['link_y'][changed]==0)&(ly[changed]==-1)).all()
        changes.append(dict(source=row['embryo'],sample=name,negatives_changed_to_unknown=int(changed.sum()),cache_before_sha256=sha(dest)))
        old['link_y']=ly;save(path,**old);changes[-1]['cache_after_sha256']=sha(path)
    for species in ['zebrafish','ascidian']:
        old={}
        for part in ['train','validation','test']:
            path=OUT/'cache/zoo'/f'{species}_{part}.npz'
            dest=FAILED/'cache/zoo'/path.name;dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():shutil.copyfile(path,dest)
            old[part]=arrays(dest)
        zoo(species)
        for part,before in old.items():
            path=OUT/'cache/zoo'/f'{species}_{part}.npz';after=arrays(path)
            # No-candidate censored rows have no supervised loss. Retain their
            # original weak weights so this repair changes only edge masks.
            different_weight=before['weight']!=after['weight']
            assert (after['target'][different_weight]<0).all()
            assert (after['link_y'][different_weight]<0).all()
            after['weight']=before['weight'];save(path,**after)
            for k in before:
                if k!='link_y':np.testing.assert_array_equal(before[k],after[k])
            changed=before['link_y']!=after['link_y'];assert ((before['link_y'][changed]==0)&(after['link_y'][changed]==-1)).all()
            changes.append(dict(source=species,sample=part,negatives_changed_to_unknown=int(changed.sum()),
                cache_before_sha256=sha(FAILED/'cache/zoo'/path.name),cache_after_sha256=sha(path)))
    index=OUT/'runtime_dataset_index.jsonl';shutil.copyfile(index,FAILED/index.name)
    records=[json.loads(s) for s in index.read_text().splitlines()]
    for r in records:
        if r['source'].startswith(('biohub_','zoo_')):r['sparse_edge_mask_version']='unknown_second_child_masked_v2'
    index.write_text(''.join(json.dumps(r)+'\n' for r in records))
    write(proof,dict(created=now(),comparative_outcomes_opened=False,reason='A single recorded child cannot contradict an unannotated second daughter',
        repair='Mask unsupported candidate edge negatives; retain positives, complete binary-fork contradictions and known different predecessors',
        unchanged=['event bag targets','point/image inputs','partitions','sampling cycles','budgets','D fits','synthetic-only pretraining'],
        affected_fits_restarted_from_original_initializations=True,old_artifacts=str(FAILED),archived_model_files=moved,cache_changes=changes,
        total_negatives_changed_to_unknown=sum(r['negatives_changed_to_unknown'] for r in changes),index_sha256=sha(index)))

def specs():
    pre=[];adapt=[]
    def add(target,name,component,domains,steps,**kw):target.append(dict(name=name,component=component,domains=domains,steps=steps,**kw))
    mix=['synthetic','zebrafish','synthetic','zebrafish','synthetic','zebrafish','ascidian','ascidian']
    for tag,domains in [('fish',['zebrafish']),('combined',['synthetic','zebrafish']),('multispecies',mix)]:
        add(pre,'G_'+tag,'G',domains,20000)
    for source in ['44b6','6bba']:
        for suffix,seed in [('',SEED),('_seed2',314159)]:
            add(pre,f'G_real_{source}{suffix}','G',[source],20000,seed=seed)
            add(pre,f'I_real_{source}{suffix}','I',[source],20000,seed=seed,init=f'D_real_{source}{suffix}')
        for arm in ['C1','C2','C3','C4','C5','C6','C1short']:
            gi={'C1':f'G_real_{source}','C2':'G_synthetic','C3':'G_fish','C4':'G_combined','C5':'G_multispecies','C6':'G_combined','C1short':None}[arm]
            ii=(f'I_real_{source}' if arm in ['C1','C3'] else f'I_randomized_{source}' if arm=='C6' else 'I_synthetic') if arm!='C1short' else None
            replay={'C1':None,'C2':['synthetic'],'C3':None,'C4':['synthetic','zebrafish'],'C5':mix,'C6':['synthetic','zebrafish'],'C1short':None}[arm]
            add(adapt,f'G_{arm}_{source}','G',[source],8000,init=gi,source=source,replay=replay)
            add(adapt,f'I_{arm}_{source}','I',[source],8000,init=ii,source=source,replay=['synthetic'] if arm in ['C2','C4','C5','C6'] else None,randomized=arm=='C6')
    return pre,adapt

def worker(name):
    from .train import fit
    jobs=sum(specs(),[]);job=next(j for j in jobs if j['name']==name);fit(**job)

def run():
    start=time.monotonic();prepare()
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_multidata_v4.py'],cwd=REPO,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    write(OUT/'sparse_edge_mask_tests.json',dict(passed=True,output=result.stdout+result.stderr))
    pre,adapt=specs();jobs=[]
    def launch(job):
        log=OUT/'logs'/f'repaired_{job["name"]}.log';started=time.monotonic()
        with log.open('a') as f:r=subprocess.run([sys.executable,'-m','multidata_training_v4.sparse_mask_repair',job['name']],stdout=f,stderr=subprocess.STDOUT)
        record=dict(model=job['name'],returncode=r.returncode,seconds=time.monotonic()-started)
        print(record,flush=True);return record
    for phase,items in [('pretraining',pre),('adaptation',adapt)]:
        status('W420-W440',state='full retraining after sparse edge mask correction',phase=phase,concurrent_gpu_processes=4)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for r in pool.map(launch,items):jobs.append(r);write(OUT/'sparse_edge_mask_retraining_progress.json',jobs)
        assert all(r['returncode']==0 for r in jobs),jobs
    # The standard driver verifies every config/hash and generates the primary lock.
    r=subprocess.run([sys.executable,'-m','multidata_training_v4','train']);assert r.returncode==0
    write(OUT/'sparse_edge_mask_retraining_receipt.json',dict(completed=True,seconds=time.monotonic()-start,jobs=jobs,primary_models_and_seed2_controls_retrained=True))

if __name__=='__main__':worker(sys.argv[1])

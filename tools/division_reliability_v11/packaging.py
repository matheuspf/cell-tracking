"""Copy only verified v11-owned artifacts into explicit immutable packages."""
from pathlib import Path
import shutil
from .common import ARCH,REPO,WORK,RESULTS,Blocked,read,sha
from .provenance import artifact,validate,seal


def run(source,seed,arm='C00'):
    if arm!='C00':return _build(source,seed,arm)
    from .source_prefetch import ownership,await_workers
    with ownership(source,seed):
        await_workers(source,seed)
        return _build(source,seed,arm)


def _build(source,seed,arm='C00'):
    from .readiness import require_production
    require_production('package');lock=read(RESULTS/'execution_lock.json')
    fit=WORK/'fits'/source/str(seed);up=read(fit/'upstream/final.json')
    if up['status']!='trained' or up['lock_identity']!=lock['identity']:raise Blocked('No retained upstream fit for this lock')
    if sha(fit/'upstream/final.pt')!=up['weights_sha256']:raise Blocked('Upstream weights changed')
    folder=WORK/'packages'/source/str(seed)/arm;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'manifest.json').exists():
        from .provenance import unseal
        return unseal(folder)
    ancestry={};parents=[]
    source_split=read(WORK/'source_partitions.json')[source]
    for clip in source_split['fit']:
        receipt=read(WORK/'preprocessed'/source/clip/'receipt.json')
        for kind,spec in [('raw_images',dict(decoded_manifest_sha256=sha(WORK/'decoded_inventory'/f'{clip}.json'))),
                          ('raw_labels',dict(files=receipt['source_annotation_files']))]:
            key=kind+':'+clip;ancestry[key]=artifact(kind,source=source,partition='fit',**spec);parents.append(key)
    cache_manifest=read(WORK/'preprocessed'/source/'manifest.json')
    ancestry['cache_constructor']=artifact('code',files={
        'preprocess.py':cache_manifest['specification']['construction_code'],
        'data.py':cache_manifest['specification']['data_code']})
    ancestry['preprocessed_fit_cache']=artifact('preprocessing',[*parents,'cache_constructor'],
        source=source,partition='fit',learned_statistics=False,
        manifest_sha256=sha(WORK/'preprocessed'/source/'manifest.json'),
        clip_receipt_hashes=cache_manifest['clip_receipt_hashes'])
    parents=['preprocessed_fit_cache',*parents]
    runtime_names=('inference.py','upstream.py','deterministic.py','data.py','graphs.py','models.py','features.py','policy.py',
                   'actions.py','guard.py','provenance.py','stage_provenance.py','common.py','resources.py')
    runtime_code={n:sha(Path(__file__).with_name(n)) for n in runtime_names}
    if arm=='C11':
        runtime_code.update({n:sha(Path(__file__).with_name(n)) for n in ('execution_precision.py','__main__.py')})
    ancestry['code']=artifact('code',files=runtime_code,fitting_core_files=lock['implementation_sha256']);parents.append('code')
    architecture={}
    for name in ('temporal_unet.py','simple_node_transformer.py'):
        src=ARCH/'src/biohub_tracking/models'/name;dest=folder/'architecture/src/biohub_tracking/models'/name
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest);architecture[name]=sha(src)
    ancestry['architecture']=artifact('architecture',files=architecture,weights_loaded=False);parents.append('architecture')
    ancestry['upstream']=artifact('model',parents,source=source,initialization='random',seed=seed,
                                  sha256=up['weights_sha256'],updates=up['updates'],lock_identity=lock['identity'])
    shutil.copy2(fit/'upstream/final.pt',folder/'upstream.pt');root='upstream'
    if arm!='C00':
        linear=read(fit/'linear/model.json');cal=read(fit/'calibration'/arm/'calibration.json')
        if linear['status']!='fitted' or cal['status']!='calibrated':raise Blocked('Head or safety selection incomplete')
        bank=WORK/'banks'/source/str(seed)/'fit'
        bank_files={}
        for clip in source_split['fit']:
            receipt=read(bank/clip/'receipt.json')
            if sha(bank/clip/'receipt.json')!=linear['source_bank_receipts'][clip] or sha(bank/clip/'training.npz')!=receipt['training_sha256']:
                raise Blocked('Fitted head bank ancestry changed')
            bank_files[clip]=dict(receipt=sha(bank/clip/'receipt.json'),arrays=receipt['training_sha256'],
                C00_graph=receipt['graph_sha256'],guard_receipt=sha(bank/clip/'worker.json'))
        ancestry['bank_code']=artifact('code',files={n:sha(Path(__file__).with_name(n)) for n in
            ('dataset.py','risk_labels.py','actions.py','features.py')},external_files=lock['external_code_sha256'],scorer_revision=lock['metric_revision'])
        ancestry['fit_bank']=artifact('preprocessing',['bank_code','upstream',*parents],source=source,partition='fit',
            learned_statistics=False,complete_source_fit_census=True,files=bank_files)
        ancestry['normalizer_code']=artifact('code',files=linear['implementation_sha256'])
        ancestry['normalizer']=artifact('normalizer',['normalizer_code','fit_bank'],source=source,partition='fit',sha256=sha(fit/'linear/normalizer.json'))
        shutil.copy2(fit/'linear/normalizer.json',folder/'normalizer.json')
        if arm=='C01':
            ancestry['head']=artifact('linear',['fit_bank','normalizer'],source=source,sha256=sha(fit/'linear/model.json'))
            shutil.copy2(fit/'linear/model.json',folder/'linear.json')
        elif arm=='C11':
            final=read(fit/'compact/final.json')
            if final['status']!='trained' or sha(fit/'compact/final.pt')!=final['weights_sha256']:raise Blocked('Compact final weights unavailable')
            ancestry['compact_code']=artifact('code',files=final['implementation_sha256'])
            mining=fit/'compact/mining';mined=read(mining/'receipt.json')
            if sha(mining/'frozen_checkpoint.pt')!=mined['signature']['checkpoint_sha256']:
                raise Blocked('Final compact mining ancestry changed')
            ancestry['midpoint']=artifact('model',['compact_code','fit_bank','normalizer'],source=source,initialization='random',
                seed=seed,updates=lock['event_updates']//2,sha256=sha(mining/'frozen_checkpoint.pt'))
            repair=read(RESULTS/'compact_precision_repair.json')
            if repair['status']!='passed':raise Blocked('Compact evaluation precision proof is missing')
            ancestry['compact_evaluation_runtime']=artifact('code',files={n:runtime_code[n] for n in ('execution_precision.py','__main__.py')},
                environment={'NVIDIA_TF32_OVERRIDE':'0'},repair_sha256=sha(RESULTS/'compact_precision_repair.json'))
            ancestry['mining_pool']=artifact('preprocessing',['midpoint','fit_bank','normalizer','compact_evaluation_runtime'],source=source,partition='fit',
                sha256=sha(mining/'receipt.json'),passes=1,source_only=True)
            ancestry['head']=artifact('model',['compact_code','fit_bank','normalizer','mining_pool'],source=source,initialization='random',
                                      seed=seed,sha256=final['weights_sha256'],updates=final['updates'])
            shutil.copy2(fit/'compact/final.pt',folder/'compact.pt')
        else:raise Blocked('Unregistered arm')
        ancestry['calibrator_code']=artifact('code',['compact_evaluation_runtime'] if arm=='C11' else [],
            files=cal['implementation_sha256'],scorer_revision=lock['metric_revision'])
        cp=['head','normalizer','calibrator_code']
        for clip in source_split['calibration']:
            for kind in ('raw_images','raw_labels'):
                key='calibration:'+kind+':'+clip
                # Calibration receipts identify the actual prediction and GEFF parents.
                ancestry[key]=artifact(kind,source=source,partition='calibration',
                                       sha256=cal['parent_manifests'][clip][kind]);cp.append(key)
        cache_files={n:dict(raw_scores=sha(fit/'calibration'/arm/n/'logits.npz'),
            score_receipt=sha(fit/'calibration'/arm/n/'logits.json'),
            label_bank_receipt=sha(WORK/'banks'/source/str(seed)/'calibration'/n/'receipt.json')) for n in source_split['calibration']}
        ancestry['calibration_observations']=artifact('calibration',cp,source=source,files=cache_files,
            learned_statistics=False,raw_scores_label_free=True)
        ancestry['calibration']=artifact('calibration',['calibration_observations',*cp],source=source,sha256=sha(fit/'calibration'/arm/'calibration.json'))
        shutil.copy2(fit/'calibration'/arm/'calibration.json',folder/'calibration.json');root='calibration'
    from .stage_provenance import validate_stages
    validate_stages(ancestry,root,source)
    files={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file()}
    value=dict(arm=arm,source=source,seed=seed,files=files,ancestry=ancestry,root_artifact=root,
               runtime_code_sha256=runtime_code,
               lock_identity=lock['identity'],label_exposure_class='source_isolated_reused_embryos',
               inference_requires_explicit_package=True,recipe_sha256=sha(REPO/'handover/division-reliability-v11/study.json'))
    if arm=='C11':value['inference_environment']={'NVIDIA_TF32_OVERRIDE':'0'}
    seal(folder,value);return {**value,'path':str(folder.relative_to(REPO))}

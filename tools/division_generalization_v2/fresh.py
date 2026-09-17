"""Cold renamed image -> complete original P0 -> frozen event module proof."""
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

from .common import (WORK,RESULTS,ROOT,PRIOR_RESULTS,inputs,read_json,write_json,sha,
                     verified_graph,load_graph,graph_hash)
from .resources import Lease,Monitor
from .screen import launch

BASE=ROOT/'image-native-tracking-v5/inference_package_validation/base'


def run():
    manifest=read_json(BASE/'manifest.json')
    for collection in ('code','package_files'):
        for relative,expected in manifest[collection].items():
            if sha(BASE/relative)!=expected:raise ValueError('Pinned full P0 package drift')
    for name,path in manifest['external_checkpoint_paths'].items():
        if sha(path)!=manifest[name+'_weights_sha256']:raise ValueError('Pinned detector checkpoint drift')
    for source,path in manifest['external_teacher_paths'].items():
        if sha(path)!=manifest['legacy_teacher_model_hashes'][source]:raise ValueError('Pinned teacher drift')
    freeze=read_json(RESULTS/'target_freeze.json')
    arm=freeze['nominee'] if freeze['nominee'] in freeze['qualified_exports'] else None
    proof=[]
    for dataset,renamed in [('44b6_d754aa59','specimen_alder'),('6bba_bb9f20c3','specimen_birch')]:
        row=next(r for r in inputs() if r['dataset']==dataset)
        source='44b6' if row['embryo']=='6bba' else '6bba'
        root=WORK/'fresh'/renamed;images=root/'images';output=root/'output'
        images.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True,exist_ok=True)
        link=images/(renamed+'.zarr')
        if not link.exists():link.symlink_to(Path(row['image_path']).resolve(),target_is_directory=True)
        p0=read_json(PRIOR_RESULTS/'model_input_manifest.json')[f'P0_{source}']
        if sha(p0['path'])!=p0['sha256']:raise ValueError('Pinned P0 residual drift')
        command=[sys.executable,'-m','pipeline_error_training.fresh_entry','--images',str(images),
            '--output',str(output),'--package',str(BASE),'--source',source,'--p0-model',p0['path'],
            '--v1',str(ROOT/'annotation-selection-v1'),'--v2',str(ROOT/'strong-tracker-v2')]
        started=time.monotonic();stages=[]
        for stage,receipt in [('baseline','inference_receipt.json'),('point-head','P0_receipt.json')]:
            timing_path=output/(stage+'.timing.json')
            if not (output/receipt).exists():
                stage_started=time.monotonic()
                empty_pipeline_cache=not (output/'fresh').exists()
                with Monitor(output/(stage+'.resources.json')),Lease('fresh/'+stage,required_gib=12.) as lease:
                    with (output/(stage+'.log')).open('a') as f:
                        result=subprocess.run([*command,'--stage',stage],stdout=f,stderr=subprocess.STDOUT)
                write_json(timing_path,dict(wall_seconds=time.monotonic()-stage_started,
                    lease_seconds=lease.seconds,wait_seconds=lease.wait_seconds,
                    returncode=result.returncode,empty_pipeline_cache_at_start=empty_pipeline_cache,
                    OS_page_cache_flushed=False,command=[*command,'--stage',stage]))
                if result.returncode:raise RuntimeError(f'Fresh {stage} failed; preserved log {output}')
            stages.append(dict(stage=stage,receipt_sha256=sha(output/receipt),
                timing=read_json(timing_path),timing_sha256=sha(timing_path)))
        baseline=load_graph(output/'P0.npz');expected=verified_graph(row)
        for k in ('nodes','edges'):np.testing.assert_array_equal(baseline[k],expected[k])
        module=baseline;module_trace=None
        if arm:
            selected=freeze['selected'][f'{arm}/{source}/20260916']['selected']
            checkpoint=WORK/'training'/arm/source/'20260916'/f'checkpoint-{selected["step"]}.pt'
            freshrow=read_json(output/'inputs.json')[0]
            safe=dict(dataset=renamed,image_path=str(link),image_shape=row['image_shape'],
                      physical_scale=row['physical_scale'],metadata_sha256=sha(link/'zarr.json'))
            job=dict(row=safe,source=source,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                output=str(output/'event'),graph_path=str(output/'P0.npz'),graph_sha256=sha(output/'P0.npz'),
                native_path=str(output/'current_evidence.npz'),native_sha256=sha(output/'current_evidence.npz'),
                calibration=selected['calibration'])
            launch(job,root/'module_job.json')
            module_trace=read_json(output/'event/trace.json')
            module=load_graph(output/'event'/selected['application']/(renamed+'.npz'))
            scored=load_graph(WORK/'target'/arm/'20260916'/'predictions'/dataset/selected['application']/(dataset+'.npz'))
            for k in ('nodes','edges'):np.testing.assert_array_equal(module[k],scored[k])
        from pipeline_error_training.serialization import export_csv,export_geff
        csv=export_csv(output/'candidate.csv',renamed,module['nodes'],module['edges'])
        geffpath=output/'fresh/candidate.geff'
        geff=export_geff(geffpath,module['nodes'],module['edges']) if not geffpath.exists() else dict(previously_exported=True)
        guards=[read_json(p) for p in (output/'startup_audits').glob('*.json')]
        if arm:guards.append(read_json(output/'event/guard.json'))
        if not guards or any(g['blocked_reads'] or g['blocked_network'] or not g['installed_before_numerical'] for g in guards):
            raise ValueError('Fresh startup denial proof failed')
        proof.append(dict(dataset=dataset,renamed=renamed,source=source,arm=arm or 'P0',
            seconds=time.monotonic()-started,exact_P0=True,exact_scored_candidate=True,
            graph_hash=graph_hash(module['nodes'],module['edges']),csv=csv,geff=geff,guards=guards,stages=stages,
            module_timing=None if module_trace is None else dict(seconds=module_trace['seconds'],**module_trace['timings']),
            pipeline_stage_service_seconds=sum(s['timing']['lease_seconds'] for s in stages)+(
                module_trace['seconds']-module_trace['timings'].get('lease_wait_seconds',0.) if module_trace else 0.),
            cold_scope='Empty pipeline artifact caches before baseline; operating-system page cache is uncontrolled',
            all_stage_timings_persisted_across_resume=True))
    result=dict(status='measured',clips=proof,full_image_reconstruction=True,
        cold_pipeline_artifacts=all(p['stages'][0]['timing']['empty_pipeline_cache_at_start'] for p in proof),
        deny_annotation_old_cache_network_from_startup=True,Kaggle_runtime_guarantee=False)
    write_json(RESULTS/'fresh_image_validation.json',result)
    return result

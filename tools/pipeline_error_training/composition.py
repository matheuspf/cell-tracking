"""Execute the single source-nominated composition for both fixed seeds."""
from pathlib import Path
import shutil
import subprocess
import sys

from .common import RESULTS,ROOT,WORK,inputs,read_json,sha,write_json


def run():
    freeze=read_json(RESULTS/'target_freeze.json')
    recipe=freeze['composition']
    if not recipe['enabled']:
        result = dict(status='not run',reason='Both independent component families did not source-qualify')
        write_json(RESULTS/'composition.json',result)
        return result
    records=[]
    for seed,suffix in [(20260915,''),(314159,'_replication')]:
        experiment='C10'+suffix
        for row in inputs():
            source='6bba' if row['embryo']=='44b6' else '44b6'
            models={kind:next((p for p in freeze['packages'] if p['arm']==recipe[kind] and p['seed']==seed and p['source']==source),None)
                    for kind in ['identity','division']}
            identity_graph=WORK/'predictions'/(recipe['identity']+suffix)/f'{row["dataset"]}.npz'
            division_graph=WORK/'predictions'/(recipe['division']+suffix)/f'{row["dataset"]}.npz'
            if any(m is None for m in models.values()) or not identity_graph.exists() or not division_graph.exists():
                records.append(dict(status='blocked',experiment=experiment,dataset=row['dataset'],reason='Frozen component prediction is unavailable'))
                continue
            root=WORK/'composition'/experiment/row['dataset']
            original=ROOT/'strong-tracker-v2/raw'/f'{row["dataset"]}.npz'
            evidence_receipt=Path(row['evidence']['path']).with_suffix('.json')
            dependencies=[dict(path=row['baselines']['P0']['path'],sha256=row['baselines']['P0']['sha256']),
                dict(path=row['evidence']['path'],sha256=row['evidence']['sha256']),dict(path=row['raw']['path'],sha256=row['raw']['sha256']),
                dict(path=str(original),sha256=read_json(evidence_receipt)['inputs']['raw_sha256']),
                dict(path=str(identity_graph),sha256=sha(identity_graph)),dict(path=str(division_graph),sha256=sha(division_graph))]
            spec=read_json(models['division']['manifest_path'])
            if 'architecture_dependency' in spec:dependencies.append(spec['architecture_dependency'])
            fresh_query=[]
            query_root=WORK/'final_point_inference'/(recipe['identity']+suffix)/row['dataset']/'fresh_native'
            for name in ['query.npz','query.json']:
                path=query_root/name
                if path.exists():
                    fresh_query.append(str(path));dependencies.append(dict(path=str(path),sha256=sha(path)))
            manifest=read_json(ROOT/'image-native-tracking-v5/inference_package_validation/base/manifest.json')
            dependencies.extend(dict(path=manifest['external_checkpoint_paths'][k],sha256=manifest[k+'_weights_sha256']) for k in ['primary','secondary'])
            safe={k:row[k] for k in ['dataset','image_path','image_shape','physical_scale','metadata_sha256']}
            job=dict(row=safe,source=source,root=str(root),experiment=experiment,dependencies=dependencies,
                package=str(Path(models['division']['manifest_path']).parent),package_sha256=models['division']['manifest_sha256'],
                identity_model_sha256=models['identity']['weights_sha256'],P0=row['baselines']['P0']['path'],
                identity_graph=str(identity_graph),division_graph=str(division_graph),evidence=row['evidence']['path'],
                original_raw=str(original),raw_pre=row['raw']['path'],fresh_query=fresh_query)
            path=root/'job.json';write_json(path,job,immutable=True)
            if not (root/'complete.json').exists():
                with (root/'predict.log').open('a') as log:
                    child=subprocess.run([sys.executable,'-m','pipeline_error_training.composition_entry',str(path)],stdout=log,stderr=subprocess.STDOUT)
                if child.returncode:
                    records.append(dict(status='failed',experiment=experiment,dataset=row['dataset'],log_sha256=sha(root/'predict.log')))
                    continue
            output=root/f'{row["dataset"]}.npz';guard=read_json(root/'guard.json')
            if guard['blocked_reads'] or guard['blocked_network'] or not guard['installed_before_numerical']:
                raise RuntimeError('Composition guard failed')
            destination=WORK/'predictions'/experiment/output.name;destination.parent.mkdir(parents=True,exist_ok=True)
            for suffix2 in ['.npz','.json','.csv']:
                if destination.with_suffix(suffix2).exists():
                    if sha(destination.with_suffix(suffix2))!=sha(output.with_suffix(suffix2)):raise ValueError('Completed composition changed')
                else:shutil.copyfile(output.with_suffix(suffix2),destination.with_suffix(suffix2))
            write_json(destination.with_suffix('.guard.json'),guard,immutable=True)
            result=dict(status='measured',experiment=experiment,dataset=row['dataset'],prediction_sha256=sha(destination))
            write_json(root/'complete.json',result,immutable=True);records.append(result)
    result = dict(status='measured' if all(r['status']=='measured' for r in records) else 'failed',
        order=recipe['order'],nominees=recipe,jobs=records,target_driven_combination_search=False)
    write_json(RESULTS/'composition.json',result)
    return result


if __name__=='__main__':
    from .resources import cpu_budget
    cpu_budget()
    raise SystemExit(1 if run()['status']=='failed' else 0)

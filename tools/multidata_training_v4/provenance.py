"""Explicit model ancestry and actual alias/partition audit."""
from collections import defaultdict
from .common import *

def model_manifest(secondary=False,subset=None,destination=None):
    receipts={p.stem:read(p) for p in (OUT/'models').glob('*.json')}
    def ancestry(name):
        r=receipts[name];parent=r['config']['init']
        return (ancestry(parent) if parent else [])+[name]
    result={}
    for name,r in receipts.items():
        if subset is not None:
            if name not in subset:continue
        elif ('seed2' in name)!=secondary:continue
        lineage=ancestry(name);domains=sorted({d for a in lineage for d in receipts[a]['direct_sources']})
        exposure=[]
        if 'synthetic' in domains:exposure.append('released simulator calibrated on Biohub 44b6')
        exposure.extend('eligible Zoo-fish trajectories rendered with '+d.removeprefix('rendered_zoo_')+' source-only appearance statistics' for d in domains if d.startswith('rendered_zoo_'))
        if any(d in domains for d in ['44b6','6bba']) or any(d.startswith('rendered_zoo_') for d in domains):
            exposure += ['incumbent detections from public neural checkpoints','historical v2 E teachers and v3 graph processing retain inherited target exposure']
        result[name]=dict(sha256=sha(OUT/'models'/f'{name}.pt'),bytes=(OUT/'models'/f'{name}.pt').stat().st_size,
            component=r['config']['component'],seed=r['config']['seed'],actual_optimizer_updates=r['actual_updates'],
            direct_sources=r['direct_sources'],initialization=r['config']['init'],ancestry=lineage,all_ancestral_sources=domains,
            inherited_exposure=exposure,appearance_calibration_source=r['config']['appearance_calibration_source'],
            calibration='final G/I weights share generator-validation temperature scaling across controls',
            operational_exploratory=True)
    dest=destination or OUT/('checkpoint_manifest_secondary.json' if secondary else 'checkpoint_manifest.json');write(dest,result)
    return result

def equivalent_histories():
    import torch
    rows=[]
    for source in ['44b6','6bba']:
        for a,b in [('I_C1','I_C3'),('I_C2','I_C4'),('I_C4','I_C5'),('G_C4','G_C6')]:
            first,second=f'{a}_{source}',f'{b}_{source}'
            ca=read(OUT/'models'/f'{first}.json')['config'];cb=read(OUT/'models'/f'{second}.json')['config']
            assert {k:v for k,v in ca.items() if k!='name'}=={k:v for k,v in cb.items() if k!='name'}
            x=torch.load(OUT/'models'/f'{first}.pt',map_location='cpu',weights_only=False)['model']
            y=torch.load(OUT/'models'/f'{second}.pt',map_location='cpu',weights_only=False)['model']
            floating=[k for k in x if x[k].is_floating_point()]
            diff=sum(float((x[k].double()-y[k].double()).square().sum()) for k in floating)
            denom=sum(float(x[k].double().square().sum()) for k in floating)
            rows.append(dict(first=first,second=second,config_equal_except_name=True,
                bitwise_equal=all(torch.equal(x[k],y[k]) for k in x),relative_parameter_l2=float(np.sqrt(diff/max(denom,1e-30))),
                maximum_parameter_difference=max(float((x[k].double()-y[k].double()).abs().max()) for k in x)))
    write(OUT/'equivalent_history_drift.json',dict(created=now(),comparisons=rows,
        purpose='Quantify repeated same-history numerical training variation, without target labels',
        bitwise_training_reproducibility_claimed=False))

def run():
    import pandas as pd
    from strong_tracker_v3.common import digest
    image_checks=[]
    for row in inputs():
        root=Path(row['image_path']);metadata=read(root/'zarr.json');array=read(root/'0/zarr.json')
        scale=metadata['attributes']['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale'][-3:]
        assert digest([metadata,array])==row['metadata_hash']
        assert array['shape']==row['image_shape'] and array['data_type']=='uint16'
        assert np.array_equal(scale,row['physical_scale'])
        image_checks.append(dict(dataset=row['dataset'],metadata_sha256=sha(root/'zarr.json'),array_metadata_sha256=sha(root/'0/zarr.json')))
    write(OUT/'biohub_metadata_audit.json',dict(passed=True,created=now(),clips=len(image_checks),dtype='uint16',
        native_grids_and_physical_scales_verified=True,full_image_chunk_hashing_repeated=False,checks=image_checks))
    ledger=pd.read_csv(OUT/'dataset_use.csv').fillna('')
    additions=[]
    for source in ['44b6','6bba']:
        sid='biohub_'+source
        if sid in set(ledger.source):continue
        additions.append(dict(source=sid,available=True,hashes_verified=False,images=True,selected=True,
            role='source_only_sparse_supervision',label_quality='experimental_sparse_GEFF',permitted_use='competition training data',
            exclusion='',coordinate_status='all native Zarr grids/dtypes/physical scales verified',alias_group=sid,
            calibration_exposure='public incumbent and historical E teacher exposure persists',
            verification_scope='fresh image metadata and cached GT/graph hashes; full sealed image chunk hashing not repeated'))
    if additions:pd.concat([ledger,pd.DataFrame(additions)],ignore_index=True).to_csv(OUT/'dataset_use.csv',index=False)
    rr=[json.loads(s) for s in (OUT/'runtime_dataset_index.jsonl').read_text().splitlines()]
    errors=[];checks={}
    for field in ['provenance_group','representation_group','source_sha256','label_sha256']:
        mapping=defaultdict(set)
        for r in rr:
            if not r['source'].startswith('synthetic') or field not in r:continue
            mapping[r[field]].add(r['partition'])
        bad={k:sorted(v) for k,v in mapping.items() if len(v)>1};errors.extend((field,k) for k in bad)
        checks[field]=dict(unique_keys=len(mapping),cross_partition_aliases=len(bad))
    assert not errors,errors[:10]
    assert len({r['sample'] for r in rr})==len(rr)
    write(OUT/'partition_audit.json',dict(passed=True,created=now(),records=len(rr),synthetic_checks=checks,
        zoo_scope='same acquisition explicitly shared across six-frame-purged time blocks; not independent embryos',
        zoo_normalization='acquisition-wide unlabeled coordinate IQR; held-out block coordinates contribute to this scale, not to training labels',
        biohub_scope='separate source-only fits; repeated/correlated crops remain exploratory',
        source_and_partition_fields_unchanged=True,index_sha256=sha(OUT/'runtime_dataset_index.jsonl')))

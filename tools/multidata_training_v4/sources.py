"""Read-only source integrity, eligibility, aliases, and frozen pilot protocol."""
from concurrent.futures import ThreadPoolExecutor
import subprocess
import pandas as pd
from .common import *

def snapshot():
    return {root.name:{str(p.relative_to(root)):[p.stat().st_size,p.stat().st_mtime_ns]
            for p in root.rglob('*') if p.is_file()} for root in [V1,STUDIES/'strong-tracker-v2',V3]}

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'preservation_before.json').exists():write(OUT/'preservation_before.json',snapshot())
    protocol=read(REPO/'handover/multidata-training-v4/experiments.json')
    protocol.update(executed_design=dict(
        geometry='relative per-axis IQR and local-neighbor scale; label-blind nearest-history; six daughters; symmetric pair head',
        images='shared triplanar 12x12 encoder, explicit three-frame validity per center; actual parent and daughter crops',
        detection='same shared compact convolutional encoder, center and subvoxel-offset heads, native static patches',
        eligible_zoo=['zebrafish','ascidian'],additional_species_weights={'ascidian':.25},
        zoo_split='contiguous first 70% train; 6-frame purge; next 15% validation; 6-frame purge; last 15% diagnostic test; one acquisition per species',
        synthetic_split='preserve prepared train; split prepared holdout into validation/test by sample hash',
        gate_probability=.5,primary_margin=4.,descriptive_margins=[2.,6.],
        decoder='local binary event-gain maximization with no-op, parent/daughter/displaced-owner conflicts and protected 2-frame incumbent fork windows',
        detector_integration='bounded center refinement then rebuild all graph-dependent evidence using incumbent native and v3 inference; isolated population change test',
        source_calibration='external validation temperature; real folds reused, no independent inner fold claim',
        selection='C4 is prespecified primary external arm; all primary arms frozen before target scoring; seed2 if an external arm qualifies',
        resource_adjustment='29 GiB available disk; compact uint8 patch caches; default full update budgets retained',
        second_seed=314159))
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==protocol
    else:write(OUT/'protocol.json',protocol)
    status('W400',state='running')
    manifest=read(ARCHIVE/'verification.json')['verified']
    checks=[(ARCHIVE/p,r['sha256']) for p,r in manifest.items()]
    checks += [(PREPARED.parent/r['path'],r['sha256']) for r in read(PREPARED.parent/'prepared_checksums.json')]
    def check(item):
        p,h=item
        if sha(p)!=h:raise ValueError('Source hash differs: '+str(p))
        return p.stat().st_size
    with ThreadPoolExecutor(max_workers=4) as pool: sizes=list(pool.map(check,checks))
    lock=read(V3/'selected_prediction_lock.json')
    for n,h in lock['hashes'].items():assert sha(V3/'selected_predictions'/f'{n}.npz')==h
    revision=subprocess.check_output(['git','-C',str(REPO/'work/annotation-selection-v1/official'),'rev-parse','HEAD'],text=True).strip()
    assert revision==protocol['metric_snapshot']
    registry=read(REPO/'handover/multidata-training-v4/dataset_registry.json')['records'];ledger=[]
    for r in registry:
        sid=r['id']; eligible=sid in ['synthetic_static','synthetic_sequences','zoo_zebrafish','zoo_ascidian']
        evidence={'synthetic_static':'archived author CC0 declaration, topic 732103','synthetic_sequences':'archived author CC0 declaration, topic 732103',
            'zoo_zebrafish':'archived organizer permission specifically covers public Zebrahub tracks; topic 734330; same originating authors/site',
            'zoo_ascidian':'originating Figshare article 8223890 CC BY 4.0; exact prepared graph/source CSV conversion audit; Guignard et al. Science 2020'}
        ledger.append(dict(source=sid,available=True,hashes_verified=True,images=r.get('images',False),
            selected=eligible,role=r['role'] if eligible else 'excluded',label_quality=r.get('label_quality','measurements_no_established_links'),
            permitted_use=evidence.get(sid,'unresolved source terms; not trained'),
            exclusion='' if eligible else ('unresolved use and temporal labels' if sid.startswith('riken') else 'originating data-use terms not established in bounded audit'),
            coordinate_status='verified native/pooled grids' if sid.startswith('synthetic') else 'unitless per-axis normalization; no physical/cadence claim',
            alias_group=r.get('alias_group',sid),calibration_exposure='44b6' if sid.startswith('synthetic') else 'not Biohub-calibrated'))
    pd.DataFrame(ledger).to_csv(OUT/'dataset_use.csv',index=False)
    source_files=[REPO/'docs/external-data-guide/dataset_inventory.json',PREPARED.parent/'reference/topic-734330.json',ARCHIVE/'topics/732103.md',
        ARCHIVE/'notebooks/josefreitasalvesneto/biohub-synthetic-dataset/biohub-synthetic-dataset.py',REPO/'tools/biohub_external_data/prepare_data.py']
    write(OUT/'source_manifest.json',dict(created=now(),files_rehashed=len(checks),bytes_rehashed=sum(sizes),
        source_evidence_hashes={str(p.relative_to(REPO)):sha(p) for p in source_files},metric_revision=revision,
        source_partitions=protocol['executed_design'],incumbent_hashes=lock['hashes'],
        exposure=['operational exploratory; embryos repeatedly reused','released simulator calibrated on 44b6',
                  'incumbent public checkpoints and v2 E teachers retain inherited target exposure'],
        legal_scope='dataset-specific evidence recorded; RIKEN excluded; no gated terms accepted'))
    print('Source files rehashed:',len(checks),'bytes:',sum(sizes),flush=True)
    status('W400',state='integrity_complete_baseline_rescore_next')

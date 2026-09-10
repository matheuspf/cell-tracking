"""Fresh deployment under an unknown clip name, with an explicit source fit."""
import subprocess
import sys
from .common import *

def run():
    original='44b6_87bba6c4';alias='unseen_v4_pilot';source='6bba';variant='C4'
    package=OUT/'inference_package';view=OUT/'unknown_name_view';view.mkdir(exist_ok=True)
    link=view/f'{alias}.zarr'
    if not link.exists():link.symlink_to(DATA/'train'/f'{original}.zarr',target_is_directory=True)
    output=OUT/'unknown_name_output';receipt=output/'inference_receipt.json'
    if not receipt.exists():
        if output.exists():raise RuntimeError('Preserve partial unknown-name output before recovery')
        with (OUT/'logs/unknown_name_pilot.log').open('w') as log:
            subprocess.run([str(package/'run.sh'),'--python',sys.executable,'--images',str(view),'--output',str(output),
                '--v1',str(V1),'--v2',str(STUDIES/'strong-tracker-v2'),'--source-model',source,'--variant',variant],
                check=True,stdout=log,stderr=subprocess.STDOUT,cwd='/tmp',timeout=1800)
    r=read(receipt);actual=arrays(output/'predictions'/f'{alias}.npz');expected=arrays(OUT/'predictions/C4'/f'{original}.npz')
    fields={k:bool(np.array_equal(actual[k],expected[k])) for k in ['nodes','edges']}
    import pandas as pd
    csv=pd.read_csv(output/'submission.csv');names=set(csv.dataset)
    passed=all(fields.values()) and names=={alias} and r['source_model']==source
    assert r['package_manifest_sha256']==sha(package/'manifest.json')
    write(OUT/'unknown_name_pilot_receipt.json',dict(completed=True,passed=passed,original_clip=original,alias=alias,
        source_model=source,variant=variant,seconds=r['seconds'],fields_equal=fields,csv_uses_alias=names=={alias},
        weights_loaded=r['graphs'][0]['weights_loaded'],annotation_reads=r['annotation_reads'],
        external_dataset_reads=r['external_dataset_reads'],offline_network_guard=r['offline_network_guard'],
        package_manifest_sha256=r['package_manifest_sha256'],
        scope='Same fixed image under an unfamiliar filename; deployment routing check, not an independent biological example'))
    assert passed,'Renaming the clip changed the frozen deployment policy'
    print('Unknown-name fresh C4 deployment matched the scored graph exactly',flush=True)

if __name__=='__main__':run()

"""Blinded design-based image census pack; manual truth is never fabricated."""
from __future__ import annotations

import numpy as np
import pandas as pd
import zarr

from .common import DATA,OUT,SEED,read_json,stage,write_json


def run(args):
    manifest=read_json(OUT/'fold_manifest.json');rng=np.random.default_rng(SEED)
    root=OUT/'blinded_census';root.mkdir(parents=True,exist_ok=True)
    if (root/'census_design_and_blank_labels.csv').exists():
        if len(list(root.glob('roi_*.npz')))!=48:raise ValueError('Incomplete existing census pack')
        print('Existing census pack retained, including any user-entered labels.',flush=True)
        return
    rows=[]
    for embryo in manifest['embryos']:
        names=[n for n in manifest['expected_samples'] if n.split('_')[0]==embryo]
        for time_stratum,(tlo,thi) in enumerate([(0,33),(33,66),(66,100)]):
            for depth_stratum in range(4):
                # Universe is dataset-weighted single-frame, nonoverlapping grid cores.
                population=len(names)*(thi-tlo)*2*8*8
                selections=rng.choice(population,2,replace=False)
                for selection in selections:
                    ci,ti,zi,yi,xi=np.unravel_index(selection,(len(names),thi-tlo,2,8,8))
                    name=names[ci];t=tlo+ti;z=(depth_stratum*2+zi)*8;y=yi*32;x=xi*32
                    arr=zarr.open_group(DATA/'train'/f'{name}.zarr',mode='r')['0']
                    image=arr[t]
                    lo=np.maximum([z-4,y-16,x-16],0);hi=np.minimum([z+12,y+48,x+48],image.shape)
                    cut=image[tuple(slice(l,h) for l,h in zip(lo,hi))]
                    roi_id=f'roi_{len(rows):03d}'
                    np.savez_compressed(root/(roi_id+'.npz'),image=cut,core_start=np.array([z,y,x])-lo,core_shape=[8,32,32])
                    rows.append(dict(roi_id=roi_id,embryo=embryo,dataset=name,t=t,time_stratum=time_stratum,depth_stratum=depth_stratum,
                                     core_z=z,core_y=y,core_x=x,inclusion_probability=2/population,
                                     sampling_population=population,visible_centers=None,ambiguous_centers=None,second_review=None))
    pd.DataFrame(rows).to_csv(root/'census_design_and_blank_labels.csv',index=False)
    (root/'README.md').write_text('Blinded census pack\n\nEach NPZ contains raw image data, core_start and core_shape. Count all visible cell centers strictly inside the core, using the halo for context. Sparse GT and model decisions are absent. Fill visible_centers and ambiguous_centers in the CSV; second-review a subset. These are dataset-weighted observations, not unique biological cells. Overlap across different released clips is unresolved. No manual labels were supplied; no true-cell prevalence estimate is available.\n')
    write_json(OUT/'quality_audit.json',dict(rois=len(rows),manual_labels_received=0,true_cell_prevalence=None,
               sampling='2 without replacement per embryo x 3 time x 4 depth strata; image-only uniform grid cores',
               candidate_precision_audit='No independent human labels available',raw_coordinate_files_stay_local=True))
    stage('S040','automated_audit_complete_manual_census_unavailable',blinded_rois=len(rows),exact_true_cell_prevalence=None)

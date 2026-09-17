"""Verify real image metadata and pin every compressed time chunk read by a fit."""
from pathlib import Path
import numpy as np
from .common import inputs,RESULTS,WORK,read_json,write_json,sha,digest


def run():
    files=[]
    for i,row in enumerate(inputs(),1):
        root=Path(row['image_path'])
        if sha(root/'zarr.json')!=row['metadata_sha256'] or sha(root/'0/zarr.json')!=row['array_metadata_sha256']:
            raise ValueError('Image metadata differs from inherited verified inputs')
        meta=read_json(root/'0/zarr.json')
        attrs=read_json(root/'zarr.json')['attributes']
        multi=attrs.get('ome',attrs)['multiscales'][0]
        assert [a['name'].lower() for a in multi['axes']]==['t','z','y','x']
        assert meta['shape']==row['image_shape'] and meta['data_type']=='uint16'
        scale=multi['datasets'][0]['coordinateTransformations'][0]['scale'][1:]
        np.testing.assert_array_equal(scale,row['physical_scale'])
        frames={str(t):sha(root/f'0/c/{t}/0/0/0') for t in range(meta['shape'][0])}
        value=dict(dataset=row['dataset'],metadata_sha256=row['metadata_sha256'],
                   frame_sha256=frames,content_sha256=digest(frames))
        write_json(WORK/'image_hashes'/(row['dataset']+'.json'),value,immutable=True)
        files.append({k:v for k,v in value.items() if k!='frame_sha256'})
        if i%25==0:print(f'Image integrity {i}/199',flush=True)
    write_json(RESULTS/'image_manifest.json',dict(clips=files,status='verified',shape='TZYX',
        scales_zyx=[1.625,.40625,.40625],frames=19900,raw_inputs_unchanged=True),immutable=True)
    return files

"""Readiness inventory; no checkpoint or target score deserialization."""
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import time
from pathlib import Path
from .common import ARCH, DATA, OLD, WORK, RESULTS, CONFIG, Blocked, now, sha, read, write


def run(data=DATA):
    started = time.monotonic()
    rows = []
    for image in sorted((data/'train').glob('*.zarr')):
        name = image.stem
        meta = read(image/'0/zarr.json')
        multiscale = read(image/'zarr.json')['attributes']['multiscales'][0]
        axes = [x['name'].lower() for x in multiscale['axes']]
        spacing = multiscale['datasets'][0]['coordinateTransformations'][0]['scale']
        if axes != ['t','z','y','x'] or meta['shape'] != [100,64,256,256] or spacing != [1.,1.625,.40625,.40625]:
            raise Blocked('Unexpected actual image metadata: '+name)
        geff = data/'train'/f'{name}.geff'
        if not (geff/'zarr.json').exists():
            raise Blocked('Missing GEFF: '+name)
        chunks = list((image/'0/c').glob('*/0/0/0'))
        if len(chunks) != 100:
            raise Blocked('Incomplete frame inventory: '+name)
        rows.append(dict(dataset=name, embryo=name[:4], shape=meta['shape'], spacing=spacing,
                         metadata_sha256=sha(image/'0/zarr.json'), geff_metadata_sha256=sha(geff/'zarr.json'),
                         compressed_bytes=sum(p.stat().st_size for p in chunks)))
    counts = {source: sum(r['embryo'] == source for r in rows) for source in ('44b6','6bba')}
    if counts != {'44b6':71,'6bba':128}:
        raise Blocked('Incomplete competition population: '+str(counts))
    # Recover exact historical memberships from the preserved recipe. Verify the
    # stated fixed seeded hashing convention independently; no outcome selection.
    sources = {}
    for source in counts:
        lineage = read(OLD/f'fits/clean/{source}/20260918/upstream/recipe.json')['lineage']
        sources[source] = dict(fit=lineage['training_clip_ids'], calibration=lineage['calibration_clip_ids'])
        for field, part in [('fit','training_clip_ids'),('calibration','calibration_clip_ids')]:
            other = read(OLD/f'fits/clean/{source}/314159/upstream/recipe.json')['lineage'][part]
            if sources[source][field] != other:
                raise Blocked('Historical seed memberships differ')
        if set(sources[source]['fit']) & set(sources[source]['calibration']):
            raise Blocked('Source split overlap')
        if sorted(sources[source]['fit']+sources[source]['calibration']) != [r['dataset'] for r in rows if r['embryo']==source]:
            raise Blocked('Source split population mismatch')
    write(WORK/'source_partitions.json', sources, immutable=True)
    deps = {}
    for package in ('torch','numpy','scipy','zarr','numcodecs','polars','tracksdata','psutil'):
        try: deps[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: deps[package] = None
    architecture = {str(p.relative_to(ARCH)):sha(p) for p in sorted(ARCH.rglob('*.py'))}
    record = dict(status='metadata_passed_duplicate_audit_pending', created_utc=now(),
                  config_sha256=sha(CONFIG), counts=counts, frames=19900, clips=rows,
                  source_partitions_sha256=sha(WORK/'source_partitions.json'),
                  architecture=architecture, dependencies=deps,
                  gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip(),
                  free_bytes=shutil.disk_usage(WORK).free, wall_seconds=time.monotonic()-started,
                  target_labels_read=False, upstream_reused=False)
    write(RESULTS/'preflight.json', record)
    return record

"""Whole-source groups with overlap unions; no random event-row partitioning."""
from collections import defaultdict

from .common import DATA, RESULTS, WORK, digest, inputs, sha, write_json


class Union:
    def __init__(self, values):
        self.parent = {x: x for x in values}

    def root(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def join(self, a, b):
        a, b = self.root(a), self.root(b)
        self.parent[max(a, b)] = min(a, b)


def split_groups(names, shared_frames):
    union = Union(names)
    for members in shared_frames.values():
        for name in members[1:]:
            union.join(members[0], name)
    groups = defaultdict(list)
    for name in names:
        groups[union.root(name)].append(name)
    assignments = {}
    for key, members in sorted(groups.items()):
        partition = 'calibration' if int(digest(key)[:8], 16) % 5 == 0 else 'fit'
        for name in members:
            assignments[name] = dict(overlap_group=key, partition=partition)
    return assignments


def run():
    manifest = dict(schema_version=1, directions={}, minimum_purge_frames=9,
                    temporal_operator='GRU', random_row_split=False,
                    independence_certified=False,
                    limitation='No global acquisition/crop offsets are provided. Exact shared frames are unioned; translated/partially overlapping crops may remain. Whole-clip inner calibration is exploratory.',
                    purge='Whole clips belong to one partition; no within-clip temporal boundary. Known overlapping clips are unioned before partitioning.')
    hashes = defaultdict(list)
    rows = inputs()
    for i, row in enumerate(rows, 1):
        for t in range(row['image_shape'][0]):
            path = DATA / 'train' / f'{row["dataset"]}.zarr/0/c/{t}/0/0/0'
            # Exact compressed-frame identity is a positive overlap witness, never a proof of non-overlap.
            hashes[sha(path)].append(row['dataset'])
        if i % 20 == 0:
            print(f'Overlap fingerprint audit {i}/199 clips', flush=True)
    shared = {k: sorted(set(v)) for k, v in hashes.items() if len(set(v)) > 1}
    for source in ['44b6', '6bba']:
        names = [r['dataset'] for r in rows if r['embryo'] == source]
        source_shared = {k: [n for n in v if n in names] for k, v in shared.items()}
        source_shared = {k: v for k, v in source_shared.items() if v}
        manifest['directions'][source] = split_groups(names, source_shared)
    write_json(WORK / 'overlap_frame_hashes.json', dict(hashes), immutable=True)
    manifest['exact_shared_frame_hashes'] = len(shared)
    manifest['fingerprints_sha256'] = sha(WORK / 'overlap_frame_hashes.json')
    write_json(RESULTS / 'split_manifest.json', manifest, immutable=True)
    print('Source grouping locked; independence remains uncertified', flush=True)

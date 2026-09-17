"""Representative group cycles, source-only hard replay, bounded immutable caches."""
from collections import Counter, OrderedDict, defaultdict
import gzip
import json
import time
import numpy as np
import torch

from .common import WORK, RESULTS, inputs, read_json, write_json, digest
from .scenes import Scenes, augment

TENSOR_KEYS=('features','event_index','incidence','query_voxels','query_time','keep',
             'utility','supported','biological','identity')


class SourceDataset:
    def __init__(self,source,partition='fit',image=True,allow_partial=False):
        self.source,self.partition,self.image=source,partition,image
        self.rows=inputs(source,partition)
        self.anchors={}
        self.groups=defaultdict(list)
        self.random_groups=defaultdict(list)
        self.positive_groups=defaultdict(list)
        self.cache=OrderedDict()
        self.visits=Counter()
        self.anchor_visits=Counter()
        self.scenes=Scenes(self.rows) if image else None
        for row in self.rows:
            path=WORK/'source'/source/row['dataset']/'anchors.json.gz'
            if not path.exists() and allow_partial:
                continue
            with gzip.open(path,'rt') as f:
                anchors=json.load(f)
            for a in anchors:
                key=f'{a["dataset"]}:{a["anchor"]}'
                a={k:v for k,v in a.items() if k not in ('decisions','labels')}
                self.anchors[key]=a
                self.groups[a['group']].append(key)
                if a['random_included']:
                    self.random_groups[a['group']].append(key)
                if a['positive']:
                    self.positive_groups[a['group']].append(key)
        # Negative-only GROUPS, not just negative alternatives of positive groups.
        self.ordinary={k:v for k,v in self.random_groups.items() if k not in self.positive_groups}
        self.positive_keys=sorted(self.positive_groups)
        self.ordinary_keys=sorted(self.ordinary)
        self.random_keys=sorted(self.random_groups)
        self.mined=[]
        if not self.anchors:
            raise ValueError('No prepared source groups available')

    def arrays(self,key):
        if key not in self.cache:
            a=self.anchors[key]
            path=WORK/'source'/self.source/a['dataset']/a['arrays']
            with np.load(path,allow_pickle=False) as f:
                self.cache[key]={k:f[k] for k in TENSOR_KEYS}
            while len(self.cache)>256:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def cycle(self,keys,index,seed,stream):
        if not keys:
            raise ValueError(f'Source {self.source} has no supported {stream} groups')
        epoch,offset=divmod(index,len(keys))
        rng=np.random.default_rng(int(digest([seed,stream,epoch])[:16],16))
        return keys[int(rng.permutation(len(keys))[offset])]

    def samples(self,step,seed,arm):
        result=[]
        for slot in range(32):
            if slot<8:
                stream='positive'
                group=self.cycle(self.positive_keys,step*8+slot,seed,stream)
                pool=self.positive_groups[group]
            elif slot<24:
                stream='random'
                group=self.cycle(self.random_keys,step*16+slot-8,seed,stream)
                pool=self.random_groups[group]
            elif arm=='J_mined' and self.mined:
                stream='mined'
                key=self.mined[(step*8+slot-24)%len(self.mined)]
                group=self.anchors[key]['group']
                pool=[key]
            else:
                stream='confuser'
                group=self.cycle(self.ordinary_keys,step*8+slot-24,seed,stream)
                pool=self.ordinary[group]
            rng=np.random.default_rng(int(digest([seed,step,slot])[:16],16))
            key=pool[int(rng.integers(len(pool)))]
            self.visits[group]+=1
            self.anchor_visits[key]+=1
            result.append(dict(key=key,stream=stream,augmentation_seed=int(rng.integers(2**31)),
                risk_design_weight=(len(pool)*len(self.random_keys)/sum(map(len,self.random_groups.values()))
                                    if stream=='random' else 0.),
                group_probability=1/(len(self.positive_keys) if stream=='positive' else
                    len(self.random_keys) if stream=='random' else len(self.ordinary_keys)),
                conditional_anchor_probability=1/len(pool)))
        return result

    def batch(self,sample,device,training=False):
        started=time.monotonic()
        key=sample['key'];a=self.anchors[key]
        arrays=self.arrays(key)
        raw=None
        if self.image:
            raw=self.scenes.get(a['dataset'],a['anchor'],a['position'],a['time'])
        loaded=time.monotonic()
        batch={k:torch.as_tensor(v,device=device) for k,v in arrays.items()}
        scene=None
        if raw is not None:
            scene=torch.as_tensor(raw,device=device,dtype=torch.float32)/255.
        if device=='cuda':torch.cuda.synchronize()
        transferred=time.monotonic()
        if scene is not None:
            if training:
                scene,batch['query_voxels']=augment(scene,batch['query_voxels'],sample['augmentation_seed'])
        if device=='cuda':torch.cuda.synchronize()
        self.last_batch_timing=dict(loader_seconds=loaded-started,transfer_seconds=transferred-loaded,
                                    augmentation_seconds=time.monotonic()-transferred)
        return batch,scene

    def audit(self):
        inverse=np.array([1/self.anchors[k]['random_probability'] for k in self.anchors
                          if self.anchors[k]['random_included']],np.float64)
        coverage=Counter()
        shapes={r['dataset']:r['image_shape'] for r in self.rows}
        for key,a in self.anchors.items():
            q=self.arrays(key)['query_voxels']
            coverage['queries']+=len(q)
            coverage['inside_fine']+=int((np.abs(q)<=np.array([7.5,31.5,31.5])).all(1).sum())
            coverage['inside_coarse']+=int((np.abs(q)<=np.array([15.,63.,63.])).all(1).sum())
            coverage['missing_temporal_frames']+=sum(not 0<=a['time']+dt<shapes[a['dataset']][0] for dt in range(-3,4))
        return dict(source=self.source,partition=self.partition,anchors=len(self.anchors),
            groups=len(self.groups),positive_groups=len(self.positive_groups),
            negative_only_groups=len(self.ordinary),random_supported_groups=len(self.random_groups),
            random_supported_anchors=len(inverse),inclusion_weights=sorted(set(inverse.tolist())),
            effective_anchor_sample_size=float(inverse.sum()**2/(inverse@inverse)) if len(inverse) else 0.,
            unknown_as_negative_count=0,group_visits=dict(self.visits),anchor_visits=dict(self.anchor_visits),
            scene_coverage=dict(coverage),coarse_outside_policy='Zero spatial query with explicit validity and physical coordinate; no clamping',
            sampling_independent_of_gt_components=True,
            risk_scope='Within supported biological-group sampling design; not a biological prevalence estimate')


def sampling_audit():
    result={}
    for source in ('44b6','6bba'):
        result[source]={p:SourceDataset(source,p,image=False).audit() for p in ('fit','calibration')}
        manifest=read_json(WORK/'source'/source/'manifest.json')
        counts=Counter()
        for row in manifest['clips']:counts.update(row['counts'])
        result[source]['enumeration']=dict(counts)
        result[source]['candidate_coverage']=dict(
            supported_compatible_events=sum(r['covered_events'] for r in manifest['clips']),
            annotated_events=sum(r['annotated_events'] for r in manifest['clips']),
            source_only=True,annotation_shortlisting_at_inference=False)
        result[source]['unknown_masks']=dict(actions=counts['unknown_actions'],
            unknown_only_anchors=counts['unknown_only_anchors'],unknown_as_negative_count=0)
    write_json(RESULTS/'sampling_audit.json',result)
    return result

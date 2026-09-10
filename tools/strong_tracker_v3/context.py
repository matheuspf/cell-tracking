"""Explicit immutable paths. Label-free inference inventory is separate from GT."""
from dataclasses import dataclass, replace
from pathlib import Path
import json

@dataclass(frozen=True)
class RunContext:
    repo: Path
    data: Path
    v1: Path
    v2: Path
    out: Path
    work: Path
    official: Path
    metric_revision: str = '075fc5f5a52d11077f9dc2b074644618f26939e2'

    @classmethod
    def default(cls, **overrides):
        repo = Path(__file__).resolve().parents[2]
        values = dict(repo=repo, data=Path('/kaggle/input/competitions/biohub-cell-tracking-during-development'),
            v1=Path('/kaggle/working/cell-tracking/annotation-selection-v1'),
            v2=Path('/kaggle/working/cell-tracking/strong-tracker-v2'),
            out=Path('/kaggle/working/cell-tracking/strong-tracker-v3'),
            work=repo/'work/strong-tracker-v3', official=repo/'work/annotation-selection-v1/official')
        values.update({k: Path(v).resolve() for k,v in overrides.items() if v is not None})
        return cls(**values)

    @property
    def full(self):
        return self.v1/'public_harmonic_full'

    def incumbent(self, name):
        if Path(name).name != name: raise ValueError('Invalid dataset name')
        return self.v2/'selected_predictions'/f'{name}.npz'

    def samples(self):
        return json.loads((self.out/'inputs.json').read_text())

    def eval_rows(self):
        return json.loads((self.v1/'inventory.json').read_text())

    def check_outputs(self):
        protected = [self.data,self.v1,self.v2,self.repo/'results']
        for target in [self.out,self.work]:
            t=target.resolve()
            for p in protected:
                p=p.resolve()
                if t==p or t in p.parents or p in t.parents:
                    raise ValueError(f'Output overlaps protected input: {p}')
        return self

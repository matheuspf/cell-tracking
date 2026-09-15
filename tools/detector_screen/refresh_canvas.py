"""Refresh embedded benchmark evidence in the managed standalone canvas."""
import json
from pathlib import Path

from .publish import OUT

CANVAS=Path('/home/mpf/.cursor/projects/home-mpf-code-kaggle-cell-tracking/canvases/detector-benchmark-20260914.canvas.tsx')


def main():
    source=json.loads((OUT/'summary.json').read_text())
    fields=['role','embryo','frames','gt','predicted','recall','misses_at_7',
        'edge_endpoint_availability','candidate_ratio_to_incumbent',
        'gains_over_incumbent_at_7','losses_from_incumbent_at_7',
        'gt_edge_endpoints_available','gt_edges_in_panel','budgets']
    data=dict(date=source['date'],selection=source['selection'],
              incumbent_failure_analysis=source['incumbent_failure_analysis'],models=[])
    for model in source['models']:
        data['models'].append(dict(model,summary=[{k:r[k] for k in fields} for r in model['summary']]))
    before=CANVAS.read_text();a=before.index('// DATA_START');b=before.index('// DATA_END')
    block='// DATA_START\nconst DATA: Data = '+json.dumps(data,separators=(',',':'))+';\n'
    CANVAS.write_text(before[:a]+block+before[b:])
    print(CANVAS)


if __name__=='__main__':main()

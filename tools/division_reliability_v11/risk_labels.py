"""Exact metric-risk shortcuts; no biological nondivision labels are created."""
import numpy as np


class Risks:
    def __init__(self,bank,official_labels):
        self.bank=bank;self.labels=official_labels
        self.parent_roles=set()
        for roles in official_labels.roles.values():
            if roles is not None:self.parent_roles.update(roles[0])
        self.components={p:official_labels.components[g] for p,g in official_labels.matches.items()}
        self.stats=dict(official_action_calls=0,known_parent_negative_groups=0,provably_unsupported_groups=0)

    def group(self,g):
        bank=self.bank;labels=self.labels;p=g['parent'];count=len(g['forks'])
        considered=bool({p,*bank.pred[p]}&self.parent_roles)
        if not considered:
            if p in labels.matches and labels.gt.successors(labels.matches[p]):
                self.stats['known_parent_negative_groups']+=1
                return np.zeros(count,np.int8),[]
            components=set()
            for daughter in set(bank.near[p])|set(bank.succ[p]):
                if daughter in self.components:components.add(self.components[daughter])
                else:
                    for q in bank.paths[daughter]:
                        if q in self.components:components.add(self.components[q])
            # Legal complete edits enforce unique ownership. No parent-window
            # support and at most one component across every possible branch
            # excludes all evaluability rules in the pinned metric-risk helper.
            if len(components)<=1:
                self.stats['provably_unsupported_groups']+=1
                return np.full(count,-1,np.int8),[]
        raw=[labels.decision(a) for a in g['forks']];self.stats['official_action_calls']+=len(raw)
        return np.array([r['metric_fork_target'] for r in raw],np.int8),sorted({e for r in raw for e in r['compatible_events']})

"""One canonical full edit and one null per prediction-only anchor."""
from pipeline_error_training.actions import Decision, Alternatives, apply_decisions, fork_support
from pipeline_error_training.bank import EventBank


def close_resources(decision,pred,succ):
    """Own the full old/new official window of every affected existing fork."""
    remove,add=decision.remove,decision.add
    changed={n for e in remove|add for n in e}
    affected=set(changed)
    for _ in range(2):
        affected.update(p for n in tuple(affected) for p in pred[n])
    def after(n):
        return (set(succ[n])-{b for a,b in remove if a==n})|{b for a,b in add if a==n}
    owned=set()
    for p in affected:
        new=after(p)
        if len(succ[p])==2 or len(new)==2:
            children=set(succ[p])|new
            owned.update([p,*pred[p],*children])
            for child in children:owned.update([*succ[child],*after(child)])
    decision.resources=decision.resources|frozenset(('node',n) for n in owned)
    return decision


def canonical_actions(bank, parent, replace=True):
    alternatives = Alternatives(bank, replace=replace)
    seen = set()
    records = []
    for event, features, _ in bank.iter_parent(parent):
        for decision in alternatives.event(event):
            key = (tuple(sorted(decision.remove)), tuple(sorted(decision.add)))
            if key in seen:
                continue
            seen.add(key)
            records.append((decision, features))
    # Full-edit ordering is independent of pair/path enumeration and GT IDs.
    records.sort(key=lambda x: (x[0].kind != 'keep', tuple(sorted(x[0].remove)), tuple(sorted(x[0].add))))
    return records, dict(alternatives.rejections)

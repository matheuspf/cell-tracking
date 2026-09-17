"""One canonical full edit and one null per prediction-only anchor."""
from collections import Counter
import copy
from pipeline_error_training.actions import Decision, Alternatives as OriginalAlternatives, apply_decisions, fork_support
from pipeline_error_training.bank import EventBank


class Alternatives(OriginalAlternatives):
    def event(self,event):
        """Exact inherited families, with structural rather than SHA dedup.

        Persisted decision keys remain unchanged. Computing their recursive JSON
        hash twice for every temporary alternative dominated crowded inference.
        """
        p,a,b,qa,qb=map(int,event)
        yield Decision('keep',tuple(event),frozenset(),frozenset(),(),(),(),frozenset())
        seen=set()
        families=[('continuation_a',{p:(a,)}),('continuation_b',{p:(b,)})]
        fork={p:(a,b)}
        for d,q in ((a,qa),(b,qb)):
            if q>=0:fork[d]=(q,)
        families.append(('division',fork))
        for d,other in ((a,b),(b,a)):
            if self.pred[other] and self.pred[other][0]!=p:
                families.append(('two_parents',{p:(d,),self.pred[other][0]:(other,)}))
        for kind,desired in families:
            for decision in self.complete(event,desired,kind):
                key=(decision.remove,decision.add)
                if key not in seen:
                    seen.add(key)
                    yield decision


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
    # The inherited constructor materializes all original edges and protection
    # windows. They are immutable for this bank, so build them once rather than
    # once per anchor. Each call owns its own rejection counter.
    if not hasattr(bank, '_v2_alternatives'):
        bank._v2_alternatives = {}
    if replace not in bank._v2_alternatives:
        bank._v2_alternatives[replace] = Alternatives(bank, replace=replace)
    alternatives = copy.copy(bank._v2_alternatives[replace])
    alternatives.rejections = Counter()
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

"""Atomic event edits and bounded MILP conflict resolution, with exact no-op."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .common import adjacency


def logit(p):
    # Owner alternatives call this millions of times with Python scalars.
    # Scalar clamping is numerically identical and avoids NumPy dispatch per
    # donor cost; vector candidate scores retain the original NumPy path.
    p=min(max(p,1e-6),1-1e-6) if isinstance(p,(float,int)) else np.clip(p,1e-6,1-1e-6)
    return np.log(p/(1-p))


def protected_external_owners(parent,daughters,incoming,evidence,probability_floor=.98,distance_ceiling_um=2.):
    """Inspect each actual owner; a free sister cannot mask a 1um owner."""
    protected=set()
    for daughter in daughters:
        owner=incoming.get(daughter)
        if owner is None or owner==parent:continue
        item=evidence.get((owner,daughter))
        if item is None or item.get('probability') is None or item.get('distance_um') is None:
            protected.add(owner);continue
        p,d=float(item['probability']),float(item['distance_um'])
        if not np.isfinite([p,d]).all() or not 0<=p<=1 or d<0:raise ValueError('invalid owner evidence')
        if p>=probability_floor and d<=distance_ceiling_um:protected.add(owner)
    return protected


@dataclass
class Action:
    candidate: int
    value: float
    remove: set
    add: set
    resources: set
    owners: list
    kind: str='division'


class BoundedActionComponents:
    """Stream full conflict closure while discarding oversized action payloads.

    Resource ownership remains represented after abstention, so a later bridge
    cannot resurrect a partial edit from an oversized component. Memory is O(node
    resources + bounded eligible alternatives), rather than O(all event rows).
    """
    def __init__(self,max_alternatives=256,max_buffered=50000):
        self.limit=max_alternatives;self.max_buffered=max_buffered
        self.parent=[];self.members={};self.count={};self.owner={};self.buffered=0
        self.actions_seen=0;self.workspace_abstentions=0
    def root(self,k):
        while self.parent[k]!=k:self.parent[k]=self.parent[self.parent[k]];k=self.parent[k]
        return k
    def add(self,action):
        self.actions_seen+=1
        roots={self.root(self.owner[r]) for r in action.resources if r in self.owner}
        if roots:
            root=min(roots);old=[self.members[r] for r in sorted(roots)]
            self.buffered-=sum(len(x) for x in old if x is not None)
            count=1+sum(self.count[r] for r in roots)
            blocked=any(x is None for x in old) or count>self.limit
            members=None if blocked else [a for xs in old for a in xs]+[action]
            for r in sorted(roots):
                if r!=root:self.parent[r]=root;del self.members[r];del self.count[r]
        else:
            root=len(self.parent);self.parent.append(root);count=1;members=[action]
        self.members[root]=members;self.count[root]=count
        self.buffered+=len(members) if members is not None else 0
        for r in sorted(action.resources):self.owner[r]=root
        if self.buffered>self.max_buffered:
            # The complete largest component abstains; its resource closure is
            # retained. This is a logged workspace safeguard, not truncation.
            key=max((r for r,m in self.members.items() if m is not None),key=lambda r:len(self.members[r]))
            self.buffered-=len(self.members[key]);self.members[key]=None;self.workspace_abstentions+=1
    def finish(self):
        actions=[a for r in sorted(self.members) if self.members[r] is not None for a in self.members[r]]
        rejected=[r for r,m in self.members.items() if m is None]
        return actions,dict(streamed_actions=self.actions_seen,buffered_actions=len(actions),
            oversized_stream_components=len(rejected),actions_in_abstained_components=sum(self.count[r] for r in rejected),
            workspace_abstentions=self.workspace_abstentions,resource_keys=len(self.owner),
            max_component_alternatives=self.limit,max_buffered_actions=self.max_buffered)


def legal_edges(nodes, edges):
    ix,pred,succ=adjacency(nodes,np.asarray(sorted(edges),np.int64).reshape(-1,2))
    return all(len(x)<=1 for x in pred) and all(len(x)<=2 for x in succ) and all(nodes[ix[b],1]==nodes[ix[a],1]+1 for a,b in edges)


def solve_actions(actions,max_alternatives=256,time_limit=2.,max_changed_edges=None):
    """Set-packing over complete edits; all-zero explicitly represents no-op."""
    resources=defaultdict(list)
    for k,a in enumerate(actions):
        for r in sorted(a.resources):resources[r].append(k)
    roots=list(range(len(actions)))
    def root(k):
        while roots[k]!=k:roots[k]=roots[roots[k]];k=roots[k]
        return k
    for ks in resources.values():
        for k in ks[1:]:roots[root(k)]=root(ks[0])
    components=defaultdict(list)
    for k in range(len(actions)):components[root(k)].append(k)
    selected=[];receipt=Counter(components=len(components),actions=len(actions))
    for ks in components.values():
        if len(ks)>max_alternatives:receipt['oversized_abstentions']+=1;continue
        rr=[];cc=[];active=defaultdict(list)
        for j,k in enumerate(ks):
            for r in sorted(actions[k].resources):active[r].append(j)
        for i,js in enumerate(active.values()):
            rr.extend([i]*len(js));cc.extend(js)
        matrix=coo_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(active),len(ks))).tocsc()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='Unrecognized options detected.*')
            res=milp(-np.array([actions[k].value-1e-9 for k in ks]),integrality=np.ones(len(ks)),
                     bounds=Bounds(0.,1.),constraints=LinearConstraint(matrix,0.,1.),
                     options={'time_limit':time_limit,'mip_rel_gap':0.,'presolve':True,'threads':1})
        if not res.success or res.x is None:receipt['solver_abstentions']+=1;continue
        chosen=[ks[j] for j in np.flatnonzero(res.x>.5) if actions[ks[j]].value>1e-9]
        if max_changed_edges is not None and sum(len(actions[k].remove)+len(actions[k].add) for k in selected+chosen)>max_changed_edges:
            receipt['edit_cap_abstentions']+=1;continue
        selected.extend(chosen);receipt['solved_components']+=1
    receipt['accepted_actions']=len(selected)
    return selected,dict(receipt)


def divisions(nodes,edges,hypotheses,probability,native,scale,threshold=.2,existing_only=False,replace=False,margin=1.5):
    ix,pred,succ=adjacency(nodes,edges)
    old={(ix[int(a)],ix[int(b)]) for a,b in edges}
    pos=nodes[:,2:]*np.asarray(scale)
    native_p={tuple(map(int,p)):float(f[0]) for p,f in zip(native['pairs'],native['edge_features']) if not f[1]}
    incoming={j:p[0] for j,p in enumerate(pred) if p}
    evidence={(i,j):dict(probability=native_p.get((i,j)),distance_um=float(np.linalg.norm(pos[j]-pos[i]))) for i,j in old}
    alternatives=defaultdict(list)
    for (i,j),p in native_p.items():alternatives[i].append((p,j))
    for i in alternatives:alternatives[i].sort(reverse=True)
    collector=BoundedActionComponents();reject=Counter()
    original_fork_probability={}
    if replace:
        for i,children in enumerate(succ):
            if len(children)!=2:continue
            a,b=sorted(children)
            if len(succ[a])>1 or len(succ[b])>1:
                original_fork_probability[i]=.99;continue
            qa=succ[a][0] if succ[a] else -1;qb=succ[b][0] if succ[b] else -1
            ks=np.flatnonzero(hypotheses['events'][:,0]==i)
            exact=ks[np.all(hypotheses['events'][ks]==[i,a,b,qa,qb],axis=1)]
            original_fork_probability[i]=max((float(probability[k]) for k in exact),default=.99)
    def fork_cost(i):
        return max(0.,float(logit(original_fork_probability.get(i,.99))-logit(threshold)))
    existing_triples=None
    if existing_only:
        # Reuse only this pure explicit-argument legacy helper. Its current
        # incumbent adjacency and centers are supplied; no legacy cache is read.
        from strong_tracker_v2.hypotheses import build as existing_hypotheses
        base=dict(nodes=nodes,edges=edges,features=native['node_features'])
        existing_triples={tuple(map(int,t)) for t in existing_hypotheses(base,native)['triples']}

    def complete(k,i,children,paths,value,kind='division'):
        desired={i:tuple(children)}
        owners=[]
        if kind=='division':
            if len(succ[i])==2:
                cost=fork_cost(i);value-=cost
                owners.append(dict(owner=i,alternative='fork_replacement',cost=cost))
            for parent in pred[i]:
                if len(succ[parent])==2:
                    if not replace:return None,'adjacent_incumbent_fork_boundary'
                    desired[parent]=(i,);cost=fork_cost(parent);value-=cost
                    owners.append(dict(owner=parent,alternative='shift_to_continuation',target=i,cost=cost))
        for d,q in paths:
            if q>=0:
                if len(succ[d])==2:
                    if not replace:return None,'daughter_fork_boundary'
                    cost=fork_cost(d);value-=cost
                    owners.append(dict(owner=d,alternative='shift_to_continuation',target=q,cost=cost))
                desired[d]=(q,)
        occupied={j for ch in desired.values() for j in ch}
        if sum(len(ch) for ch in desired.values())!=len(occupied):return None,'merged_paths'
        displaced={incoming[j] for j in occupied if j in incoming and incoming[j] not in desired}
        protected=protected_external_owners(i,children,incoming,evidence)
        # Apply the same individual check to future daughter-path ownership.
        for d,q in paths:
            if q>=0:protected|=protected_external_owners(d,[q],incoming,evidence)
        for o in sorted(displaced):
            if len(succ[o])==2 and not replace:return None,'owner_fork_boundary'
            stolen=[j for j in succ[o] if j in occupied]
            oldp=max([native_p.get((o,j),0.) for j in stolen] or [0.])
            if len(succ[o])==2:
                remaining=[j for j in succ[o] if j not in occupied]
                if remaining:
                    target=remaining[0];desired[o]=(target,);occupied.add(target)
                    cost=fork_cost(o)+max(.5,float(logit(max(oldp,.01))-logit(native_p.get((o,target),.01))))
                    owners.append(dict(owner=o,alternative='fork_to_continuation',target=target,cost=cost));value-=cost
                    continue
            # A donor can use a proven free continuation (or a target released by
            # this complete edit); no guessed displaced-node correspondence.
            donor=None
            for p,j in alternatives[o]:
                if j in occupied:continue
                owner=incoming.get(j)
                released=owner in desired and j not in desired[owner]
                if owner is None or released or owner==o:
                    if p>=max(.5,oldp-.05):donor=(p,j);break
            if donor is None:
                if o in protected:return None,'protected_individual_owner'
                desired[o]=();cost=2.+max(0.,float(logit(oldp)))
                owners.append(dict(owner=o,alternative='termination',cost=cost))
            else:
                p,j=donor;desired[o]=(j,);occupied.add(j)
                cost=max(.25,float(logit(max(oldp,.01))-logit(p)))
                owners.append(dict(owner=o,alternative='continuation',target=j,cost=cost))
            value-=cost
        remove={(s,j) for s in desired for j in succ[s]}
        add={(s,j) for s,ch in desired.items() for j in ch}
        common=remove&add;remove-=common;add-=common
        if not remove and not add:return None,'no_op'
        if value<=0:return None,'owner_cost_or_margin'
        # Sources, old/new targets and branch evidence are all conflict resources.
        # This also prevents repeated neighboring-frame versions of one event.
        resources={('s',s) for s in desired}|{('t',j) for _,j in remove|add}
        resources|={('event_node',s) for s in [i,*children,*pred[i],*[q for _,q in paths if q>=0]]}
        return Action(k,float(value),remove,add,resources,owners,kind),None

    scores=logit(np.asarray(probability))-float(logit(threshold))-margin
    for k in np.flatnonzero(scores>0):
        i,a,b,qa,qb=map(int,hypotheses['events'][k])
        if existing_only:
            if (i,a,b) not in existing_triples:continue
            chosen_a=succ[a][0] if len(succ[a])==1 else -1
            chosen_b=succ[b][0] if len(succ[b])==1 else -1
            if (qa,qb)!=(chosen_a,chosen_b):continue
        if len(succ[i])==2 and not replace:continue
        action,why=complete(k,i,(a,b),[(a,qa),(b,qb)],float(scores[k]))
        if action is None:reject[why]+=1
        else:collector.add(action)
    if replace:
        # Suppress a low-confidence incumbent fork by scoring its complete
        # one-daughter continuation; this is a separate declared arm.
        for i,p in original_fork_probability.items():
            value=float(logit(threshold)-logit(p))-margin
            if value<=0:continue
            for d in succ[i]:
                action,why=complete(-1,i,(d,),[],value+.1*native_p.get((i,d),0.),'suppression')
                if action is not None:collector.add(action)
    actions,stream_receipt=collector.finish()
    chosen,receipt=solve_actions(actions,max_changed_edges=max(2,int(.02*len(edges))))
    receipt.update(stream_receipt)
    selected=set(old);ledger=[]
    for k in chosen:
        a=actions[k];selected.difference_update(a.remove);selected.update(a.add)
        convert=lambda es:[[int(nodes[i,0]),int(nodes[j,0])] for i,j in sorted(es)]
        owners=[]
        for item in a.owners:
            item=dict(item,owner=int(nodes[item['owner'],0]))
            if 'target' in item:item['target']=int(nodes[item['target'],0])
            owners.append(item)
        affected=sorted({s for edge in a.remove|a.add for s in edge})
        native_support=[dict(source_id=int(nodes[i,0]),target_id=int(nodes[j,0]),probability=native_p.get((i,j)),
            probability_missing=(i,j) not in native_p,distance_um=float(np.linalg.norm(pos[j]-pos[i]))) for i,j in sorted(a.remove|a.add)]
        ledger.append(dict(candidate=a.candidate,value=a.value,kind=a.kind,removed_edges=convert(a.remove),added_edges=convert(a.add),
            canonical_nodes=[[int(v) for v in nodes[s]] for s in affected],owner_alternatives=owners,native_support=native_support,
            event_probability=float(probability[a.candidate]) if a.candidate>=0 else None,solver_status='optimal'))
    if selected==old:out=edges.copy()
    else:out=np.asarray(sorted((int(nodes[i,0]),int(nodes[j,0])) for i,j in selected),np.int64).reshape(-1,2)
    if not legal_edges(nodes,set(map(tuple,out))):raise RuntimeError('Combined event edits violate graph contract')
    receipt.update(rejected=dict(reject),edges_removed=len(old-selected),edges_added=len(selected-old),
                   forks_before=sum(len(s)==2 for s in succ),forks_after=sum(v==2 for v in Counter(a for a,b in out).values()))
    return out,receipt,ledger

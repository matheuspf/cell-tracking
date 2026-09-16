"""Image-only graph serialization matching the persisted delta representation."""
import numpy as np


def delta_order(nodes,base_nodes):
    positions={int(n[0]):i for i,n in enumerate(nodes)};old=set(map(int,base_nodes[:,0]))
    assert len(positions)==len(nodes)
    for n in base_nodes:
        if int(n[0]) in positions:assert np.array_equal(n,nodes[positions[int(n[0])]])
    order=[positions[int(n[0])] for n in base_nodes if int(n[0]) in positions]
    order.extend(i for i,n in enumerate(nodes) if int(n[0]) not in old)
    assert len(order)==len(nodes)
    return nodes[np.asarray(order,dtype=np.int64)]

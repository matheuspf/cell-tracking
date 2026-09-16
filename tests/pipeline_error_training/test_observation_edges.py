from pipeline_error_training.observation_edges import ObservationEdges


def test_incident_and_raw_queries_use_ids_and_keep_all_incident_edges():
    index = ObservationEdges([(10,30),(30,42),(30,70),(90,100)],[(2,8),(8,15),(8,21)])
    assert index.removed({30:800,42:801})=={(10,30),(30,42),(30,70)}
    assert index.removed({99:900})==set()
    assert index.restored([2,8,15],1000)=={(1002,1008),(1008,1015)}
    assert index.restored([2,15],1000)==set()

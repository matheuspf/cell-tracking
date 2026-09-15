"""Protect the observation graph, physical matching gate, and full score boundary."""
import csv

import numpy as np
import pytest

from tools.cellpose_ultrack.track import validate_graph, write_csv


def division_graph():
    # Deliberately nonconsecutive, large observation IDs must survive CSV export.
    nodes = np.array([[10000001, 0, 8, 16, 16], [20000009, 1, 8, 16, 16],
                      [30000017, 2, 8, 16, 13], [30000036, 2, 8, 16, 19],
                      [40000008, 3, 8, 16, 12], [40000021, 3, 8, 16, 20]], dtype=np.int64)
    edges = np.array([[10000001, 20000009], [20000009, 30000017], [20000009, 30000036],
                      [30000017, 40000008], [30000036, 40000021]], dtype=np.int64)
    return nodes, edges


def test_csv_preserves_observation_identity_and_both_daughters(tmp_path):
    nodes, edges = division_graph()
    validate_graph(nodes, edges, (4, 16, 32, 32))
    path = tmp_path / "submission.csv"
    write_csv(path, "test_clip", nodes, edges)
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(r["node_id"]) for r in rows if r["row_type"] == "node"] == nodes[:, 0].tolist()
    daughters = [int(r["target_id"]) for r in rows if r["source_id"] == "20000009"]
    assert daughters == [30000017, 30000036]


@pytest.mark.parametrize("kind", ["duplicate", "gap", "missing_parent", "merge", "three_daughters"])
def test_invalid_temporal_graphs_cannot_reach_scoring(kind):
    nodes, edges = division_graph()
    if kind == "duplicate":
        edges = np.vstack([edges, edges[0]])
    elif kind == "gap":
        edges = np.vstack([edges, [10000001, 30000017]])
    elif kind == "missing_parent":
        edges = np.vstack([edges, [98765, 20000009]])
    elif kind == "merge":
        edges = np.vstack([edges, [30000017, 40000021]])
    else:
        nodes = np.vstack([nodes, [30000099, 2, 8, 16, 24]])
        edges = np.vstack([edges, [20000009, 30000099]])
    with pytest.raises(ValueError):
        validate_graph(nodes, edges, (4, 16, 32, 32))


def test_official_metric_includes_division_and_total_node_adjustment():
    pytest.importorskip("tracksdata")
    from tools.annotation_selection.metric_adapter import aggregate, evaluate_graph
    nodes, edges = division_graph()
    row, _, _ = evaluate_graph("division", nodes, edges, nodes, edges, [1.625, .40625, .40625], 6)
    assert aggregate([row], ["division"])["score"] == pytest.approx(1.1)
    missing = edges[~np.all(edges == [20000009, 30000036], axis=1)]
    broken, _, _ = evaluate_graph("division", nodes, missing, nodes, edges, [1.625, .40625, .40625], 6)
    assert (broken["edge_tp"], broken["edge_fn"], broken["division_tp"]) == (4, 1, 0)
    # Unmatched nodes are not edge FP, but they must affect the full score.
    extra = np.array([[99900000 + i, i % 4, 40, 40, 40] for i in range(6)], dtype=np.int64)
    over, _, _ = evaluate_graph("division", np.vstack([nodes, extra]), edges, nodes, edges,
                               [1.625, .40625, .40625], 6)
    assert over["edge_fp"] == 0
    assert aggregate([over], ["division"])["score"] == pytest.approx(1.0)


def test_match_audit_enforces_physical_z_spacing_and_time():
    pytest.importorskip("tracksdata")
    from tools.cellpose_ultrack.evaluate import assert_matching
    nodes, _ = division_graph()
    gt = nodes.copy()
    gt[0, 2] += 5  # 8.125 micrometers, despite only five native voxels.
    with pytest.raises(ValueError, match="7 micrometer"):
        assert_matching({10000001: 10000001}, nodes, gt, [1.625, .40625, .40625])
    gt = nodes.copy()
    gt[0, 1] += 1
    with pytest.raises(ValueError, match="timepoints"):
        assert_matching({10000001: 10000001}, nodes, gt, [1.625, .40625, .40625])


def test_hierarchy_filter_handles_six_voxel_component():
    pytest.importorskip("ultrack")
    import higra as hg
    from ultrack.core.segmentation.hierarchy import create_hierarchies
    from ultrack.utils import labels_to_contours
    labels = np.zeros((1, 12, 32, 32), dtype=np.uint32)
    labels[0, 4:8, 12:20, 12:20] = 1
    labels[0, 1, 1, 1:7] = 2
    foreground, contours = labels_to_contours(labels)

    def materialize(factor):
        return [node for hierarchy in create_hierarchies(
            foreground[0], contours[0], hierarchy_fun=hg.watershed_hierarchy_by_area,
            min_area=20, min_area_factor=factor, max_area=100000, min_frontier=.05,
        ) for node in hierarchy.nodes]

    with pytest.raises(RuntimeError, match="Region too small"):
        materialize(4.)
    assert materialize(2.5)


@pytest.mark.parametrize("appearance,division,second_iou,expected_fork", [
    (-.001, -.001, .2, True),
    (-.1, -.1, .2, True),
    (-.001, -.011, .2, False),
    (-.001, -.011, .4, True),
    (-.001, -1.01, .4, False),
])
def test_real_solver_fork_competes_with_a_new_track(appearance, division, second_iou, expected_fork):
    """Check event-cost semantics in the actual CBC objective, not a mock formula."""
    pytest.importorskip("ultrack")
    from ultrack.config.config import TrackingConfig
    from ultrack.core.solve.solver.mip_solver import MIPSolver

    solver = MIPSolver(TrackingConfig(solver_name="CBC", n_threads=1, time_limit=10,
                                     solution_gap=0, appear_weight=appearance,
                                     disappear_weight=-.001, division_weight=division,
                                     link_function="power", power=4, bias=0))
    solver._model.verbose = 0
    ids = np.array([101, 205, 309])
    solver.add_nodes(ids, np.array([True, False, False]), np.array([False, True, True]))
    solver.add_edges(np.array([101, 101]), np.array([205, 309]), np.array([.6, second_iou]))
    solver.set_standard_constraints()
    # Keep both child observations: the only question is a fork versus an appearance.
    solver.enforce_nodes_solution_value(ids, "node", True)
    solver.optimize()
    assert (solver._divisions[0].x > .5) == expected_fork
    assert sum(e.x > .5 for e in solver._edges) == (2 if expected_fork else 1)
    assert (solver._appearances[2].x > .5) != expected_fork


@pytest.mark.parametrize("case,expected_fork", [
    ("both_continue", True),
    ("one_missing", False),
    ("candidate_not_selected", False),
    ("clip_end", True),
    ("right_anchor_both_continue", True),
    ("right_anchor_one_missing", False),
    ("uncommitted_overlap", True),
])
def test_real_solver_daughter_persistence_and_right_seams(case, expected_fork):
    pytest.importorskip("ultrack")
    from ultrack.config.config import TrackingConfig
    from ultrack.core.solve.solver.mip_solver import MIPSolver
    from tools.cellpose_ultrack.learned_divisions import attach_pair_objective
    from tools.cellpose_ultrack.persistent_divisions import attach_persistence
    solver = MIPSolver(TrackingConfig(solver_name="CBC", n_threads=1, time_limit=10,
                                     solution_gap=0, appear_weight=-.001,
                                     disappear_weight=-.001, division_weight=-.011))
    solver._model.verbose = 0
    internal = case in ("both_continue", "one_missing", "candidate_not_selected")
    ids = np.array([101,205,309,410,511] if internal else [101,205,309])
    times = {101:0,205:1,309:1,410:2,511:2}
    end = 2 if internal else 1
    solver.add_nodes(ids, np.array([times[n] == 0 for n in ids]), np.array([times[n] == end for n in ids]))
    es = [(101,205),(101,309)]
    if internal:
        es.append((205,410))
        if case != "one_missing":
            es.append((309,511))
    es = np.asarray(es)
    solver.add_edges(es[:,0], es[:,1], np.full(len(es), .6))
    solver.set_standard_constraints()
    solver.enforce_nodes_solution_value(ids, "node", True)
    if case == "candidate_not_selected":
        solver.enforce_edges_solution_value(np.array([309]), np.array([511]), False)
    attach_pair_objective(solver, np.array([[101,205,309]]), np.array([.04]))
    committed = []
    if case.startswith("right_anchor"):
        committed = [(410,2,205),(511,2,309 if case.endswith("both_continue") else -1)]
    receipt = attach_persistence(solver, times, 1 if case == "clip_end" else 3, committed)
    solver.optimize()
    assert (solver._divisions[0].x > .5) == expected_fork
    if case.startswith("right_anchor"):
        assert receipt["right_boundary_constraints"] == 2
    elif case == "clip_end":
        assert receipt["clip_final_exemptions"] == 2
    elif case == "uncommitted_overlap":
        assert receipt["uncommitted_overlap_deferrals"] == 2


@pytest.mark.parametrize("both_continue", [False, True])
def test_real_solver_preserves_inherited_fork_at_left_seam(both_continue):
    pytest.importorskip("ultrack")
    import mip
    from ultrack.config.config import TrackingConfig
    from ultrack.core.solve.solver.mip_solver import MIPSolver
    from tools.cellpose_ultrack.learned_divisions import attach_pair_objective
    from tools.cellpose_ultrack.persistent_divisions import attach_persistence
    solver = MIPSolver(TrackingConfig(solver_name="CBC", n_threads=1, time_limit=10))
    solver._model.verbose = 0
    ids = np.array([205,309,410,511])
    solver.add_nodes(ids, np.array([True,True,False,False]), np.array([False,False,True,True]))
    es = np.array([(205,410),(309,511)] if both_continue else [(205,410)])
    solver.add_edges(es[:,0], es[:,1], np.full(len(es), .6))
    solver.set_standard_constraints()
    solver.enforce_nodes_solution_value(ids, "node", True)
    attach_pair_objective(solver, np.empty((0,3),np.int64), np.empty(0))
    record = attach_persistence(solver, {101:0,205:1,309:1,410:2,511:2}, 3,
                                [(101,0,-1),(205,1,101),(309,1,101)])
    assert record["inherited_daughter_constraints"] == 2
    status = solver._model.optimize()
    assert status == (mip.OptimizationStatus.OPTIMAL if both_continue else mip.OptimizationStatus.INFEASIBLE)


@pytest.mark.parametrize("second_iou,bonus,allowed,expected_fork", [
    (.2, .04, True, True),
    (.4, -.04, True, False),
    (.4, 0., True, True),
    (.4, .04, False, False),
])
def test_real_solver_uses_evidence_for_the_actual_daughter_pair(second_iou, bonus, allowed, expected_fork):
    pytest.importorskip("ultrack")
    from ultrack.config.config import TrackingConfig
    from ultrack.core.solve.solver.mip_solver import MIPSolver
    from tools.cellpose_ultrack.learned_divisions import attach_pair_objective
    solver = MIPSolver(TrackingConfig(solver_name="CBC", n_threads=1, time_limit=10,
                                     solution_gap=0, appear_weight=-.001,
                                     disappear_weight=-.001, division_weight=-.011,
                                     link_function="power", power=4, bias=0))
    solver._model.verbose = 0
    ids = np.array([101,205,309])
    solver.add_nodes(ids, np.array([True,False,False]), np.array([False,True,True]))
    solver.add_edges(np.array([101,101]), np.array([205,309]), np.array([.6,second_iou]))
    solver.set_standard_constraints()
    solver.enforce_nodes_solution_value(ids, "node", True)
    pairs = np.array([[101,205,309]],np.int64) if allowed else np.empty((0,3),np.int64)
    values = np.array([bonus]) if allowed else np.empty(0)
    receipt = attach_pair_objective(solver,pairs,values)
    solver.optimize()
    assert receipt["pair_variables"] == int(allowed)
    assert (solver._divisions[0].x > .5) == expected_fork
    assert sum(e.x > .5 for e in solver._edges) == (2 if expected_fork else 1)

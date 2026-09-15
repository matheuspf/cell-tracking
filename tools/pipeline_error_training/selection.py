"""Source-only nomination under the already frozen full-graph screen rules."""
from pathlib import Path

import numpy as np

from .common import RESULTS, WORK, inputs, read_json, sha, verified_evidence, write_json


def identity_loss(source, arm, seed=20260915):
    """Held-source lineage-group NLL versus the unchanged native link offset."""
    from .guard import install
    # Called in a dedicated directional subprocess by the command below.
    install(source=source)
    losses, controls = [], []
    for row in inputs():
        if row['embryo'] != source:
            continue
        table = WORK/'calibration'/arm/source/str(seed)/f'{row["dataset"]}.npz'
        if not table.exists():
            continue
        with np.load(table, allow_pickle=False) as arrays:
            values, labels, groups = arrays['pair_score'], arrays['pair_y'], arrays['pair_group']
        with np.load(WORK/'source'/source/'pairs'/f'{row["dataset"]}.npz', allow_pickle=False) as arrays:
            index = arrays['index']
        native = verified_evidence(row)['edge_features'][index, 23]
        loss = np.logaddexp(0., values)-labels*values
        baseline = np.logaddexp(0., native)-labels*native
        for group in np.unique(groups):
            selected = groups == group
            losses.append(float(loss[selected].mean())); controls.append(float(baseline[selected].mean()))
    receipt = dict(status='measured', source=source, arm=arm, seed=seed, groups=len(losses),
        raw_source_grouped_pair_nll=float(np.mean(losses)) if losses else None,
        unchanged_native_offset_grouped_pair_nll=float(np.mean(controls)) if controls else None,
        calibration_applied=False, source_only=True, no_target_scores_read=True,
        baseline='Fixed native logits on P0 candidate pairs, not a new fitted control')
    write_json(WORK/'training'/arm/source/str(seed)/'identity_diagnostic.json', receipt, immutable=True)
    return receipt


def directional(arm, source):
    model = WORK/'training'/arm/source/'20260915'
    screen_path = WORK/'source_screen'/arm/source/'20260915/summary.json'
    if not (model/'frozen_package.json').exists() or not screen_path.exists():
        return dict(qualified=False, reason='Complete source fit, calibration or graph replay unavailable')
    screen = read_json(screen_path)
    calibration = read_json(model/'calibration.json')
    rule = read_json(RESULTS/'source_screen_lock.json')['qualification']
    graph_safe = (screen['delta'] >= -rule['max_source_score_regression'] and
                  screen['measured']['division_fp'] <= screen['baseline']['division_fp']+rule['max_extra_division_fp'])
    distinct = screen['delta'] > 1e-12
    control = None
    better_loss = False
    if arm in ['D10_adapted', 'D20_temporal']:
        control = 'D10_frozen' if arm=='D10_adapted' else 'D20_compact'
        reference = WORK/'training'/control/source/'20260915/calibration.json'
        if reference.exists():
            current = calibration['source_grouped_decision']['before']['loss']
            baseline = read_json(reference)['source_grouped_decision']['before']['loss']
            better_loss = current is not None and baseline is not None and current < baseline-1e-9
    elif arm == 'A10':
        path = model/'identity_diagnostic.json'
        if path.exists():
            diagnostic = read_json(path)
            a, b = diagnostic['raw_source_grouped_pair_nll'], diagnostic['unchanged_native_offset_grouped_pair_nll']
            better_loss = a is not None and b is not None and a < b-1e-9
            control = diagnostic['baseline']
    elif arm == 'O10_swap':
        # P0's deterministic keep decision is the graph control. A finite held
        # selector loss alone cannot qualify a selector without a source benefit.
        distinct = distinct or (screen['measured']['matched_nodes'] > screen['baseline']['matched_nodes'])
        control = 'P0 unchanged observations'
    qualified = graph_safe and (better_loss or distinct)
    return dict(qualified=qualified, source=source, arm=arm, source_delta=screen['delta'],
        source_graph_safe=graph_safe, source_loss_better_than_control=better_loss,
        distinct_source_graph_tradeoff=distinct, control=control,
        source_division_fp_change=screen['measured']['division_fp']-screen['baseline']['division_fp'],
        screen_sha256=sha(screen_path), calibration_sha256=sha(model/'calibration.json'),
        model_sha256=sha(model/'model.pt'), source_independence_certified=False)


def nominate():
    for queue in ['queue', 'observation_queue']:
        if read_json(WORK/queue/'progress.json')['status'] != 'complete':
            raise RuntimeError('Every independent primary lane must finish or record failure before nomination')
    candidates = {}
    for arm in ['D10_adapted', 'D20_temporal', 'A10', 'O10_swap']:
        directions = {s: directional(arm, s) for s in ['44b6', '6bba']}
        candidates[arm] = dict(directions=directions, qualified=all(d['qualified'] for d in directions.values()))
    def choose(arms):
        available = [a for a in arms if candidates[a]['qualified']]
        return min(available, key=lambda a: (-np.mean([d['source_delta'] for d in candidates[a]['directions'].values()]), a)) if available else None
    result = dict(status='source_only_frozen', primary_seed=20260915, replication_seed=314159,
        division_nominee=choose(['D10_adapted', 'D20_temporal']), identity_nominee=choose(['A10', 'O10_swap']),
        candidates=candidates, tie_break='Largest mean source full-clip delta, then lexical arm name',
        source_screen_lock_sha256=sha(RESULTS/'source_screen_lock.json'), new_target_scores_read=False,
        independent_generalization_claim=False, replacement_nomination=False,
        composition_order=['identity_or_observation', 'event'],
        clean_end_to_end_transfer='unestablished: inherited exposed upstream checkpoints and teachers')
    write_json(RESULTS/'nomination.json', result, immutable=True)
    return result


if __name__ == '__main__':
    import argparse
    from .resources import cpu_budget
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--identity-source', choices=['44b6','6bba'])
    args = parser.parse_args()
    cpu_budget()
    if args.identity_source:
        identity_loss(args.identity_source, 'A10')
    else:
        print(nominate())

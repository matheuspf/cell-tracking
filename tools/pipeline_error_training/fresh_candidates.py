"""Full renamed-image proofs for the source-frozen primary nominees."""
import sys

from .common import RESULTS, read_json, write_json


def run():
    freeze=read_json(RESULTS/'target_freeze.json')
    nomination=read_json(RESULTS/'nomination.json')
    from .fresh_validation import run as fresh
    records=[]
    arms = [nomination['division_nominee'], nomination['identity_nominee']]
    if freeze['composition']['enabled']:
        arms.append('C10')
    for arm in arms:
        if arm:
            try:
                records.append(fresh(arm,candidate=True))
            except Exception as error:
                # A failed component proof must not cancel an independent one.
                failure = dict(arm=arm,status='failed',error_type=type(error).__name__,
                    reason=str(error),new_target_threshold_or_checkpoint_selection=False)
                write_json(RESULTS/f'fresh_candidate_{arm}_failure.json',failure)
                records.append(failure)
    result=dict(status='measured' if records and all(r['status']=='measured' for r in records) else ('failed' if records else 'not run'),
        reason=None if records else 'No source-qualified nominee; full control fresh proof is reported separately',
        experiments=[r['arm'] for r in records],jobs=records,new_target_threshold_or_checkpoint_selection=False,
        composition_required=freeze['composition']['enabled'])
    write_json(RESULTS/'fresh_candidate_validation.json',result)
    return result


if __name__=='__main__':
    from .resources import cpu_budget
    cpu_budget()
    raise SystemExit(1 if run()['status']=='failed' else 0)

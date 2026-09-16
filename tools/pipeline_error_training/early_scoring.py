"""Score complete frozen inventories while independent point inference runs."""
import fcntl
import time

import psutil

from .common import RESULTS, WORK, inputs, now, read_json, sha, write_json
from .finish_queue import execute, verify_existing_freeze
from .resources import cpu_budget


def inventory_ready(arm, rows, work=WORK):
    for row in rows:
        name = row['dataset']
        root = work/'final_inference'/name if arm.startswith('D') else work/'final_point_inference'/arm/name
        if not (root/'complete.json').is_file():
            return False
        prediction = work/'predictions'/arm/name
        if not all(prediction.with_suffix(s).is_file() for s in ['.npz','.json','.csv','.guard.json']):
            return False
    return bool(rows)


def run():
    cpu_budget()
    verify_existing_freeze()
    plan = read_json(RESULTS/'early_scoring_schedule.json')
    frozen = read_json(RESULTS/'target_freeze.json')
    pending = [a for a in plan['eligible_experiments'] if a in frozen['experiments']]
    rows, jobs = inputs(), []
    root = WORK/'early_scoring'
    root.mkdir(parents=True,exist_ok=True)
    with (root/'runner.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while pending:
            active = read_json(WORK/'finish_queue/active.json')
            try:
                alive = psutil.Process(active['pid']).cmdline()==active['command']
            except psutil.Error:
                alive = False
            if not alive or active['job'] not in ['final_predictions','point_predictions']:
                break
            ready = next((a for a in pending if inventory_ready(a,rows)),None)
            write_json(root/'progress.json',dict(status='running',jobs=jobs,pending=pending,ready=ready))
            if ready is None:
                time.sleep(15)
                continue
            # The scorer's global lock also covers the final queue and exports.
            jobs.append(execute('score-'+ready,'accounting',[ready],update_active=False))
            pending.remove(ready)
            write_json(RESULTS/'early_scoring_execution.json',dict(status='measured',jobs=jobs,
                pending=pending,schedule_sha256=sha(RESULTS/'early_scoring_schedule.json'),
                target_freeze_sha256=sha(RESULTS/'target_freeze.json'),updated=now(),
                target_results_used_for_model_or_threshold_selection=False))
        write_json(root/'progress.json',dict(status='complete',jobs=jobs,pending=pending,
            reason='Any remaining audits are handled by the final queue',completed=now()))


if __name__=='__main__':
    run()

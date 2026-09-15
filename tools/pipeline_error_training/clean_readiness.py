"""Read-only current-process and completed-checkpoint check for the optional lane."""
from .common import REPO, RESULTS, now, read_json, sha, write_json
from .resources import process_snapshot


def run():
    root = REPO/'work/incumbent-comparison-20260914'
    plan = REPO/'results/incumbent-comparison-20260914/native-training-plan.json'
    expected_plan = sha(plan)
    processes = [p for p in process_snapshot()
                 if any('incumbent_comparison.train_native' in a for a in p['command'])]
    fits = []
    for source in ['44b6', '6bba']:
        progress_path = root/f'native-{source}-progress.json'
        progress = read_json(progress_path)
        folder = root/'models/native'/f'source-{source}-seed-314159'
        final = folder/'final.json'
        completed = final.exists() and (folder/'final.pt').exists()
        checkpoint_hash = None
        if completed:
            receipt = read_json(final)
            checkpoint_hash = sha(folder/'final.pt')
            if checkpoint_hash != receipt['sha256'] or receipt['plan_sha256'] != expected_plan:
                raise ValueError('Completed optional native checkpoint failed provenance verification')
        fits.append(dict(source=source, target=progress['target'],
            status='measured' if completed else 'blocked', final_checkpoint_available=completed,
            checkpoint_sha256=checkpoint_hash, progress_sha256=sha(progress_path),
            last_recorded_epoch=progress['epoch'], required_epochs=progress['target_epochs'],
            last_recorded_updates=progress['updates'], last_progress_utc=progress['updated_utc'],
            final_checkpoint=str(folder/'final.pt'),
            reason=None if completed else 'Required fixed final 400-epoch checkpoint is absent'))
    available = all(r['final_checkpoint_available'] for r in fits)
    result = dict(status='measured' if available else 'blocked', checked_utc=now(),
        both_completed_directional_checkpoints_available=available, fits=fits,
        running_native_training_pids=[p['pid'] for p in processes],
        running_job_claim_based_on_current_processes=True, original_plan_sha256=expected_plan,
        another_study_modified_or_resumed=False, clean_comparison_score=None,
        reason='Completed assets require the declared clean-bank comparison before a transfer claim' if available
            else 'Both fixed final source-only upstream checkpoints are required; intermediate snapshots are excluded')
    write_json(RESULTS/'clean_upstream_readiness.json', result)
    return result


if __name__ == '__main__':
    print(run())

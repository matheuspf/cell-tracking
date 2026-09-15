"""Preserve invalid attempts and record protocol repairs before target revelation."""
from pathlib import Path

from .common import RESULTS, WORK, now, read_json, sha, write_json


def first_repair():
    import psutil
    for process in psutil.process_iter(['pid', 'cmdline']):
        args = process.info['cmdline'] or []
        if 'pipeline_error_training' in args and 'train' in args:
            raise RuntimeError('Wait for this study fit to exit before preserving its files')
    root = WORK / 'invalid/attempt1'
    root.mkdir(parents=True, exist_ok=True)
    moved = []
    for name in ['source', 'training', 'source_feasibility', 'training_pilot']:
        source, target = WORK / name, root / name
        if source.exists():
            if target.exists():
                raise FileExistsError(target)
            source.rename(target)
            moved.append(name)
    # Existing scientific input hashes, unchanged crops, baseline scores and D00
    # trace are valid. Preserve logs with their original filenames as well.
    receipt = dict(created=now(), status='invalid_artifacts_preserved', moved=moved,
        reason='Complete-decision labels did not penalize removed supported GT edges; augmentation recipe omitted specified blur and crop-sampling perturbations.',
        target_model_scores_read=False, baseline_target_scores_already_published=True,
        unchanged=['candidate union', 'model families', 'decoder', '178 total updates', 'seeds', 'source/target directions'],
        files={str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file()})
    write_json(root / 'preservation.json', receipt, immutable=True)
    from .augment import CONFIG
    lock = read_json(RESULTS / 'execution_lock.json')
    lock.update(augmentations=CONFIG, supersedes_sha256=sha(RESULTS / 'execution_lock.json'),
                repair=receipt['reason'], invalid_attempt_manifest_sha256=sha(root / 'preservation.json'),
                no_target_driven_selection=True)
    write_json(RESULTS / 'execution_repair_lock.json', lock, immutable=True)
    write_json(RESULTS / 'implementation_repairs.json', dict(attempts=[dict(
        status='failed', artifact_manifest_sha256=sha(root / 'preservation.json'),
        reason=receipt['reason'], source_only=True, stopped_only_this_study_process=True)]), immutable=True)

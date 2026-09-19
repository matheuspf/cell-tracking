"""Preserve historical evidence without importing a numerical/model package."""
import json
import subprocess
from pathlib import Path
from .common import REPO, OLD, WORK, RESULTS, BRANCH, Blocked, now, sha, read, write


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], text=True).strip()


def active_jobs():
    result = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            argv = (proc/'cmdline').read_bytes().decode().split('\0')
            # Avoid recording unrelated personal process arguments or this scanner.
            matches = [x for x in argv if any(y in x for y in
                       ('clean_validation_v10', 'division_reliability_v11', 'incumbent_comparison.train_native'))]
            if matches and not any('\n' in x for x in matches):
                result.append(dict(pid=int(proc.name), modules=matches,
                                   start_ticks=(proc/'stat').read_text().split()[21]))
        except (OSError, UnicodeError, IndexError):
            continue
    return result


def run():
    if git('branch', '--show-current') != BRANCH:
        raise Blocked('Execution is authorized only on the ready branch')
    if (RESULTS/'reconciliation.json').exists():
        return read(RESULTS/'reconciliation.json')
    snapshot = WORK/'preservation'
    snapshot.mkdir(parents=True, exist_ok=True)
    # Porcelain records worktree line-ending changes that diff may normalize away.
    status=subprocess.check_output(['git','-C',str(REPO),'status','--porcelain=v1'],text=True)
    dirty = [line[3:] for line in status.splitlines() if not line.startswith('??')]
    user_files = {}
    for name in dirty:
        if name.startswith(('tools/division_reliability_v11/', 'results/division-reliability-v11/')):
            continue
        path = REPO/name
        if path.is_file():
            target = snapshot/'user-files'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(path.read_bytes())
            user_files[name] = sha(target)
    artifacts = []
    for path in sorted(OLD.rglob('*')):
        if path.is_file() and any(part in path.parts for part in ('fits', 'interruptions', 'engineering')):
            if path.suffix in ('.pt', '.json', '.jsonl', '.py'):
                artifacts.append(dict(path=str(path.relative_to(OLD)), bytes=path.stat().st_size, sha256=sha(path)))
    write(snapshot/'v10-artifacts.json', artifacts, immutable=True)
    cells = []
    for p in sorted(OLD.glob('fits/clean/*/*/upstream/recipe.json')):
        recipe = read(p)
        folder = p.parent
        progress = read(folder/'progress.json')
        missing_code = [name for name in recipe['code_sha256']
                        if not (REPO/'tools/clean_validation_v10'/name).exists()]
        cells.append(dict(source=recipe['source'], seed=recipe['seed'],
                          planned_updates=recipe['planned_updates'], progress=progress,
                          checkpoint='final.pt' if (folder/'final.pt').exists() else 'resume.pt',
                          checkpoint_path=str(folder.relative_to(REPO)),
                          checkpoint_sha256=sha(folder/('final.pt' if (folder/'final.pt').exists() else 'resume.pt')),
                          recipe_sha256=sha(p), parent_hashes=recipe['lineage']['parent_hashes'],
                          missing_code=missing_code, reuse_qualified=False,
                          reason='Matching trainer source and recursive parent manifest not present; receipts alone do not prove eligibility.'))
    leases = [read(p) for p in OLD.glob('resources/*.json')]
    record = dict(created_utc=now(), branch=BRANCH, head=git('rev-parse', 'HEAD'),
                  remote_planning_base='08061f0a224387594dd3e915462bafb5d5928fb2',
                  worktrees=git('worktree', 'list', '--porcelain').replace(str(REPO), '<repo>'),
                  local_v10_code_commits=[], v10_cells=cells, active_jobs=active_jobs(),
                  preservation_manifest=dict(path='work/division-reliability-v11/preservation/v10-artifacts.json',
                                             sha256=sha(snapshot/'v10-artifacts.json'), files=len(artifacts)),
                  preserved_user_files=user_files, upstream_reused_cells=[],
                  inherited_resource_receipt_count=len(leases),
                  inherited_compute_scope='Historical receipt accounting retained separately; no new v11 lease charged yet.',
                  prior_target_exposure='Both embryos used in historical research. v10 interruption receipts state target_scores_opened=false; absent code/lock limits independent verification.',
                  v11_target_scores_opened=False, P0_modified=False)
    write(RESULTS/'reconciliation.json', record)
    return record

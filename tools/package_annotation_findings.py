"""Export verified aggregate findings from the sealed local study; never mutate it."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path('/kaggle/working/cell-tracking/annotation-selection-v1')
DEFAULT_DEST = ROOT / 'results/annotation-selection-v1'
COPIES = (
    'coverage.csv', 'retention.csv', 'public_retention.csv',
    'classifier_metrics.csv', 'calibration.csv', 'feature_profiles.csv',
    'image_seed_results.csv', 'matched_budget_summary.csv',
    'random_exact_seed_results.csv', 'uncertainty.csv', 'matching_summary.csv',
    'environment.json', 'preregistration.json', 'public_selector_preregistration.json',
    'status.json', 'validation_receipt.json',
    *(f'plots/{name}.{ext}' for name in (
        'calibration', 'coverage', 'deleted_group', 'feature_profiles',
        'retention_and_score') for ext in ('png', 'svg')),
)
FORBIDDEN_KEYS = {'candidate_id', 'node_id', 'matched_gt_id', 'source_id',
                  'target_id', 't', 'z', 'y', 'x'}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def assert_aggregate(value):
    if isinstance(value, dict):
        if FORBIDDEN_KEYS & value.keys():
            raise ValueError(f'Nonaggregate fields: {FORBIDDEN_KEYS & value.keys()}')
        for item in value.values():
            assert_aggregate(item)
    elif isinstance(value, list):
        for item in value:
            assert_aggregate(item)


def sanitize_public(value):
    # The original integrity audit includes six individual predicted coordinates.
    # Publish their aggregate count and handling, retaining the audit locally.
    audit = value['spatial_integrity']
    rows = audit.pop('rows')
    audit['individual_coordinate_rows_removed'] = len(rows)
    audit['affected_clips'] = len({r['dataset'] for r in rows})
    audit['detail_location'] = 'LOCAL_ARTIFACTS.md#public_coordinate_audit-json'
    value['graph_parity_checks_passed'] = len(value.pop('graph_parity'))
    assert_aggregate(value)
    return value


def package(source: Path, dest: Path):
    source, dest = source.resolve(), dest.resolve()
    if source == dest or source in dest.parents or dest in source.parents:
        raise ValueError('Publication destination must be separate from sealed artifacts')
    manifest_bytes = (source / 'artifact_manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    inventory = {r['path']: r for r in manifest['files']}
    verified = {}
    outputs = {}

    def read(name):
        blob = (source / name).read_bytes()
        expected = inventory[name]
        if sha(blob) != expected['sha256'] or len(blob) != expected['bytes']:
            raise ValueError(f'Sealed source failed verification: {name}')
        verified[name] = expected
        return blob

    def write(name, blob):
        if isinstance(blob, str):
            blob = blob.encode()
        path = dest / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        outputs[name] = {'path': name, 'bytes': len(blob), 'sha256': sha(blob)}

    def write_json(name, value):
        assert_aggregate(value)
        write(name, json.dumps(value, indent=2, allow_nan=False) + '\n')

    # Read and verify every input before writing any exported artifact.
    inputs = {name: read(name) for name in (*COPIES, 'summary.json',
              'public_summary.json', 'dashboard.html', 'final_report.md')}
    for name in COPIES:
        blob = inputs[name]
        if name.endswith('.csv'):
            for row in csv.DictReader(blob.decode().splitlines()):
                assert_aggregate(row)
        elif name.endswith('.json'):
            assert_aggregate(json.loads(blob))
    summary = json.loads(inputs['summary.json'])
    summary['public'] = sanitize_public(summary['public'])
    public = sanitize_public(json.loads(inputs['public_summary.json']))
    assert summary['public'] == public
    dashboard = inputs['dashboard.html'].decode()
    pattern = r'(<script type="application/json" id="data">)(.*?)(</script>)'
    match = re.search(pattern, dashboard, re.S)
    embedded = json.loads(match[2])
    assert embedded['summary'] == json.loads(inputs['summary.json'])
    embedded['summary'] = summary
    assert_aggregate(embedded)
    dashboard = re.sub(pattern, lambda m: m[1] + json.dumps(
        embedded, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
        + m[3], dashboard, count=1, flags=re.S)

    rows = list(csv.DictReader(inputs['public_retention.csv'].decode().splitlines()))
    eligible = [r for r in rows if r['embryo'] == 'pooled' and
                r['model_id'] not in {'identity', 'oracle', 'random'}]
    best = max(eligible, key=lambda r: float(r['delta']))
    comparison = {
        'interpretation': 'Diagnostic local transfer; checkpoint contamination. '
                          'No hidden-test uplift estimate. Best setting is hindsight only.',
        'public_score_snapshot': 'configs/notebooks.json, 2026-09-08',
        'best_downloaded_public_score': 0.946,
        'fully_evaluated_public_notebook': 'flexonafft/biohub-harmonic-fusion v29',
        'public_notebooks_fully_evaluated': 1,
        'eligible_filter_definition': 'All pooled public rows excluding identity, '
                                      'oracle and random models; no further filtering.',
        'eligible_filter_count': len(eligible),
        'positive_pooled_delta_count': sum(float(r['delta']) > 0 for r in eligible),
        'best_hindsight_deployable_setting': best,
        'best_hindsight_setting_both_embryos': [r for r in rows
            if r['variant_id'] == best['variant_id'] and r['embryo'] != 'pooled'],
        'fixed_primary': public['primary'],
        'focus3d_measured': False,
    }
    assert len(eligible) == 72 and comparison['positive_pooled_delta_count'] == 0

    report = inputs['final_report.md'].decode()
    links = re.findall(r'\]\(([^)]+)\)', report)
    missing = sorted({link for link in links if not link.startswith(('https:', '/'))
                      and link not in COPIES and link not in {
                          'public_summary.json', 'dashboard.html', 'artifact_manifest.json'}})
    def anchor(name):
        return re.sub(r'[^a-z0-9_-]', '-', name.lower())

    def target(link):
        if link == 'artifact_manifest.json':
            return 'bundle_manifest.json'
        if link == 'final_report.md':
            return 'report.md'
        if link.startswith('/home/mpf/code/kaggle/cell-tracking/'):
            return '../../' + link.removeprefix('/home/mpf/code/kaggle/cell-tracking/')
        if link in missing:
            return 'LOCAL_ARTIFACTS.md#' + anchor(link)
        return link

    report = re.sub(r'\]\(([^)]+)\)', lambda m: '](' + target(m[1]) + ')', report)
    report = ('> Portable publication of the completed 2026-09-08 local experiment. '
              'Aggregate numbers are unchanged. Links to large or individual-level '
              'evidence lead to the local artifact inventory. The source report and '
              'manifest hashes are in [bundle_manifest.json](bundle_manifest.json). '
              'See [baseline comparison](README.md) and '
              '[FOCUS-3D findings](../../handover/annotation-selection-v1/FOCUS3D.md).\n\n'
              + report)
    report = report.replace('No Kaggle submission, notebook publication, forum post, push, or hidden-test claim was made.',
                            'The original execution made no Kaggle submission, notebook publication, forum post, push, or hidden-test claim. Findings were subsequently prepared for Git publication at the user\'s request.')
    report = report.replace('All machine-readable artifacts and source/command hashes are indexed in [artifact_manifest.json](bundle_manifest.json)',
                            'The portable exports and their sealed source hashes are indexed in [bundle_manifest.json](bundle_manifest.json); the complete local manifest is described in [LOCAL_ARTIFACTS.md](LOCAL_ARTIFACTS.md)')
    dashboard = re.sub(r'href="([^"]+)"', lambda m: 'href="' + target(m[1]) + '"', dashboard)
    dashboard = dashboard.replace('>artifact_manifest.json<', '>bundle_manifest.json<')
    dashboard = dashboard.replace('<p id="measuredoutcome"></p>',
        '<p id="measuredoutcome"></p><div class="notice"><strong>Public baseline follow-up.</strong> '
        'None of the 72 learned or confidence filter settings improved pooled local score. '
        'The best hindsight setting changes 0.911774 to 0.911222. The downloaded public '
        'leaderboard score is 0.946; these are different evaluation populations. '
        'FOCUS-3D has not been benchmarked. <a href="README.md">Findings and evidence</a> '
        '· <a href="../../handover/annotation-selection-v1/FOCUS3D.md">FOCUS-3D assessment</a>.</div>')
    for name in COPIES:
        blob = inputs[name]
        if name.endswith('.svg'):
            blob = ('\n'.join(line.rstrip() for line in blob.decode().splitlines()) + '\n').encode()
        write(name, blob)
    write_json('full_summary.json', summary)
    write_json('public_summary.json', public)
    write_json('baseline_comparison.json', comparison)
    write('report.md', report)
    write('dashboard.html', dashboard)

    inventory_text = (
        '# Local evidence inventory\n\n'
        'The complete sealed artifact store remains at '
        f'`{source}` on the original machine. It is **not in Git**. '
        f'Its manifest lists {len(manifest["files"]):,} files and '
        f'{manifest["total_bytes"]:,} bytes ({manifest["total_bytes"] / 2**30:.2f} GiB). '
        'Raw patches, image crops, checkpoints, per-node matching and predictions '
        'are excluded from this public repository. Aggregate sweep tables, plots, '
        'the report and the self-contained dashboard are included. No raw data '
        'or original notebooks were changed during publication.\n\n'
        f'Source `artifact_manifest.json` SHA-256: `{sha(manifest_bytes)}`. '
        'This identifies the complete original inventory; `bundle_manifest.json` '
        'identifies the portable export. Historical local receipts describe execution '
        'before the later findings push.\n\n'
        'The following report links refer to local-only evidence. Paths are relative '
        'to the sealed store above. Hashes are copied from its manifest; these '
        'excluded files were not reread during packaging.\n\n')
    for name in sorted(missing):
        receipt = inventory.get(name)
        inventory_text += f'<a id="{anchor(name)}"></a>\n## {name}\n\n'
        inventory_text += f'Local path: `{source / name}`.\n\n'
        if receipt:
            inventory_text += f'{receipt["bytes"]:,} bytes; SHA-256 `{receipt["sha256"]}`.\n\n'
        else:
            inventory_text += 'Consult the complete local manifest for this evidence.\n\n'
    write('LOCAL_ARTIFACTS.md', inventory_text.rstrip() + '\n')
    write_json('bundle_manifest.json', {
        'schema_version': 1, 'study_id': 'annotation-selection-v1',
        'source_artifact_manifest_sha256': sha(manifest_bytes),
        'source_execution_commit': manifest['local_commit'],
        'source_artifact_count': len(manifest['files']),
        'source_total_bytes': manifest['total_bytes'],
        'source_inputs_verified': list(verified.values()),
        'transformations': ['Individual public coordinate rows and per-clip graph parity records removed from JSON.',
                            'Report/dashboard links made portable; baseline follow-up notice added.',
                            'Generated SVG line-ending whitespace normalized; plotted values unchanged.',
                            'No scientific values changed; comparison recomputed from exported aggregate sweep.'],
        'exported_files': list(outputs.values()),
        'manifest_scope': 'Generated exports only; README, concise summary and validation receipt are tracked separately in Git.',
    })
    print(f'Published {len(outputs)} files; verified {len(verified)} sealed source inputs.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--dest', type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()
    package(args.source, args.dest)

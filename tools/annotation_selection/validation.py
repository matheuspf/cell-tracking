"""Record executable checks and a final source-hash receipt for artifact sealing."""
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

from .common import OUT,OFFICIAL,REPO,WORK,now,read_json,sha,write_json


def command(argv,log,env):
    with (OUT/'commands.jsonl').open('a') as f:
        f.write(json.dumps(dict(time=now(),argv=argv,cwd=str(REPO),producer='validation'))+'\n')
    with log.open('w') as f:
        subprocess.run(argv,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)


def counts(path):
    suites=list(ET.parse(path).getroot().iter('testsuite'))
    totals={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
    if totals['failures'] or totals['errors']:raise ValueError('Test failures in JUnit receipt')
    return totals['tests']-totals['skipped'],totals['skipped']


def run(args):
    env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(REPO/'tools'),
             POLARS_MAX_THREADS='2',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    helper=WORK/'validation-helpers.log'
    command([sys.executable,'-m','unittest','discover','-s','handover/annotation-selection-v1','-p','test_*.py','-v'],helper,env)
    helper_count=int(re.search(r'Ran (\d+) tests',helper.read_text()).group(1))
    study_xml=WORK/'validation-study.xml';study_log=WORK/'validation-study.log'
    command([sys.executable,'-m','pytest','-q','tests','--junitxml='+str(study_xml)],study_log,env)
    study,study_skips=counts(study_xml)
    if study_skips:raise ValueError('The study environment skipped required integration checks')
    inspection=shutil.which('python')
    if not inspection:raise ValueError('Activate the existing cell-tracking inspection environment for browser validation')
    inspection_xml=WORK/'validation-inspection.xml';inspection_log=WORK/'validation-inspection.log'
    command([inspection,'-m','pytest','-q','tests','--junitxml='+str(inspection_xml)],inspection_log,env)
    inspection_count,inspection_skips=counts(inspection_xml)
    upstream_logs=[WORK/'upstream-tests.log',WORK/'upstream-sandbox-tests.log']
    env_record=read_json(OUT/'environment.json')
    current_source=all(sha(OFFICIAL/p)==h for p,h in env_record['metric_files'].items())
    cached=current_source and all(p.exists() and re.search(r'\d+ passed',p.read_text()) for p in upstream_logs)
    if cached:
        upstream=sum(int(re.findall(r'(\d+) passed',p.read_text())[-1]) for p in upstream_logs)
    else:
        upstream_xml=WORK/'validation-upstream.xml';upstream_log=WORK/'validation-upstream.log'
        official_env={**env,'PYTHONPATH':str(OFFICIAL/'src')}
        command([sys.executable,'-m','pytest','-q',str(OFFICIAL/'tests/test_metrics.py'),
                 str(OFFICIAL/'tests/test_division_metrics.py'),str(OFFICIAL/'tests/test_division_sandbox_examples.py'),
                 '--junitxml='+str(upstream_xml)],upstream_log,official_env)
        upstream,skipped=counts(upstream_xml)
        if skipped:raise ValueError('Upstream fixtures were skipped')
        upstream_logs=[upstream_log]
    browser=WORK/'validation-browser.log'
    command([inspection,'-m','annotation_selection','browser-validate'],browser,env)
    paths=[*sorted((REPO/'tools/annotation_selection').glob('*.py')),*sorted((REPO/'tests/annotation_selection').glob('*.py'))]
    logs=[helper,study_log,inspection_log,browser,*upstream_logs]
    write_json(OUT/'validation_receipt.json',dict(created=now(),passed=True,helper_tests=helper_count,
               study_tests=study,inspection_tests=inspection_count,inspection_skips=inspection_skips,upstream_tests=upstream,
               upstream_pass_receipts_reused=cached,unchanged_upstream_source=current_source,
               study_python=sys.executable,inspection_python=inspection,
               code_hashes={str(p.relative_to(REPO)):sha(p) for p in paths},
               logs={str(p.relative_to(WORK)):sha(p) for p in logs},dashboard_sha256=sha(OUT/'dashboard.html')))
    print(f'Validated: {helper_count} helper, {study} study, {inspection_count} inspection, {upstream} upstream tests; offline browser checks passed.',flush=True)

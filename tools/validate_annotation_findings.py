"""Check the portable findings from a checkout, without competition data."""
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from package_annotation_findings import DEFAULT_DEST, ROOT, assert_aggregate, sha


def validate(dest: Path, browser_check: bool):
    dest = dest.resolve()
    manifest = json.loads((dest / 'bundle_manifest.json').read_text())
    for item in manifest['exported_files']:
        blob = (dest / item['path']).read_bytes()
        assert len(blob) == item['bytes'] and sha(blob) == item['sha256'], item['path']
    counts = {}
    for path in dest.glob('*.csv'):
        rows = list(csv.DictReader(path.open()))
        assert_aggregate(rows)
        counts[path.name] = len(rows)
    assert counts['retention.csv'] == 2385 and counts['public_retention.csv'] == 975
    for path in dest.glob('*.json'):
        assert_aggregate(json.loads(path.read_text()))
    html = (dest / 'dashboard.html').read_text()
    payload = json.loads(re.search(
        r'<script type="application/json" id="data">(.*?)</script>', html, re.S)[1])
    assert_aggregate(payload)
    assert payload['summary'] == json.loads((dest / 'full_summary.json').read_text())
    links = [(dest / 'dashboard.html', link) for link in re.findall(r'href="([^"]+)"', html)]
    for path in list(dest.glob('*.md')) + [
            ROOT / 'handover/annotation-selection-v1/NEXT_AGENT.md',
            ROOT / 'handover/annotation-selection-v1/FOCUS3D.md']:
        links.extend((path, link) for link in re.findall(r'\]\(([^)]+)\)', path.read_text()))
    for origin, link in links:
        if link.startswith(('https://', 'http://')):
            continue
        path, _, anchor = link.partition('#')
        assert not path.startswith('/'), (origin, link)
        target = origin.parent / path if path else origin
        if browser_check and target.resolve() == dest / 'publication_validation.json':
            continue  # This receipt is written only after the browser checks pass.
        assert target.is_file(), (origin, link)
        if anchor:
            assert f'id="{anchor}"' in target.read_text(), (origin, link)
    result = dict(created=datetime.now(timezone.utc).isoformat(), passed=True,
                  exported_hashes_verified=len(manifest['exported_files']),
                  csv_rows=counts, relative_links_checked=len(links),
                  individual_coordinate_fields_absent=True,
                  html_sha256=sha((dest / 'dashboard.html').read_bytes()))
    if browser_check:
        from playwright.sync_api import sync_playwright
        errors, requests, checks = [], [], []
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(offline=True, viewport={'width': 1440, 'height': 1050})
            page = context.new_page()
            page.on('pageerror', lambda e: errors.append(str(e)))
            page.on('request', lambda r: requests.append(r.url)
                    if r.url.startswith(('http:', 'https:')) else None)
            page.goto((dest / 'dashboard.html').as_uri(), wait_until='load')
            assert page.locator('#cards .card').count() == 4
            assert page.evaluate('D.summary.public.sample_count') == 199
            for lane in ('clean', 'public'):
                page.select_option('#lane', lane)
                for embryo in ('44b6', '6bba', 'pooled'):
                    page.select_option('#embryo', embryo)
                    for model in ('hgb_all_leaf7', 'image_only_seed20260908'):
                        page.select_option('#model', model)
                        for policy in ('nodes', 'tracklets', 'tracklets_fork_protected'):
                            page.select_option('#policy', policy)
                            assert page.locator('#points tbody tr').count() == 7
                            assert page.locator('svg').count() == 8
                            assert page.locator('#score circle').count() == 21
                            checks.append(dict(lane=lane, embryo=embryo, model=model, policy=policy))
            page.select_option('#focus', 'high')
            assert page.locator('#score circle').count() == 12
            with page.expect_download() as download:
                page.click('#download')
            plotted = json.loads(Path(download.value.path()).read_text())
            assert plotted['lane'] == 'public' and len(plotted['series']) == 3
            assert all(len(s['rows']) == 7 for s in plotted['series'])
            page.select_option('#lane', 'clean')
            page.select_option('#embryo', 'pooled')
            page.select_option('#model', 'hgb_all_leaf7')
            page.select_option('#policy', 'tracklets')
            page.select_option('#focus', 'all')
            screenshot_dir = ROOT / 'work/annotation-selection-v1/publication'
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(screenshot_dir / 'dashboard.png'))
            page.set_viewport_size({'width': 390, 'height': 844})
            assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+2')
            page.screenshot(path=str(screenshot_dir / 'dashboard-mobile.png'))
            context.close()
            browser.close()
        assert not errors and not requests, (errors, requests)
        result.update(offline=True, browser_checks=checks,
                      mobile_no_horizontal_overflow=True, downloaded_embedded_plot_data=True,
                      javascript_errors=errors, network_requests=requests)
        (dest / 'publication_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'browser_checks'}, indent=2))
    if browser_check:
        print(f'Passed {len(checks)} offline browser combinations.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dest', type=Path, default=DEFAULT_DEST)
    parser.add_argument('--browser', action='store_true', help='Also exercise Chromium and write the publication receipt')
    args = parser.parse_args()
    validate(args.dest, args.browser)

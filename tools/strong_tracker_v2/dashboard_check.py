"""Offline browser validation; run with the documented inspection interpreter."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    out = Path('/kaggle/working/cell-tracking/strong-tracker-v2')
    path = out / 'dashboard.html'
    expected = json.loads((out / 'summary.json').read_text())
    errors, network, states = [], [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
        page = browser.new_page(viewport={'width': 1280, 'height': 900}, device_scale_factor=1)
        page.on('pageerror', lambda exc: errors.append(str(exc)))
        page.on('console', lambda msg: errors.append(msg.text) if msg.type == 'error' else None)

        def route_request(route):
            if route.request.url.startswith(('https:', 'http:')):
                network.append(route.request.url)
                route.abort()
            else:
                route.continue_()

        page.route('**/*', route_request)
        page.goto(path.as_uri())
        page.wait_for_function('window.dashboardReady === true')
        assert page.locator('#headline').inner_text() == f"{expected['selected']['pooled']['score']:.6f}"
        families = page.locator('#family option').all_text_contents()
        for embryo in ['pooled', '44b6', '6bba']:
            page.select_option('#embryo', embryo)
            for family in families:
                page.select_option('#family', label=family)
                count = page.locator('#results tbody tr').count()
                correct = page.evaluate('''([e,f]) => D.points.filter(r =>
                    r.embryo === e && (f === 'All' || r.family === f)).length''', [embryo, family])
                assert count == correct and count > 0
                assert page.locator('#scores rect').count() == min(count, 12)
                assert page.locator('#oracles tbody tr').count() == 4
                for element in page.locator('svg').all():
                    svg = element.inner_html()
                    assert 'NaN' not in svg and 'Infinity' not in svg, (embryo, family, svg[:200])
                states.append({'embryo': embryo, 'family': family, 'rows': count})
        for source in ['44b6', '6bba']:
            page.select_option('#train-source', source)
            assert page.locator('#learning circle').count() == 150
        page.select_option('#train-source', '44b6')
        page.select_option('#embryo', 'pooled')
        page.select_option('#family', 'All')
        assert page.locator('#results tbody tr').count() == 104
        assert page.locator('#results tbody tr.selected').count() == 1
        with page.expect_download() as download:
            page.click('#download')
        exported = json.loads(Path(download.value.path()).read_text())
        assert len(exported['points']) == 312
        assert exported['summary']['selected_variant'] == expected['selected_variant']
        assert len(exported['curves']) == 300
        screenshots = []
        for name, width, height in [('desktop', 1280, 900), ('mobile', 390, 844)]:
            page.set_viewport_size({'width': width, 'height': height})
            assert page.evaluate('document.documentElement.scrollWidth') <= width
            screenshot = out / f'dashboard_{name}.png'
            page.screenshot(path=str(screenshot), full_page=True)
            screenshots.append({'name': screenshot.name, 'width': width, 'height': height,
                'sha256': hashlib.sha256(screenshot.read_bytes()).hexdigest()})
        version = browser.version
        browser.close()
    assert not errors, errors
    assert not network, network
    receipt = dict(created=datetime.now(timezone.utc).isoformat(), passed=True,
        dashboard_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), chromium_version=version,
        states=states, screenshots=screenshots, page_errors=errors, network_requests=network,
        export_aggregate_rows=312, training_curve_points=300, horizontal_overflow=False)
    (out / 'dashboard_validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'Dashboard passed: {len(states)} population/family states, both sources, JSON export, desktop/mobile, zero network requests')


if __name__ == '__main__':
    main()

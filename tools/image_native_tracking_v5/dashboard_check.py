"""Offline Chromium verification of measured dashboard interactions and exports."""
import csv,sys
from playwright.sync_api import sync_playwright
from .common import *

def run():
    path=OUT/'dashboard.html';data=read(OUT/'dashboard_data.json');errors=[];network=[];states=[]
    if (OUT/'HOCT_feature_audit.json').exists():
        assert len(data.get('feature_coverage',[]))==3,'Completed feature audit missing from dashboard payload'
        assert len(data.get('feature_distribution',[]))==57
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport=dict(width=1360,height=960))
        page.on('pageerror',lambda e:errors.append(str(e)))
        def route(r):
            if r.request.url.startswith(('http:','https:')):network.append(r.request.url);r.abort()
            else:r.continue_()
        page.route('**/*',route);page.goto(path.as_uri())
        assert not errors,errors
        page.wait_for_selector('#scoreTable tbody tr')
        if not data.get('fresh') and data.get('fresh_progress'):
            count=data['fresh_progress'].get('image_clips_executed',0)
            assert page.locator('#freshCaption').inner_text().startswith(f'{count} of 6 complete image executions')
        for em in ['pooled','44b6','6bba']:
            page.select_option('#embryo',em)
            expected=sum(r['embryo']==em and not r['variant'].startswith('Oracle_') for r in data['scores'])
            assert page.locator('#scoreTable tbody tr').count()==expected
            assert page.locator('#scoreChart rect').count()==expected
            states.append(dict(embryo=em,rows=expected))
        for row in data['training']:
            page.select_option('#trainingRun',row['key'])
            assert row['key'] in page.locator('#trainingCaption').inner_text()
            assert len(page.locator('#lossChart polyline').get_attribute('points'))>5
        if data.get('feature_coverage'):
            assert page.locator('#featureCoverage tbody tr').count()==3
            page.locator('#featureSection summary').click()
            for em in ['pooled','44b6','6bba']:
                page.select_option('#featurePopulation',em)
                assert page.locator('#featureDistribution tbody tr').count()==19
            page.select_option('#featurePopulation','pooled')
            page.locator('#featureSection summary').click()
        if data.get('HOCT_unit_audit'):
            assert page.locator('#featureUnits').is_visible()
            assert 'upstream-default-voxel comparison was not run' in page.locator('#featureUnits').inner_text()
            assert page.locator('#unitSensitivity tbody tr').count()==2
        with page.expect_download() as info:page.locator('#download').click()
        dest=OUT/'dashboard_download.csv';info.value.save_as(dest)
        with dest.open() as f:downloaded=list(csv.DictReader(f))
        assert len(downloaded)==len(data['scores'])
        screenshots=[]
        for label,width,height in [('desktop',1360,960),('mobile',390,844)]:
            page.set_viewport_size(dict(width=width,height=height))
            assert page.evaluate('document.documentElement.scrollWidth')<=width
            screenshot=OUT/f'dashboard_{label}.png';page.screenshot(path=str(screenshot),full_page=True)
            screenshots.append(dict(file=screenshot.name,sha256=sha(screenshot),width=width,height=height))
        version=browser.version;browser.close()
    assert not errors and not network,(errors,network)
    write(OUT/'dashboard_validation.json',dict(passed=True,at=now(),states=states,errors=errors,network_requests=network,
        screenshots=screenshots,browser_version=version,python=sys.executable,dashboard_sha256=sha(path),downloaded_rows=len(downloaded),
        HOCT_feature_filter_populations=3 if data.get('feature_coverage') else 0,
        HOCT_unit_limitation_visible=bool(data.get('HOCT_unit_audit'))))
    print('Offline dashboard browser checks passed',flush=True)

if __name__=='__main__':run()

"""Offline Chromium checks following the existing v5 reporting checks."""
import csv
from playwright.sync_api import sync_playwright
from .common import *

def run():
    path=REPO/'results/segmentation-tracking-v6/dashboard.html';errors=[];network=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport=dict(width=1360,height=960))
        page.on('pageerror',lambda error:errors.append(str(error)))
        def route(r):
            if r.request.url.startswith(('http:','https:')):network.append(r.request.url);r.abort()
            else:r.continue_()
        page.route('**/*',route);page.goto(path.as_uri());page.wait_for_function('window.dashboardReady===true')
        for em in ['pooled','44b6','6bba']:
            page.select_option('#embryo',em)
            assert page.locator('#scores tbody tr').count()==6
            assert page.locator('#scores tbody tr.blocked').count()==4
            assert '0.9348649864' in page.locator('#scores').inner_text() if em=='pooled' else True
        with page.expect_download() as download:page.locator('#download').click()
        output=SCRATCH/'dashboard';output.mkdir(parents=True,exist_ok=True)
        csvpath=output/'download.csv';download.value.save_as(csvpath)
        with csvpath.open() as f:rows=list(csv.DictReader(f))
        assert len(rows)==18 and sum(r['status']=='blocked' and r['score']=='' for r in rows)==12
        for label,width,height in [('desktop',1360,960),('mobile',390,844)]:
            page.set_viewport_size(dict(width=width,height=height))
            assert page.evaluate('document.documentElement.scrollWidth')<=width
            page.screenshot(path=str(output/f'{label}.png'),full_page=True)
        version=browser.version;browser.close()
    assert not errors and not network
    record=dict(passed=True,browser_version=version,desktop_and_mobile=True,download_rows=18,
                blank_blocked_score_rows=12,external_requests=network,page_errors=errors,
                dashboard_sha256=sha(path))
    write(OUT/'dashboard_check.json',record)
    print('Offline dashboard checks passed',flush=True)

if __name__=='__main__':run()

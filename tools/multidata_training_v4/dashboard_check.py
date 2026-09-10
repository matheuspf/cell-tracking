"""Offline browser verification of the measured, responsive dashboard."""
import csv
import re
import sys
from playwright.sync_api import sync_playwright
from .common import *

def run():
    path=OUT/'dashboard.html';summary=read(OUT/'study_summary.json');errors=[];network=[];states=[]
    variants=summary['completed_graph_variants']
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport=dict(width=1280,height=900))
        page.on('pageerror',lambda e:errors.append(str(e)))
        def route(r):
            if r.request.url.startswith(('http:','https:')):network.append(r.request.url);r.abort()
            else:r.continue_()
        page.route('**/*',route);page.goto(path.as_uri());page.wait_for_selector('#scores tbody tr')
        page.select_option('#scope','all')
        assert f'{summary["selected"]["score"]:.12f}' in page.locator('#summary').inner_text()
        for embryo in ['pooled','44b6','6bba']:
            page.select_option('#embryo',embryo)
            for order in ['score','variant']:
                page.select_option('#order',order)
                assert page.locator('#scores tbody tr').count()==variants
                assert page.locator('#plot rect').count()==variants
                states.append(dict(embryo=embryo,order=order,rows=variants))
        page.select_option('#scope','training')
        arms=page.locator('#scores tbody tr td:first-child').all_text_contents()
        assert page.locator('#plot rect').count()==sum(bool(re.fullmatch(r'C(?:0|1short|[1-7])(?:_seed2)?',v)) for v in arms)
        for name in page.locator('#model option').all_text_contents()[::5]:
            page.select_option('#model',name)
            assert name in page.locator('#curve').text_content()
            assert len(page.locator('#curve polyline').get_attribute('points'))>50
        page.select_option('#curvemetric','heldout');page.select_option('#model','G_C4_44b6')
        assert 'synthetic' in page.locator('#curve').text_content()
        assert len(page.locator('#curve polyline').get_attribute('points'))>50
        page.select_option('#model','G_real_44b6')
        assert 'No recorded' in page.locator('#curve').text_content()
        page.select_option('#curvemetric','loss')
        if page.locator('#stresssection').count():
            for case in page.locator('#stresscase option').all_text_contents():
                page.select_option('#stresscase',case)
                for component,count in [('G',6),('I',5)]:
                    page.select_option('#stresscomponent',component)
                    assert page.locator('#stress tbody tr').count()==count
        if page.locator('#renderedsection').count():assert page.locator('#rendered tbody tr').count()==8
        for name in page.locator('#calmodel option').all_text_contents()[::4]:
            page.select_option('#calmodel',name)
            assert name in page.locator('#reliability').text_content()
            assert page.locator('#reliability circle').count()>0
        with page.expect_download() as info:page.locator('#download').click()
        download=info.value;dest=OUT/'dashboard_download.csv';download.save_as(dest)
        with dest.open() as f:downloaded=list(csv.DictReader(f))
        assert len(downloaded)==variants*3
        page.select_option('#embryo','pooled');page.select_option('#order','score')
        screenshots=[]
        for label,width,height in [('desktop',1280,900),('mobile',390,844)]:
            page.set_viewport_size(dict(width=width,height=height))
            assert page.evaluate('document.documentElement.scrollWidth')<=width
            screenshot=OUT/f'dashboard_{label}.png';page.screenshot(path=str(screenshot),full_page=True)
            screenshots.append(dict(file=screenshot.name,sha256=sha(screenshot),width=width,height=height))
        version=browser.version;browser.close()
    assert not errors,errors
    assert not network,network
    write(OUT/'dashboard_validation.json',dict(passed=True,created=now(),states=states,page_errors=errors,network_requests=network,
        screenshots=screenshots,chromium_version=version,browser_python=sys.executable,
        dashboard_sha256=sha(path),download_rows=len(downloaded),horizontal_overflow=False))
    print('offline dashboard verified',variants,'variants',flush=True)

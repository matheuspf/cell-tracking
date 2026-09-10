"""Offline browser assertions for the actual measured dashboard."""
import json
import os
from pathlib import Path
from playwright.sync_api import sync_playwright
from .context import RunContext
from .common import now,sha,write_json

def main():
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:32])
    ctx=RunContext.default();path=ctx.out/'dashboard.html';summary=json.loads((ctx.out/'summary.json').read_text())
    errors=[];network=[];states=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport=dict(width=1280,height=900))
        page.on('pageerror',lambda e:errors.append(str(e)))
        def route(r):
            if r.request.url.startswith(('http:','https:')):network.append(r.request.url);r.abort()
            else:r.continue_()
        page.route('**/*',route);page.goto(path.as_uri());page.wait_for_selector('#rows tr')
        assert page.locator('#score').inner_text()==f'{summary["selected"]["score"]:.12f}'
        for embryo in ['pooled','44b6','6bba']:
            page.select_option('#embryo',embryo)
            assert page.locator('#rows tr').count()==summary['variants']
            assert page.locator('#rows tr.selected').count()==1
            states.append(dict(embryo=embryo,rows=page.locator('#rows tr').count()))
        page.fill('#filter','incumbent');assert page.locator('#rows tr').count()==1
        page.fill('#filter','');page.select_option('#embryo','pooled')
        page.locator('details').last.locator('summary').click()
        page.select_option('#sampleVariant',summary['selected_variant'])
        assert page.locator('#samples tr').count()==199
        page.fill('#sampleFilter','44b6');assert page.locator('#samples tr').count()==71
        page.fill('#sampleFilter','');page.locator('details').last.locator('summary').click()
        shots=[]
        for label,width,height in [('desktop',1280,900),('mobile',390,844)]:
            page.set_viewport_size(dict(width=width,height=height))
            assert page.evaluate('document.documentElement.scrollWidth')<=width
            screenshot=ctx.out/f'dashboard_{label}.png';page.screenshot(path=str(screenshot),full_page=True)
            shots.append(dict(file=screenshot.name,sha256=sha(screenshot),width=width,height=height))
        version=browser.version;browser.close()
    assert not errors,errors
    assert not network,network
    receipt=dict(created=now(),passed=True,states=states,page_errors=errors,network_requests=network,
        screenshots=shots,chromium_version=version,dashboard_sha256=sha(path),horizontal_overflow=False)
    write_json(ctx.out/'dashboard_validation.json',receipt);print(receipt)

if __name__=='__main__':main()

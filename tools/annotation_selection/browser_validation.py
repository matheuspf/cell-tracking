"""Exercise the delivered HTML artifacts in a real browser with networking off."""
import json

from .common import OUT,now,sha,write_json


def run(args):
    from playwright.sync_api import sync_playwright
    failures=[];requests=[];checks=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True)
        context=browser.new_context(offline=True,viewport={'width':1440,'height':1050})
        page=context.new_page()
        page.on('pageerror',lambda e:failures.append(str(e)))
        page.on('request',lambda r:requests.append(r.url) if r.url.startswith(('http:','https:')) else None)
        page.goto((OUT/'dashboard.html').as_uri(),wait_until='load')
        assert page.locator('#cards .card').count()==4
        assert page.evaluate('D.summary.public.sample_count')==199
        for lane in ['clean','public']:
            page.select_option('#lane',lane)
            for embryo in ['44b6','6bba','pooled']:
                page.select_option('#embryo',embryo)
                for model in ['hgb_all_leaf7','image_only_seed20260908']:
                    page.select_option('#model',model)
                    for policy in ['nodes','tracklets','tracklets_fork_protected']:
                        page.select_option('#policy',policy)
                        assert page.locator('#points tbody tr').count()==7
                        assert page.locator('svg').count()==8
                        assert page.locator('#score circle').count()==21
                        checks.append(dict(lane=lane,embryo=embryo,model=model,policy=policy,points=7))
        page.select_option('#focus','high')
        assert page.locator('#score circle').count()==12
        with page.expect_download() as download:
            page.click('#download')
        payload=json.loads(open(download.value.path()).read())
        assert payload['lane']=='public' and len(payload['series'])==3
        assert all(len(s['rows'])==7 for s in payload['series'])
        page.select_option('#lane','clean');page.select_option('#model','hgb_all_leaf7')
        page.select_option('#policy','tracklets');page.select_option('#focus','all')
        page.screenshot(path=str(OUT/'dashboard_validation.png'),full_page=False)
        page.screenshot(path=str(OUT/'dashboard_full_page.png'),full_page=True)
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+2')
        page.screenshot(path=str(OUT/'dashboard_mobile.png'),full_page=False)
        audits=[]
        page.set_viewport_size({'width':1100,'height':920})
        for folder in ['blinded_census','blinded_candidate_audit']:
            path=OUT/folder/'viewer.html'
            page.goto(path.as_uri(),wait_until='load')
            assert page.locator('#sample option').count()==48
            page.click('#next');assert page.locator('#id').inner_text().endswith('001')
            page.locator('#z').fill('0');page.locator('#z').dispatch_event('input')
            page.wait_for_timeout(200)
            assert page.evaluate('Array.from(document.getElementById("image").getContext("2d").getImageData(0,0,32,32).data).some((v,i)=>i%4!==3&&v>0)')
            assert page.locator('#answer').input_value()==''
            page.screenshot(path=str(OUT/folder/'viewer_validation.png'),full_page=True)
            audits.append(dict(viewer=folder,sha256=sha(path),options=48,canvas_nonblank=True,assessment_fields_blank=True))
        context.close();browser.close()
    if failures or requests:raise ValueError(dict(javascript_errors=failures,network_requests=requests))
    write_json(OUT/'audit_browser_validation.json',dict(created=now(),passed=True,viewers=audits,network_requests=requests,errors=failures))
    write_json(OUT/'dashboard_validation.json',dict(created=now(),passed=True,html_sha256=sha(OUT/'dashboard.html'),
               offline=True,checks=checks,mobile_no_horizontal_overflow=True,downloaded_embedded_plot_data=True,
               network_requests=requests,javascript_errors=failures,screenshot='dashboard_validation.png'))
    print(f'Offline browser validation passed: {len(checks)} selector combinations, mobile view, JSON export, both audit viewers.',flush=True)

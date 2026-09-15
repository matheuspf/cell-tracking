"""Verify report arithmetic, every division scene and real browser interactions."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

from .detection_server import connect, clip_info, snapshot
from .pipeline import DEFAULT_OUTPUT, REPO, sha256, write_json
from .tracking_server import connect_tracking, get_case, list_cases, report


def verify_data(root):
    db=connect_tracking(root); data=report(db); source=connect(root)
    checked=0
    for model, result in data['models'].items():
        s=result['summaries']['all']; counts=s['counts']; clips=result['clips']
        assert len(clips)==199
        assert sum(s['categories'].values())==list_cases(db,dict(model=model))['total']
        assert s['categories']['edge_endpoint']+s['categories']['edge_link']==counts['edge_fn']
        denominator=counts['edge_tp']+counts['edge_fp']+counts['edge_fn']
        adjusted=sum(max(0,r['edge_tp']*(1.1-.1*r['num_pred_nodes']/r['estimated_total'])) for r in clips)/denominator
        score=adjusted+.1*counts['division_tp']/(counts['division_tp']+counts['division_fp']+counts['division_fn'])
        assert abs(score-s['score'])<1e-12
        for category,count in s['categories'].items():
            last=list_cases(db,dict(model=model,filter=category,offset='999999'))
            assert last['total']==count and last['offset']+len(last['rows'])==count
        # Every division scene plus up to three examples per edge evidence group.
        cases=db.execute("SELECT key FROM cases WHERE model=? AND category LIKE 'division_%' ORDER BY dataset,t,key",(model,)).fetchall()
        for stage in ['selection','proposal_gap','assignment','association','competing','unmatched','wrong_identity']:
            cases += db.execute('SELECT key FROM cases WHERE model=? AND stage=? ORDER BY dataset,t,key LIMIT 3',(model,stage)).fetchall()
        for (key,) in cases:
            payload=get_case(root,db,key); case=payload['case'];scene=payload['scene']
            clip,mi=clip_info(source,case['dataset'],model)
            common=snapshot(str(root/'detection-review/snapshots'/clip['snapshot']))
            final=snapshot(str(root/'detection-review/snapshots'/mi['snapshot']))
            lookup={**{f'g:{n[0]}':n for n in common['gt']},**{f'p:{n[0]}':n for n in final['nodes']}}
            points={p['key']:p for p in scene['points']}
            for point in points.values():
                row=lookup[point['key']]
                assert point['t']==row[1] and point['zyx']==row[2:].tolist()
                assert point['t'] in scene['times']
            for kind,edges,prefix in [('expected',common['gt_edges'],'g'),('actual',final['edges'],'p')]:
                allowed={(f'{prefix}:{a}',f'{prefix}:{b}') for a,b in edges}
                assert all((e['source'],e['target']) in allowed for e in scene[kind])
                assert all(e['source'] in points and e['target'] in points for e in scene[kind])
            if case['category']=='division_fn':
                assert f"g:{case['node']}" in points
                assert payload['matching']=='Official division-window matching'
            if case['category']=='division_fp':
                assert sum(e['source']==f"p:{case['node']}" for e in scene['actual'])>=2
            checked+=1
    db.close();source.close()
    return dict(passed=True,scenes_checked=checked,all_division_scenes=True,official_error_partition=True,
                independent_score_arithmetic=True,last_page_all_model_categories=True)


def verify_browser(url,root):
    output=root/'tracking-review/checks';output.mkdir(parents=True,exist_ok=True)
    errors,failed,external=[],[],[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True)
        page=browser.new_page(viewport=dict(width=1600,height=1050))
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('response',lambda r:failed.append([r.status,r.url]) if r.status>=400 else None)
        page.on('request',lambda r:external.append(r.url) if not r.url.startswith(url) and not r.url.startswith('blob:') else None)

        def ready():
            page.wait_for_function('''() => {const t=window.pipelineReview?.state;return t?.payload && !t.busy &&
              t.payload.case.model===t.model && t.volumes.size===t.payload.scene.times.length &&
              [...document.querySelectorAll('#t-images canvas')].every(c=>c.dataset.case===t.key); }''',timeout=60000)
            assert page.locator('#t-error').is_hidden(),page.locator('#t-error').inner_text()

        page.goto(url+'#tab=report')
        page.wait_for_selector('.tr-score-strip')
        for model in ['pooled','best']:
            page.locator('#r-model').select_option(model)
            for embryo in ['all','44b6','6bba']:
                page.locator('#r-embryo').select_option(embryo)
                expected=page.evaluate('(x)=>window.pipelineReview.state.report.models[x[0]].summaries[x[1]].score',[model,embryo])
                assert page.locator('.tr-score-strip strong').first.inner_text()==f'{expected:.6f}'
        page.locator('#r-model').select_option('pooled');page.locator('#r-embryo').select_option('all')
        page.screenshot(path=str(output/'report-desktop.png'),full_page=True)
        page.locator('[data-review="division_fn"][data-stage="division_no_fork"]').first.click();ready()
        assert page.locator('#t-total').inner_text()=='97 error flags'
        assert page.evaluate('window.pipelineReview.state.payload.scene.times.length')<=5
        page.screenshot(path=str(output/'division-desktop.png'),full_page=True)

        # Every image is a real native frame. Match grayscale hashes independently.
        import zarr
        state=page.evaluate('''() => {const t=window.pipelineReview.state;return {dataset:t.payload.case.dataset,times:t.payload.scene.times};}''')
        source=connect(root);clip,_=clip_info(source,state['dataset'],'pooled');source.close()
        array=zarr.open_group(clip['image_root'],mode='r')['0']
        for t in state['times']:
            lo,hi=clip['display_limits'];raw=np.asarray(array[t]);gray=np.clip((raw.astype(np.float32)-lo)*255/max(hi-lo,1),0,255).astype(np.uint8)
            digest=page.evaluate('''async(t)=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',window.pipelineReview.state.volumes.get(t)))).map(x=>x.toString(16).padStart(2,'0')).join('')''',t)
            assert digest==hashlib.sha256(gray.tobytes()).hexdigest()
        for plane in ['xy','xz','yz']:
            page.locator('#t-plane').select_option(plane)
            assert page.locator('#t-images canvas').first.get_attribute('data-plane')==plane
            assert page.evaluate('''() => {const t=window.pipelineReview.state,g=t.payload.scene.times.map(x=>window.pipelineReview.frameGeometry(x));return g.every(x=>x.field===g[0].field && JSON.stringify(x.center)===JSON.stringify(g[0].center));}''')
        page.locator('#t-plane').select_option('xy')
        page.locator('#t-next').click();ready();second=page.evaluate('window.pipelineReview.state.key')
        page.reload();ready();assert page.evaluate('window.pipelineReview.state.key')==second
        page.locator('#t-notes').fill('Automated verification note');page.locator('#t-save').click()
        with page.expect_download() as download:page.locator('#t-export').click()
        assert download.value.suggested_filename=='tracking-review-notes.json'
        page.reload();ready();assert page.locator('#t-notes').input_value()=='Automated verification note'
        # Detection links must open the exact annotation, including its full-clip match.
        href=page.locator('#t-centers a').first.get_attribute('href')
        other=browser.new_page();other.goto(url.rstrip('/')+href)
        other.wait_for_function('window.detectionReview?.state.payload && !window.detectionReview.state.caseLoading',timeout=60000)
        from urllib.parse import parse_qs,urlsplit
        assert other.evaluate('window.detectionReview.state.key')==parse_qs(urlsplit(href).fragment)['dcase'][0]
        other.close()
        page.locator('#t-reset').click();ready()
        for model in ['pooled','best']:
            page.locator('#t-model').select_option(model);ready()
            for category in ['edge_endpoint','edge_link','edge_fp','division_fn','division_fp']:
                page.locator('#t-filter').select_option(category);ready()
                expected=page.evaluate('(x)=>window.pipelineReview.state.report.models[x[0]].summaries.all.categories[x[1]]',[model,category])
                assert page.evaluate('window.pipelineReview.state.total')==expected
                page.locator('#t-jump').fill(str(expected));page.locator('#t-go').click();ready()
                page.wait_for_function('(n)=>{const t=window.pipelineReview.state;return t.offset+t.rows.findIndex(x=>x.key===t.key)+1===n;}',arg=expected)
                assert page.locator('#t-next').is_disabled()
        page.locator('#t-stage').select_option('association')
        page.wait_for_selector('#t-empty',state='visible')
        assert page.locator('#t-case').is_hidden()
        page.locator('#t-reset').click();ready()
        for width in [390,768,1600]:
            page.set_viewport_size(dict(width=width,height=1000))
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
            page.screenshot(path=str(output/f'tracking-{width}.png'),full_page=True)
            page.locator('#tab-report').click();page.wait_for_selector('.tr-score-strip')
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
            page.screenshot(path=str(output/f'report-{width}.png'),full_page=True)
            page.locator('#tab-tracking').click();ready()
        # Existing tabs continue to work after visiting both additions.
        page.locator('#tab-detection').click()
        page.wait_for_function('window.detectionReview?.state.frame && !window.detectionReview.state.caseLoading',timeout=60000)
        assert page.locator('#tracking-panel').is_hidden()
        page.locator('#tab-comparison').click()
        page.wait_for_function('window.centerComparison?.state.frame',timeout=60000)
        assert page.locator('#comparison-panel').is_visible()
        assert not errors,errors
        assert not failed,failed
        assert not external,external
        browser.close()
    return dict(passed=True,report_scope_combinations=6,tracking_model_categories=10,
                all_last_cases_reachable=True,native_frame_hashes=True,own_frame_centers=True,
                shared_physical_crop=True,planes=['xy','xz','yz'],deep_links=True,detection_links=True,
                notes_export_restore=True,empty_filter_recovery=True,viewport_widths=[390,768,1600],
                existing_tabs=True,errors=errors,failed_requests=failed,external_requests=external)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--url',default='http://localhost:8767/')
    args=parser.parse_args()
    result=dict(data=verify_data(args.output))
    print('TRACKING_DATA_VERIFIED',result['data'],flush=True)
    result['browser']=verify_browser(args.url,args.output)
    result['source_sha256']={p.name:sha256(p) for p in Path(__file__).parent.glob('tracking_*') if p.is_file()}
    write_json(REPO/'results/pipeline-errors-20260915/validation.json',result)
    print('TRACKING_BROWSER_VERIFIED',result['browser'],flush=True)


if __name__=='__main__':
    main()

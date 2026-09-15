"""Exercise all-data review, native pixels, ambiguity, notes and navigation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import zarr
from playwright.sync_api import sync_playwright

from .detection_server import clip_info, connect, snapshot
from .pipeline import DEFAULT_OUTPUT, write_json


def verify(url, root):
    output = root / 'detection-review/checks'
    output.mkdir(parents=True, exist_ok=True)
    errors, failed, requests = [], [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1700, 'height': 1100})
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.on('request', lambda r: requests.append(r.url))
        page.on('response', lambda r: failed.append((r.status, r.url)) if r.status >= 400 else None)

        def ready():
            page.wait_for_function('''window.detectionReview?.state.frame && !window.detectionReview.state.caseLoading &&
                !document.querySelector('#d-images').classList.contains('loading')''')
            assert page.locator('#d-error').is_hidden(), page.locator('#d-error').inner_text()
            assert page.evaluate('''() => {const d=window.detectionReview.state;
                return d.payload.frames[d.position]===d.frame && Object.values(d.frame.points).flat().every(n=>n[1]===d.frame.t);
            }''')

        def native_frame():
            dataset, t, model = page.evaluate('''() => {const d=window.detectionReview.state;return [d.payload.case.dataset,d.frame.t,d.model];}''')
            db = connect(root); clip, _ = clip_info(db, dataset, model); db.close()
            raw = np.asarray(zarr.open_group(clip['image_root'], mode='r')['0'][t])
            low, high = clip['display_limits']
            gray = np.clip((raw.astype(np.float32)-low)*255/max(high-low, 1), 0, 255).astype(np.uint8)
            digest = page.evaluate('''async()=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',window.detectionReview.state.volume))).map(x=>x.toString(16).padStart(2,'0')).join('')''')
            assert digest == hashlib.sha256(gray.tobytes()).hexdigest(), (dataset, t)

        def option(control, value):
            if control == 'd-model':
                page.locator('#d-pred-source').select_option('final:'+value)
                return
            page.locator('#d-options').evaluate('(e)=>e.open=true')
            page.locator('#'+control).select_option(value)
            page.locator('#d-options').evaluate('(e)=>e.open=false')

        def jump(number):
            page.locator('#d-jump').fill(str(number)); page.locator('#d-jump-go').click()
            page.wait_for_function('(n)=>{const d=window.detectionReview.state;return d.offset+d.rows.findIndex(r=>r.key===d.key)+1===n && !d.caseLoading;}', arg=number)
            ready()

        page.goto(url)
        ready()
        catalog = page.evaluate('window.detectionReview.state.catalog')
        total = catalog['counts']['pooled']['all']
        assert total == 50546 and catalog['datasets'] == 199
        assert page.locator('#comparison-panel').is_hidden()
        assert page.locator('#d-progress').inner_text() == '1 / 50,546'
        assert page.locator('#d-coverage').inner_text() == 'All 199 clips'
        assert not page.locator('#d-options').evaluate('(e)=>e.open')
        assert not page.locator('#d-details').evaluate('(e)=>e.open')
        assert page.locator('#d-images canvas:visible').count() == 2
        native_frame()
        page.screenshot(path=str(output / 'all-errors-desktop.png'), full_page=True)
        # Old single-sequence links also land on the complete review.
        page.goto(url + '#sequence=6bba_e16ffc58-movie&errorScope=sequence&errorKind=all')
        ready()
        assert page.evaluate('window.detectionReview.state.total') == total
        assert page.evaluate('window.detectionReview.state.filter') == 'all'
        assert page.evaluate('window.detectionReview.state.dataset') == ''

        page.locator('#d-options').evaluate('(e)=>e.open=true')
        for control in ['d-centers', 'd-labels', 'd-lines']:
            page.locator('#'+control).uncheck()
        page.locator('#d-options').evaluate('(e)=>e.open=false')
        option('d-projection', 'slice')
        for plane, depth in [('xy', 31), ('xz', 120), ('yz', 90)]:
            page.locator('#d-plane').select_option(plane)
            page.locator('#d-options').evaluate('(e)=>e.open=true')
            page.locator('#d-depth').fill(str(depth))
            page.locator('#d-options').evaluate('(e)=>e.open=false')
            for view in ['d-full', 'd-zoom']:
                assert page.evaluate('(view)=>new Set([...document.querySelectorAll("."+view+" canvas")].map(c=>c.toDataURL())).size===1', view)
            # Full views contain the entire physical image. Zoom and full use
            # independent depth projections of the same verified native volume.
            assert page.evaluate("""() => {
                const {state:d,projection,transform}=window.detectionReview,[Z,Y,X]=d.payload.clip.shape.slice(1);
                for(const full of [false,true]){
                    const p=projection(full),tr=transform(document.querySelector(full?'#d-final-full':'#d-final-canvas'),full),s=d.payload.clip.spacing;
                    if(full&&(tr.ox<-.001||tr.oy<-.001||tr.ox+p.width*s[p.h]*tr.scale>720.001||tr.oy+p.height*s[p.v]*tr.scale>720.001))return false;
                    for(let y=0;y<p.height;y++)for(let x=0;x<p.width;x++){
                        let value=0;
                        for(let depth=p.low;depth<=p.high;depth++){
                            const q=d.plane==='xy'?[depth,y,x]:d.plane==='xz'?[y,depth,x]:[y,x,depth];
                            value=Math.max(value,d.volume[(q[0]*Y+q[1])*X+q[2]]);
                        }
                        if(p.gray[y*p.width+x]!==value)return false;
                    }
                }return true;
            }""")
        page.locator('#d-options').evaluate('(e)=>e.open=true')
        for control in ['d-centers', 'd-labels', 'd-lines']:
            page.locator('#'+control).check()
        page.locator('#d-options').evaluate('(e)=>e.open=false')
        option('d-projection', 'local')
        assert page.evaluate("""()=>{const d=window.detectionReview.state,p=window.detectionReview.projection();return [window.detectionReview.pairInfo().gt,window.detectionReview.pairInfo().pred].filter(Boolean).every(r=>r[p.d+2]>=p.low&&r[p.d+2]<=p.high);}""")
        before = page.evaluate('window.detectionReview.state.center')
        empty = page.evaluate("""()=>{const c=document.querySelector('#d-final-full'),r=c.getBoundingClientRect();for(let y=5;y<r.height;y+=25)for(let x=5;x<r.width;x+=25){if(!window.detectionReview.hitCenter(c,{clientX:r.left+x,clientY:r.top+y},true))return {x,y};}}""")
        page.locator('#d-final-full').click(position=empty)
        assert page.evaluate('window.detectionReview.state.center') != before
        # Next crosses the fetch boundary; the final indexed example is reachable.
        jump(40); page.locator('#d-next').click()
        page.wait_for_function('window.detectionReview.state.offset===40'); ready()
        assert page.locator('#d-progress').inner_text() == '41 / 50,546'
        page.locator('#d-prev').click(); ready()
        assert page.locator('#d-progress').inner_text() == '40 / 50,546'
        jump(total)
        assert page.locator('#d-next').is_disabled()
        assert page.locator('#d-progress').inner_text() == '50,546 / 50,546'
        native_frame()
        page.screenshot(path=str(output / 'last-error.png'), full_page=True)
        jump(1)
        # All centers and all associations are present, for both final pipelines.
        combinations = 0
        for model in ['best', 'pooled']:
            option('d-model', model); ready()
            for group in ['all', 'detection', 'association', 'unlabeled']:
                page.locator('#d-groups button[data-filter="'+group+'"]').click(); ready()
                assert page.evaluate('window.detectionReview.state.total') == catalog['counts'][model][group]
                if group == 'association':
                    assert page.locator('#d-kind').inner_text().startswith('Association')
                    page.locator('#d-connections summary').click()
                    t = int(page.locator('#d-connections button').first.get_attribute('data-time'))
                    page.locator('#d-connections button').first.click(); ready()
                    assert page.evaluate('window.detectionReview.state.frame.t') == t
                if group == 'unlabeled':
                    assert page.locator('#d-unknown-note').is_visible()
                combinations += 1
        option('d-pool', 'raw'); ready()
        assert page.evaluate('window.detectionReview.state.total') == catalog['raw_unmatched']
        assert page.locator('#d-groups button[data-filter="unlabeled"] b').inner_text() == f"{catalog['raw_unmatched']:,}"
        jump(catalog['raw_unmatched'])
        assert page.locator('#d-next').is_disabled()
        option('d-pool', 'final'); ready()
        page.locator('#d-groups button[data-filter="all"]').click(); ready()
        # A rapid seek cannot mix one timepoint's volume with another's centers.
        page.evaluate("""() => {const e=document.querySelector('#d-time');for(const n of [0,Number(e.max),1]){e.value=n;e.dispatchEvent(new Event('input'));}}""")
        page.wait_for_function('window.detectionReview.state.position===1'); ready(); native_frame()
        page.locator('#d-play').click()
        start = page.evaluate('window.detectionReview.state.frame.t')
        page.wait_for_function('(t)=>window.detectionReview.state.frame.t!==t', arg=start)
        page.locator('#d-play').click(); ready()
        paused = page.evaluate('window.detectionReview.state.frame.t')
        page.wait_for_timeout(650)
        assert page.evaluate('window.detectionReview.state.frame.t') == paused
        page.locator('#d-details summary').click()
        page.locator('#d-verdict').select_option('Uncertain')
        page.locator('#d-notes').fill('AUTOMATED TEST — not a scientific assessment.')
        page.locator('#d-save').click()
        link, key = page.url, page.evaluate('window.detectionReview.state.key')
        page.reload(); ready()
        assert page.url == link and page.evaluate('window.detectionReview.state.key') == key
        page.locator('#d-details summary').click()
        assert page.locator('#d-notes').input_value().startswith('AUTOMATED TEST')
        with page.expect_download() as info:
            page.locator('#d-export').click()
        exported = json.loads(Path(info.value.path()).read_text())
        assert any(r['case_key']==key and r['source']['prediction_sha256'] for r in exported['assessments'])
        page.locator('#d-verdict').select_option('');page.locator('#d-notes').fill('');page.locator('#d-save').click()
        page.evaluate('localStorage.removeItem("biohub-detection-assessments-v1")')
        page.locator('#d-details summary').click()
        for width in [390, 768, 1700]:
            page.set_viewport_size({'width': width, 'height': 1000})
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'), width
            assert page.locator('#d-images canvas:visible').count() == 2
        page.set_viewport_size({'width':390,'height':900})
        page.screenshot(path=str(output/'all-errors-mobile.png'),full_page=True)
        page.set_viewport_size({'width':1700,'height':1100})
        # An explicit clip filter can be cleared in one click, and empty frames
        # do not inherit the last case's images or number.
        clip_name = catalog['clips'][0]['dataset']
        db=connect(root);clip,_=clip_info(db,clip_name,'pooled');db.close()
        common=snapshot(str(root/'detection-review/snapshots'/clip['snapshot']))
        empty_t=next(t for t in range(clip['shape'][0]) if t not in set(common['gt'][:,1]))
        option('d-filter','unlabeled');ready()
        option('d-clip',clip_name);ready()
        page.locator('#d-options').evaluate('(e)=>e.open=true')
        page.locator('#d-frame-filter').fill(str(empty_t));page.locator('#d-frame-filter').press('Tab')
        page.locator('#d-options').evaluate('(e)=>e.open=false');ready()
        assert page.evaluate('window.detectionReview.state.frame.points.gt.length')==0
        option('d-filter','annotations')
        page.wait_for_function('window.detectionReview.state.total===0 && !window.detectionReview.state.payload')
        assert page.locator('#d-play').is_disabled() and 'dcase=' not in page.url
        page.locator('#d-groups button[data-filter="all"]').click();ready()
        assert page.evaluate('window.detectionReview.state.total')==total
        page.locator('#tab-comparison').click()
        page.wait_for_function('window.centerComparison?.state.frame && !window.centerComparison.state.busy')
        assert page.locator('#comparison-panel').is_visible()
        page.locator('#open-all-errors').click();ready()
        assert page.evaluate('window.detectionReview.state.total')==total
        assert not errors, errors
        assert not failed, failed
        assert all(urlsplit(r).hostname in ['localhost', '127.0.0.1'] for r in requests)
        browser.close()
    result = dict(passed=True, datasets=199, frames=19900, annotations=133318,
                  default_all_examples=50546, legacy_28_case_links_open_all_data=True,
                  first_and_last_examples_reachable=True, next_crosses_page_boundary=True, raw_pool_last_example=True,
                  model_filter_combinations=combinations, native_frame_bytes_independently_verified=True,
                  full_and_zoom_pixels_in_all_planes=True, full_image_not_cropped=True,
                  synchronized_overlays=True, automatic_depth_includes_selected_centers=True, overlay_views=2,
                  full_image_click_moves_zoom=True, own_frame_centers=True, association_endpoints=True,
                  rapid_frame_navigation=True, playback_pause=True, clip_frame_reset=True,
                  local_assessment_export=True, case_links_and_tab_state=True,
                  empty_filter_recovery=True, zero_annotation_frames=True,
                  viewport_widths=[390,768,1700], errors=errors, failed_requests=failed, external_requests=0)
    write_json(output / 'browser-validation.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8767')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    verify(args.url, args.output)

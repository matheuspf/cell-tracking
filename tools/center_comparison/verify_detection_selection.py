"""Check the rendered radius, source colors, physical distances and real picks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright

from .detection_server import connect
from .pipeline import DEFAULT_OUTPUT, write_json


def verify(url, root):
    output = root / 'detection-review/checks'
    output.mkdir(parents=True, exist_ok=True)
    db = connect(root)
    exact = db.execute("SELECT key FROM cases WHERE model='pooled' AND entity='gt' AND distance=0 LIMIT 1").fetchone()['key']
    conflict = db.execute("SELECT key FROM cases WHERE model='pooled' AND kind='conflict' LIMIT 1").fetchone()['key']
    db.close()
    errors, failed = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1700, 'height': 1100})
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.on('response', lambda r: failed.append((r.status, r.url)) if r.status >= 400 else None)

        def ready():
            page.wait_for_function('''window.detectionReview?.state.frame && !window.detectionReview.state.caseLoading &&
                !document.querySelector('#d-images').classList.contains('loading')''')
            assert page.locator('#d-error').is_hidden()

        def state():
            return page.evaluate('''()=>{const d=window.detectionReview.state;
                return {key:d.key,time:d.frame.t,gt:d.targetGt,pred:d.selectedPred,method:d.method,model:d.model};}''')

        def pair():
            info = page.evaluate('window.detectionReview.pairInfo()')
            spacing = page.evaluate('window.detectionReview.state.payload.clip.spacing')
            if info['gt'] and info['pred']:
                expected = sum(((a-b)*s)**2 for a, b, s in zip(info['gt'][2:], info['pred'][2:], spacing))**.5
                assert abs(expected-info['distance']) < 1e-10
                assert info['inside'] == (expected <= 7)
                assert f'{expected:.2f} µm' in page.locator('#d-selection-readout').inner_text()
            assert page.locator('#d-pred-choice').input_value() == str(info['pred'][0])
            assert page.locator('#d-nearby-picks button[aria-pressed=true]').get_attribute('data-pred') == str(info['pred'][0])
            return info

        def colors():
            # Inspect actual pixels around the selected centers, including when
            # their coordinates coincide; a single highlight color is insufficient.
            assert page.evaluate('''()=>{
                const {pairInfo,transform,coords}=window.detectionReview,info=pairInfo();
                for(const full of [false,true]){
                    const c=document.querySelector(full?'#d-final-full':'#d-final-canvas'),tr=transform(c,full);
                    const pixels=c.getContext('2d').getImageData(0,0,c.width,c.height).data;
                    for(const [row,color] of [[info.gt,[255,209,102]],[info.pred,[66,228,245]]]){
                        if(!row)continue;const [cx,cy]=coords(row,tr);let count=0;
                        for(let y=Math.max(0,Math.floor(cy-18));y<Math.min(c.height,cy+18);y++)
                            for(let x=Math.max(0,Math.floor(cx-18));x<Math.min(c.width,cx+18);x++){
                                const i=4*(y*c.width+x);if(color.every((v,k)=>pixels[i+k]===v))count++;
                            }
                        if(count<5)return false;
                    }
                }return true;
            }''')

        def ring_checks(expected=True):
            # Capture arcs submitted to the real canvas renderer, independently
            # of ringGeometry(). A prediction must never acquire the GT radius.
            draws = page.evaluate('''()=>{
                const proto=CanvasRenderingContext2D.prototype,arc=proto.arc,draws=[];
                proto.arc=function(x,y,r,...rest){if(this.canvas.id.startsWith('d-final')&&r>16)draws.push({id:this.canvas.id,x,y,r});return arc.call(this,x,y,r,...rest);};
                try{document.querySelector('#d-brightness').dispatchEvent(new Event('input'));}finally{proto.arc=arc;}
                const d=window.detectionReview.state,[h,v]=d.plane==='xy'?[2,1]:d.plane==='xz'?[2,0]:[1,0];
                return {draws,shape:d.payload.clip.shape.slice(1),spacing:d.payload.clip.spacing,center:d.center,field:d.field,h,v,gt:window.detectionReview.pairInfo().gt};
            }''')
            assert len(draws['draws']) == (2 if expected else 0), draws
            for ring in draws['draws']:
                full = ring['id'] == 'd-final-full'
                h, v = draws['h'], draws['v']
                spacing, shape, gt = draws['spacing'], draws['shape'], draws['gt']
                field = max(shape[h]*spacing[h], shape[v]*spacing[v]) if full else draws['field']
                center = [(n-1)/2 for n in shape] if full else draws['center']
                assert abs(ring['r'] / 720 * field - 7) < 1e-10
                for axis, coord in [(h, 'x'), (v, 'y')]:
                    assert abs(ring[coord] - (360 + (gt[axis+2]-center[axis])*spacing[axis]*720/field)) < 1e-8

        page.goto(url)
        ready()
        initial = state()
        labels = page.locator('#d-pred-source option').all_text_contents()
        assert labels == ['Final output · C4_m6', 'Final output · P0', 'Detector candidates · before tracking']
        assert 'Raw' not in page.locator('#detection-panel').inner_text()
        assert pair()['assignment'] == 'none' and not pair()['inside']
        original_matches = page.evaluate('JSON.stringify(window.detectionReview.state.frame.matches)')
        page.screenshot(path=str(output / 'clear-overlay-default.png'), full_page=True)
        for plane in ['xy', 'xz', 'yz']:
            page.locator('#d-plane').select_option(plane)
            page.locator('#d-fit-pair').click()
            ring_checks()
            colors()
        page.locator('#d-plane').select_option('xy')
        page.locator('#d-fit-pair').click()
        page.screenshot(path=str(output / 'clear-overlay-desktop.png'), full_page=True)

        # The chip, select control, canvas and displayed distance share one pick.
        second = page.locator('#d-nearby-picks button').nth(1)
        pred = int(second.get_attribute('data-pred'))
        second.click()
        assert state()['pred'] == pred and pred != initial['pred']
        pair()
        for full, selector in [(False, '#d-final-canvas'), (True, '#d-final-full')]:
            point = page.evaluate('''full=>{
                const {state:d,transform,coords,projection,hitCenter}=window.detectionReview;
                const c=document.querySelector(full?'#d-final-full':'#d-final-canvas'),b=c.getBoundingClientRect(),tr=transform(c,full),p=projection(full);
                for(const {row} of d.ranked){
                    if(row[0]===d.selectedPred||row[p.d+2]<p.low||row[p.d+2]>p.high)continue;
                    const [x,y]=coords(row,tr);if(x<20||y<20||x>700||y>700)continue;
                    const pos={x:x/c.width*b.width,y:y/c.height*b.height};
                    const hit=hitCenter(c,{clientX:b.left+pos.x,clientY:b.top+pos.y},full);
                    if(hit?.method==='pred'&&hit.row[0]===row[0])return {...pos,id:row[0]};
                }return null;
            }''', full)
            assert point, selector
            page.locator(selector).click(position={'x': point['x'], 'y': point['y']})
            assert state()['pred'] == point['id']
            pair()
        assert page.evaluate('JSON.stringify(window.detectionReview.state.frame.matches)') == original_matches
        # All predictions, including those outside the current crop/depth, are selectable.
        far = page.evaluate('window.detectionReview.state.ranked.at(-1).row[0]')
        page.locator('#d-pred-choice').select_option(str(far))
        assert state()['pred'] == far
        pair()
        assert page.evaluate('''()=>{const {pairInfo,projection,coords,transform}=window.detectionReview,p=projection(),c=document.querySelector('#d-final-canvas');
            return [pairInfo().gt,pairInfo().pred].every(r=>{const [x,y]=coords(r,transform(c));return x>=0&&x<=720&&y>=0&&y<=720&&r[p.d+2]>=p.low&&r[p.d+2]<=p.high;});}''')

        # Source/model changes preserve the user's GT and time, rather than
        # silently replacing it with the case's original annotation.
        other_gt = page.evaluate('window.detectionReview.state.frame.points.gt.find(r=>r[0]!==window.detectionReview.state.targetGt)[0]')
        page.locator('#d-gt-choice').select_option(str(other_gt))
        for source, method, model in [('raw','raw','pooled'), ('final:pooled','final','pooled'), ('final:best','final','best'), ('raw','raw','best')]:
            page.locator('#d-pred-source').select_option(source)
            ready()
            selected = state()
            assert selected['gt'] == other_gt and selected['time'] == initial['time']
            assert selected['method'] == method and selected['model'] == model
            assert selected['key'].split(':')[1:] == initial['key'].split(':')[1:]
            assert page.locator('#d-pred-choice option').count() == page.evaluate('window.detectionReview.state.frame.points[window.detectionReview.state.method].length')
            pair()
        page.locator('#d-nearby-picks button').nth(2).click()
        selected, link = state(), page.url
        page.reload()
        ready()
        assert state() == selected and page.url == link
        page.locator('#d-forward').click()
        ready()
        assert state()['time'] != selected['time']
        assert page.evaluate('''()=>{const {state:d,pairInfo}=window.detectionReview;return [pairInfo().gt,pairInfo().pred].filter(Boolean).every(r=>r[1]===d.frame.t);}''')

        # These are real indexed examples, not modified data or fake screenshots.
        for key, name in [(exact, 'exact-match'), (conflict, 'assignment-conflict')]:
            page.goto(url + '#' + urlencode(dict(tab='detection',review='2',dfilter='annotations',dcase=key)))
            page.reload()  # Open the saved view, including after a hash-only navigation.
            ready()
            info = pair()
            assert info['inside']
            if name == 'exact-match':
                assert info['distance'] == 0 and info['assignment'] == 'this'
                assert info['gt'][2:] == info['pred'][2:]
                assert 'Matched to this GT' in page.locator('#d-selection-readout').inner_text()
            else:
                assert info['assignment'] == 'other' and info['assignedGT'] != info['gt'][0]
                assert 'Matched to GT' in page.locator('#d-selection-readout').inner_text()
                assert page.locator('.d-assignment.other').is_visible()
                # The actual assigned GT is also in the zoom's image depth.
                assert page.evaluate('''()=>{const {state:d,pairInfo,projection}=window.detectionReview,p=projection(),r=d.frame.points.gt.find(r=>r[0]===pairInfo().assignedGT);return r[p.d+2]>=p.low&&r[p.d+2]<=p.high;}''')
            colors()
            ring_checks()
            page.screenshot(path=str(output / (name + '.png')), full_page=True)

        # Same projected XY does not imply a valid 3D match. Boundary and
        # zero-distance assignment checks use a pure helper, leaving data intact.
        assert page.evaluate('''()=>{
            const compare=window.detectionReview.centerComparison,g=[1,0,10,20,30];
            const z=compare(g,[2,0,15,20,30],[1.625,.40625,.40625]);
            const edge=compare(g,[2,0,17,20,30],[1,1,1]);
            const same=compare(g,[2,0,10,20,30],[1,1,1],1),other=compare(g,[2,0,10,20,30],[1,1,1],3);
            return z.distance===8.125&&!z.inside&&edge.distance===7&&edge.inside&&same.assignment==='this'&&other.assignment==='other';
        }''')
        page.locator('#d-gt-choice').select_option('')
        assert 'No GT selected' in page.locator('#d-selection-readout').inner_text()
        ring_checks(expected=False)
        page.goto(url)
        page.reload()
        ready()
        for width in [390, 768, 1700]:
            page.set_viewport_size({'width': width, 'height': 1000})
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'), width
            assert page.locator('#d-images canvas:visible').count() == 2
            assert page.locator('#d-pred-source').is_visible()
            assert page.locator('#d-pred-choice').is_visible()
            if width == 390:
                page.screenshot(path=str(output/'clear-overlay-mobile.png'),full_page=True)
        assert not errors, errors
        assert not failed, failed
        browser.close()
    result = dict(passed=True,gt_color='#ffd166',prediction_color='#42e4f5',
                  both_colors_in_full_and_zoom_pixels=True,zero_distance_keeps_both_colors=True,
                  real_radius_arcs_7um_on_gt_all_planes=True,no_gt_no_radius=True,
                  independent_3d_distances=True,depth_only_outside_radius=True,
                  nearest_chips_and_all_prediction_dropdown=True,click_prediction_full_and_zoom=True,
                  far_prediction_fits_crop_and_depth=True,selection_does_not_change_assignments=True,
                  sources_preserve_selected_gt_and_frame=True,source_counts_correct=True,
                  source_and_selected_pair_restore_from_link=True,frame_selection_uses_own_time=True,
                  assignment_distinct_from_radius=True,conflicting_assignment_gt_visible=True,
                  real_example_keys=dict(exact=exact,conflict=conflict),viewport_widths=[390,768,1700],
                  errors=errors,failed_requests=failed)
    write_json(output/'selection-validation.json', result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8767')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    verify(args.url, args.output)

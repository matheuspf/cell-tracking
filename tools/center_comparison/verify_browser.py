"""Exercise the local viewer against its real exported volumes in headless Chromium."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright


def verify(url, output):
    output.mkdir(parents=True, exist_ok=True)
    errors, failed, requests = [], [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1600, 'height': 1100})
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda request: requests.append(request.url))
        page.on('response', lambda response: failed.append((response.status, response.url)) if response.status >= 400 else None)
        page.goto(url + '#tab=comparison&sequence=6bba_e16ffc58-movie&t=20')
        page.wait_for_function('window.centerComparison?.state.frame?.t === 20')
        assert page.locator('#sequence option').count() == 8
        assert page.locator('#assignments tr').count() > 0
        assert page.locator('#error').is_hidden()
        assert page.locator('#current').input_value() == 'pooled'
        page.locator('.score-details summary').click()
        assert '0.935178' in page.locator('#best-overall').inner_text()
        assert '199 evaluation clips' in page.locator('#best-overall').inner_text()
        for control in ['gt', 'centers', 'masks', 'lines']:
            page.locator('#' + control).uncheck()
        # All panels must show exactly the same image when overlays are hidden.
        assert page.evaluate('new Set([...document.querySelectorAll("#panels canvas")].map(c=>c.toDataURL())).size === 1')
        page.locator('#projection').select_option('slice')
        for plane, depth in [('xy', 31), ('xz', 120), ('yz', 90)]:
            page.locator('#plane').select_option(plane)
            page.locator('#depth').fill(str(depth))
            assert page.evaluate('new Set([...document.querySelectorAll("#panels canvas")].map(c=>c.toDataURL())).size === 1')
            assert page.evaluate('''() => {
              const {state:s,projection}=window.centerComparison,p=projection(),[Z,Y,X]=s.seq.shape;
              const axes=s.plane==='xy'?[2,1,0]:s.plane==='xz'?[2,0,1]:[1,0,2];
              for(const [x,y] of [[0,0],[13,7],[p.width-1,p.height-1]]) {
                const q=[0,0,0];q[axes[0]]=x;q[axes[1]]=y;q[axes[2]]=s.depth;
                const offset=(q[0]*Y+q[1])*X+q[2],i=y*p.width+x;
                if(p.gray[i]!==s.volume.gray[offset]||p.labels[i]!==s.volume.labels[offset])return false;
              }return true;
            }''')
        page.locator('#plane').select_option('xy')
        page.locator('#projection').select_option('mip')
        assert page.locator('#depth').is_disabled()
        for control in ['gt', 'centers', 'masks', 'lines']:
            page.locator('#' + control).check()
        comparisons = 0
        for current in ['pooled', 'best', 'detector', 'selected', 'watershed']:
            page.locator('#current').select_option(current)
            for focus in ['focus_centroid', 'focus_weighted', 'focus_peak']:
                page.locator('#focus').select_option(focus)
                for radius in ['1', '2', '3', '5', '7']:
                    page.locator('#radius').select_option(radius)
                    expected = page.evaluate('''() => {
                      const s=window.centerComparison.state, n=s.frame.points.gt.length;
                      return [s.current,s.focus].map(method => (100*s.frame.matches[method][s.radius].length/n).toFixed(1)+'%');
                    }''')
                    assert page.locator('#current-recall').inner_text() == expected[0]
                    assert page.locator('#focus-recall').inner_text() == expected[1]
                    comparisons += 1
        page.locator('#current').select_option('detector')
        page.locator('#focus').select_option('focus_centroid')
        page.locator('#assignments button').first.click()
        assert page.locator('#projection').input_value() == 'slab'
        assert page.evaluate('window.centerComparison.state.zoom') == 3
        assert 'Annotated node' in page.locator('#inspector').inner_text()
        assert 'Exact center' in page.locator('#inspector').inner_text()
        share_url = page.url
        selection = page.evaluate('window.centerComparison.state.selection')
        page.reload()
        page.wait_for_function('window.centerComparison?.state.selection != null')
        assert page.url == share_url
        assert page.evaluate('window.centerComparison.state.selection') == selection
        assert page.evaluate('window.centerComparison.state.zoom') == 3
        page.screenshot(path=str(output / 'selected-cell.png'), full_page=True)
        page.locator('#reset').click()
        page.locator('#projection').select_option('mip')
        page.locator('#fps').select_option('8')
        page.locator('#play').click()
        page.wait_for_function('window.centerComparison.state.frame.t >= 22')
        page.locator('#play').click()
        # A previously requested frame may complete once; the timer must stop.
        page.wait_for_function('!window.centerComparison.state.busy')
        paused = page.locator('#time-label').inner_text()
        page.wait_for_timeout(500)
        assert page.locator('#time-label').inner_text() == paused
        page.locator('#time').fill('0')
        page.locator('#time').fill('15')
        page.locator('#time').fill('3')
        page.wait_for_function('window.centerComparison.state.frame.t === 23 && !window.centerComparison.state.busy')
        assert page.locator('#time-label').inner_text() == '23'
        page.locator('#mode').select_option('tracks')
        assert not page.locator('#current').is_disabled()
        assert page.locator('#current').input_value() == 'best'
        assert 'no division model' in page.locator('#mode-note').inner_text()
        assert page.evaluate('window.centerComparison.state.frame.links.selected.length > 0')
        assert page.evaluate('window.centerComparison.state.frame.links.focus_centroid.length > 0')
        page.screenshot(path=str(output / 'temporal-links.png'), full_page=True)
        def ready_case():
            page.wait_for_function('window.centerComparison?.state.activeError != null && window.errorCaseView?.current?.id === window.centerComparison.state.activeError && !window.centerComparison.state.busy')

        # Each timepoint has its own real microscopy; the two stages of
        # center matching and temporal linking are explicitly distinguished.
        for method, score in [('pooled', '0.935178'), ('best', '0.934865')]:
            page.locator('#error-model').select_option(method)
            page.locator('#error-scope').select_option('all')
            page.locator('#error-categories button[data-category="link_missing"]').click()
            ready_case()
            assert page.locator('#current').input_value() == method
            assert page.locator('#radius').input_value() == '7'
            assert page.locator('#mode').input_value() == 'tracks'
            if not page.locator('.score-details').evaluate('(e)=>e.open'):
                page.locator('.score-details summary').click()
            assert score in page.locator('#best-overall').inner_text()
            assert page.locator('#error-detail .case-step.found').count() >= 2
            assert 'missing' in page.locator('#error-detail .case-step.failed').inner_text()
            assert page.locator('#case-images canvas').count() == 2
            assert page.evaluate('new Set(window.errorCaseView.current.tiles.map(t=>t.t)).size === 2')
            assert 'Recovered' in page.locator('#case-connections').inner_text()
            assert 'Missing' in page.locator('#case-connections').inner_text()
            assert 'Correct' in page.locator('#case-connections').inner_text()
            page.locator('#error-kind').select_option('link_wrong')
            ready_case()
            assert 'Incorrect' in page.locator('#case-connections').inner_text()
            active = page.evaluate('window.centerComparison.state.activeError')
            page.locator('#error-next').click()
            ready_case()
            assert page.evaluate('window.centerComparison.state.activeError') != active
            error_url = page.url
            page.reload()
            ready_case()
            assert page.url == error_url
            assert page.locator('#current').input_value() == method
            page.screenshot(path=str(output / f'{method}-error-review.png'), full_page=True)
            page.locator('#error-kind').select_option('link_detection')
            ready_case()
            assert 'Endpoint missing' in page.locator('#error-detail').inner_text()
            page.locator('#error-kind').select_option('center_missing')
            ready_case()
            assert page.locator('#mode').input_value() == 'matches'
            assert 'SAME FRAME' in page.locator('#error-detail').inner_text()
            assert page.evaluate('new Set(window.errorCaseView.current.tiles.map(t=>t.t)).size === 1')
            assert page.evaluate('''() => {
                const [a,b]=window.errorCaseView.current.tiles;
                return a.gray.every((v,i)=>v===b.gray[i]);
            }''')
            assert 'Nearest:' in page.locator('#case-connections').inner_text()
            page.screenshot(path=str(output / f'{method}-missing-center.png'), full_page=True)
            # Rapid case changes must finish with the last requested case's
            # images, even when an earlier request was still loading.
            page.locator('#error-kind').select_option('link_wrong')
            page.locator('#error-kind').select_option('center_missing')
            ready_case()
            assert 'SAME FRAME' in page.locator('#error-detail').inner_text()
            page.locator('#error-kind').select_option('center_offset')
            ready_case()
            assert 'Center matched' in page.locator('#error-detail').inner_text()
            assert 'Above the 3 µm' in page.locator('#error-detail').inner_text()
            page.locator('#error-kind').select_option('division_missing')
            ready_case()
            assert 'daughter' in page.locator('#error-detail').inner_text()
            for plane, depth_axis in [('xy', 0), ('xz', 1), ('yz', 2)]:
                page.locator('#case-plane').select_option(plane)
                page.wait_for_function('(axis) => window.errorCaseView?.current?.tiles[0]?.d === axis', arg=depth_axis)
                # Independent native-volume samples catch accidentally rendering
                # the target frame twice or swapping axes in the crop.
                assert page.evaluate('''async () => {
                    const s=window.centerComparison.state, tiles=window.errorCaseView.current.tiles, shape=s.seq.shape;
                    for(const tile of tiles){
                        const response=await fetch(`frames/${tile.key}.gray.gz`);
                        const raw=new Uint8Array(await new Response(response.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
                        for(const [x,y] of [[0,0],[Math.floor(tile.width/2),Math.floor(tile.height/2)],[tile.width-1,tile.height-1]]) {
                            const p=[0,0,0];p[tile.h]=tile.u0+x;p[tile.v]=tile.v0+y;
                            let value=0;
                            if(p[tile.h]>=0&&p[tile.h]<shape[tile.h]&&p[tile.v]>=0&&p[tile.v]<shape[tile.v])
                                for(let d=tile.low;d<=tile.high;d++) {p[tile.d]=d;value=Math.max(value,raw[(p[0]*shape[1]+p[1])*shape[2]+p[2]]);}
                            if(value!==tile.gray[y*tile.width+x])return false;
                        }
                    }return true;
                }''')
            page.locator('#case-plane').select_option('xy')
            ready_case()
            page.locator('#case-projection').select_option('mip')
            page.wait_for_function('window.errorCaseView?.current?.tiles.every(t=>t.low===0 && t.high===63)')
            page.locator('#case-projection').select_option('local')
            ready_case()
            page.locator('#case-isolate').uncheck()
            ready_case()
            page.locator('#case-isolate').check()
            ready_case()
            for width in [390, 768, 1600]:
                page.set_viewport_size({'width': width, 'height': 1100})
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), width
            page.set_viewport_size({'width': 390, 'height': 900})
            page.screenshot(path=str(output / f'{method}-error-mobile.png'), full_page=True)
            page.set_viewport_size({'width': 1600, 'height': 1100})
            page.locator('#case-clear').click()
            assert page.locator('#case-visual').is_hidden()
            assert page.locator('#comparison-view').evaluate('(e)=>e.open')
        page.locator('#mode').select_option('matches')
        page.locator('#sequence').select_option('44b6_8f5ab931-movie')
        page.wait_for_function("window.centerComparison.state.seq.id === '44b6_8f5ab931-movie' && !window.centerComparison.state.busy")
        assert not page.locator('#play').is_disabled()
        page.locator('#sequence').select_option('44b6_a21120c2-pilot')
        page.wait_for_function("window.centerComparison.state.seq.id === '44b6_a21120c2-pilot' && !window.centerComparison.state.busy")
        assert page.locator('#play').is_disabled()
        assert page.locator('#time-label').inner_text() == '25'
        page.locator('#next').click()
        page.wait_for_function('window.centerComparison.state.frame.t === 75 && !window.centerComparison.state.busy')
        assert page.evaluate('window.centerComparison.state.frame.links.focus_centroid.length') == 0
        # Actual mask pixels are independently pickable when no point markers are shown.
        for control in ['gt', 'centers', 'lines']:
            page.locator('#' + control).uncheck()
        pick = page.evaluate('''() => {
          const p=window.centerComparison.projection();
          for(let y=30;y<p.height-30;y++) for(let x=30;x<p.width-30;x++) {
            const label=p.labels[y*p.width+x];
            if(label)return {x:(x+.5)/p.width,y:(y+.5)/p.height,label};
          }
        }''')
        box = page.locator('#focus-canvas').bounding_box()
        page.mouse.click(box['x'] + pick['x'] * box['width'], box['y'] + pick['y'] * box['height'])
        assert page.evaluate('window.centerComparison.state.selection.id') == pick['label']
        assert 'FOCUS instance' in page.locator('#inspector').inner_text()
        for width in [390, 768, 1600]:
            page.set_viewport_size({'width': width, 'height': 1000})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), width
        page.set_viewport_size({'width': 390, 'height': 900})
        page.screenshot(path=str(output / 'mobile.png'), full_page=True)
        assert not errors, errors
        assert not failed, failed
        assert all(urlsplit(r).hostname in ('127.0.0.1', 'localhost') for r in requests), requests
        browser.close()
    receipt = dict(passed=True, method_radius_combinations=comparisons, errors=errors, failed_requests=failed,
        synchronized_pixels=True, native_axis_and_mask_pixel_checks=True, playback_and_pause=True,
        rapid_seek=True, snapshot_gap_guard=True, temporal_links=True, mask_picking=True,
        selection_and_view_link=True, viewport_widths=[390, 768, 1600], external_requests=0)
    receipt.update(complete_pipeline_models=['pooled', 'best'], error_navigation=True,
                   error_context_frames=True, error_share_links=True, whole_population_scores=True,
                   separate_center_and_link_cases=True, paired_frame_pixels_verified=True,
                   correct_daughter_preserved=True, case_isolation=True, error_mobile_layout=True)
    (output / 'browser-validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8767')
    parser.add_argument('--output', type=Path, default=Path('/kaggle/working/cell-tracking/center-comparison/checks'))
    args = parser.parse_args()
    verify(args.url, args.output)

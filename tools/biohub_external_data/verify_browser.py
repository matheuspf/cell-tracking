"""Check the offline guide's interactions, links and responsive rendering."""
from pathlib import Path
from urllib.parse import urlsplit, unquote
import json
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parents[2] / "work/biohub-data-guide"
checks = root / "checks"
checks.mkdir(exist_ok=True)
visuals = json.loads((root / "visual_examples.json").read_text())
trajectories = json.loads((root / "trajectory_examples.json").read_text())
errors, requests, results = [], [], []
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("request", lambda request: requests.append(request.url))
    page.goto((root / "index.html").as_uri(), wait_until="load")
    assert page.locator("#point-layer circle").count() == 201
    assert page.locator("#sample-count").inner_text() == "201"
    page.locator("#frame-slider").fill("5")
    assert page.locator("#sample-count").inner_text() == "248"
    assert page.locator("#point-layer circle").count() == 248
    page.locator("#show-labels").uncheck()
    assert page.locator("#point-layer circle").count() == 0
    page.locator("#show-labels").check()
    page.locator("#sample-select").select_option("real")
    assert page.locator("#sample-count").inner_text() == "1"
    assert page.locator("#frame-slider").is_disabled()
    assert page.locator("#projection").get_attribute("viewBox") == "0 0 256 256"
    page.locator("#sample-select").select_option("static")
    assert page.locator("#sample-count").inner_text() == "201"
    page.locator("#coord-z").fill("10")
    page.locator("#coord-y").fill("40")
    page.locator("#coord-x").fill("80")
    assert page.locator("#coord-output").inner_text() == "Pooled: (10, 10, 20)\nµm: (16.25, 16.25, 32.5)"
    page.locator("#coord-z").fill("64")
    assert "Use 0 ≤ Z < 64" in page.locator("#coord-error").inner_text()
    page.locator("#coord-z").fill("")
    assert "Enter a valid point" in page.locator("#coord-output").inner_text()
    for name, value in [("z", "32"), ("y", "128"), ("x", "128")]:
        page.locator("#coord-" + name).fill(value)
    # Exercise the newly added examples against their source-derived metadata.
    assert page.locator(".gallery-tile").count() == 6
    page.locator("#gallery-label-toggle").check()
    assert page.locator(".gallery-labels circle").count() == sum(x["count"] for x in visuals["gallery"])
    assert page.locator(".gallery-labels").first.is_visible()
    page.locator("#gallery-label-toggle").uncheck()
    assert not page.locator(".gallery-labels").first.is_visible()
    for i, sample in enumerate(visuals["depth"]):
        page.locator("#depth-sample").select_option(str(i))
        for z in [0, sample["default_z"], 63]:
            page.locator("#depth-slider").fill(str(z))
            expected = sum(min(63, int(p[0] + .5)) == z for p in sample["points_zyx"])
            assert page.locator("#depth-points circle").count() == expected
            assert page.locator("#depth-image image").get_attribute("href") == sample["slices"][z]
        page.locator("#depth-label-toggle").uncheck()
        assert page.locator("#depth-points circle").count() == 0
        page.locator("#depth-label-toggle").check()
    page.locator("#depth-sample").select_option("0")
    page.locator("#division-label-toggle").uncheck()
    assert not page.locator(".division-marks").first.is_visible()
    page.locator("#division-label-toggle").check()
    assert page.locator(".filmstrip .visual-figure").count() == len(visuals["division"]["frames"])
    for i, item in enumerate(trajectories["zoo"]):
        page.locator("#zoo-species").select_option(str(i))
        for plane in ["xy", "xz", "yz"]:
            page.locator("#zoo-plane").select_option(plane)
            for time in [item["window"]["start"], item["window"]["end"]]:
                page.locator("#zoo-time").fill(str(time))
                assert page.locator("#zoo-current circle").count() == sum(n["t"] == time for n in item["nodes"])
                by_id = {n["id"]: n for n in item["nodes"]}
                assert page.locator("#zoo-edges line").count() == sum(by_id[v]["t"] <= time for _, v in item["edges"])
                assert page.locator("#zoo-forks circle").count() == sum(by_id[v]["t"] < time for v in item["division_parent_ids"])
        if item["species"] == "mouse":
            assert page.locator("#zoo-forks circle").count() == 0
    page.locator("#zoo-species").select_option("5")
    page.locator("#zoo-plane").select_option("xy")
    for i, frame in enumerate(trajectories["riken"]["frames"]):
        page.locator("#riken-time").fill(str(i))
        assert page.locator("#riken-points circle").count() == frame["count"]
    page.locator("#riken-time").fill("0")
    for i, feature in enumerate(trajectories["riken"]["features"]):
        page.locator("#riken-measurement").select_option(str(i))
        missing = any(v is None for v in feature["fwhm_size_zyx_um"])
        assert ("missing" in page.locator("#riken-measurements").inner_text()) == missing
        rectangle_available = all(feature["fwhm_min_zyx_um"][j] is not None and feature["fwhm_max_zyx_um"][j] is not None for j in [1, 2])
        assert page.locator("#riken-box rect").count() == (2 if rectangle_available else 1)
    page.locator("#riken-measurement").select_option("0")
    links = page.locator("a[href]").evaluate_all("(elements) => elements.map(e => e.getAttribute('href'))")
    local_links = 0
    for href in links:
        parts = urlsplit(href)
        if parts.scheme or parts.netloc:
            continue
        if parts.path:
            assert (root / unquote(parts.path)).resolve().is_file(), href
        elif parts.fragment:
            assert page.locator("[id=" + json.dumps(parts.fragment) + "]").count() == 1, href
        local_links += 1
    for detail in page.locator("details").all():
        detail.locator("summary").click()
        assert detail.get_attribute("open") is not None
        detail.locator("summary").click()
        assert detail.get_attribute("open") is None
    page.locator("#sample-select").select_option("sequence")
    page.locator("#frame-slider").fill("0")
    for width in [320, 390, 768, 1440]:
        page.set_viewport_size({"width": width, "height": 1000})
        page.evaluate("window.scrollTo(0,0)")
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), f"Overflow at {width}"
        results.append({"width": width, "no_document_overflow": True})
        if width in [390, 1440]:
            page.screenshot(path=str(checks / f"guide-{width}-top.png"))
            # Fixed navigation should not cover long element screenshots.
            screenshot_style = page.add_style_tag(content=".bar,.skip{visibility:hidden!important}")
            page.locator(".explorer").screenshot(path=str(checks / f"guide-{width}-explorer.png"))
            page.locator(".gallery-grid").screenshot(path=str(checks / f"guide-{width}-gallery.png"))
            page.locator(".depth-shell").screenshot(path=str(checks / f"guide-{width}-depth.png"))
            page.locator("#zoo-viewer .plot-shell").screenshot(path=str(checks / f"guide-{width}-zoo.png"))
            page.locator("#riken-viewer .riken-grid").screenshot(path=str(checks / f"guide-{width}-riken.png"))
            screenshot_style.evaluate("(element)=>element.remove()")
    page.locator("#heatmaps").screenshot(path=str(checks / "guide-heatmap.png"))
    page.locator("#division-film").screenshot(path=str(checks / "guide-division-film.png"))
    page.set_viewport_size({"width": 1440, "height": 1000})
    page.evaluate("window.scrollTo(0,0)")
    page.pdf(path=str(checks / "print-preview.pdf"), format="A4", print_background=True)
    assert not errors, errors
    remote_requests = [url for url in requests if url.startswith(("http:", "https:"))]
    assert not remote_requests, remote_requests
    report = {"status": "passed", "browser": "Chromium " + browser.version,
              "offline_file_url": True, "remote_requests": remote_requests, "javascript_errors": errors,
              "valid_local_links_and_anchors": local_links, "viewports": results,
              "interactions": ["six-frame slider", "sample selector", "point toggle", "coordinate units",
                               "invalid/empty coordinate handling", "collapsible detail sections",
                               "six gallery overlays", "two 64-plane depth volumes", "division crop overlays",
                               "six Zoo species and three projection planes", "ten RIKEN point frames",
                               "ten measurement features including missing FWHM"],
              "new_data_validation": ["checks/visual-examples-verification.json", "checks/trajectory-verification.json"],
              "print_render": "checks/print-preview.pdf"}
    browser.close()
(checks / "browser-verification.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))

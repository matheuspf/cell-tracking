"""Build the standalone offline HTML from measured local data."""
from pathlib import Path
import html
import json
import copy

root = Path(__file__).resolve().parents[2] / "work/biohub-data-guide"
docs = Path(__file__).resolve().parents[2] / "docs/external-data-guide"
catalog = json.loads((root / "catalog.json").read_text())
visual = json.loads((root / "visual_examples.json").read_text())
trajectories = json.loads((root / "trajectory_examples.json").read_text())


def escape(value):
    return html.escape(str(value), quote=True)


def raster_svg(image, width, height, title, points=(), group_class="", radius=None):
    radius = width / 155 if radius is None else radius
    rings = "".join(f'<circle cx="{p[2]:.4f}" cy="{p[1]:.4f}" r="{radius:.4f}" />' for p in points)
    raster = f'<image href="{image}" width="{width}" height="{height}"/>' if image else ""
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">'
        f'<title>{escape(title)}</title>{raster}'
        f'<g class="{escape(group_class)}" fill="none" stroke="#dcf984" stroke-width="{width/400:.4f}"'
        f' style="filter:drop-shadow(0 0 {width/450:.4f}px #0e241c)">{rings}</g></svg>'
    )


gallery = []
for item in visual["gallery"]:
    z, y, x = item["shape_zyx"]
    count_label = "released centres" if item["kind"] == "synthetic" else "available annotations"
    gallery.append(
        '<figure class="visual-figure gallery-tile">'
        + raster_svg(item["image"], x, y, item["title"] + " — max-Z projection", item["points_zyx"], "gallery-labels")
        + f'<figcaption><strong>{escape(item["title"])}</strong>'
        + f'<span>{item["count"]:,} {count_label} · {z} × {y} × {x} voxels</span>'
        + f'<span class="mini-source">{escape(item["sample"])} · t={item["t"]}</span>'
        + f'<span class="mini-source">Display range: {item["contrast"]["low"]:g}–{item["contrast"]["high"]:g} intensity units</span>'
        + '</figcaption></figure>'
    )

heat = visual["heatmap"]
hz, hy, hx = heat["shape_zyx"]
heat_panels = []
for image, points, title, caption in [
    (heat["image"], [], "1 · Image", f'Simulated fluorescence, sampled to {hz}³; max-Z projection.'),
    (None, heat["points_zyx"], "2 · Point labels", f'{heat["count"]} source centres projected onto XY. Depth is retained in the training arrays.'),
    (heat["target_image"], [], "3 · Generated heatmap", f'Max-Z response from Gaussian centre targets, σ={heat["sigma_um"]:g} µm; values 0–1.'),
]:
    heat_panels.append('<figure class="visual-figure">' + raster_svg(image, hx, hy, title, points)
                       + f'<figcaption><strong>{title}</strong><span>{caption}</span></figcaption></figure>')

division = visual["division"]
dz, dy, dx = division["crop_shape_zyx"]
division_frames = []
for frame in division["frames"]:
    marks = []
    for p in frame["points"]:
        x, y = p["xy_crop"]
        colour = "#ffbf68" if p["role"] == "parent" else "#d9fa8d"
        # Labels are drawn in a fixed SVG grid; offsets scale with the crop.
        marks.append(f'<circle cx="{x:.4f}" cy="{y:.4f}" r=".6" fill="none" stroke="{colour}" stroke-width=".16"/>'
                     f'<text x="{x+.85:.4f}" y="{y-.7:.4f}" fill="{colour}" stroke="#0d2018" stroke-width=".12"'
                     f' paint-order="stroke" font-size=".9">{p["node_id"]}</text>')
    ids = ", ".join(str(p["node_id"]) for p in frame["points"])
    division_frames.append(
        f'<figure class="visual-figure"><svg viewBox="0 0 {dx} {dy}" role="img" aria-label="Synthetic lineage crop at t={frame["t"]}, nodes {ids}">'
        f'<image href="{frame["image"]}" width="{dx}" height="{dy}"/>'
        f'<g class="division-marks">{"".join(marks)}</g></svg>'
        f'<figcaption><strong>t = {frame["t"]}</strong><span class="frame-label-list">Node IDs: {ids}</span></figcaption></figure>'
    )

# Images already rendered above need not be duplicated in the runtime JSON.
runtime_visual = copy.deepcopy(visual)
for item in runtime_visual["gallery"]:
    item.pop("image")
runtime_visual["heatmap"].pop("image")
runtime_visual["heatmap"].pop("target_image")
for frame in runtime_visual["division"]["frames"]:
    frame.pop("image")

def embedded_json(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).replace("<", "\\u003c")

names = {"ascidian": "Ascidian", "drosophila": "Drosophila", "elegans": "C. elegans",
         "mouse": "Mouse", "tribolium": "Tribolium", "zebrafish": "Zebrafish"}
rows = []
for item in catalog["zoo"]:
    name = Path(item["prepared"]).stem.removesuffix("_graph")
    cells = "".join(f'<td class="num">{item[key]:,}</td>' for key in ("frames", "nodes", "tracklets", "division_parents"))
    rows.append(f'<tr><td><a href="{html.escape(item["prepared"], quote=True)}">{names[name]}</a></td>{cells}</tr>')
replacements = {
    "@@ZOO_MILLIONS@@": f'{sum(x["nodes"] for x in catalog["zoo"])/1e6:.2f}',
    "@@ZOO_ROWS@@": "\n".join(rows),
    "@@FIRST_IMAGE@@": catalog["previews"]["sequence"][0]["image"],
    "@@CATALOG_JSON@@": embedded_json(catalog),
    "@@VISUAL_JSON@@": embedded_json(runtime_visual),
    "@@TRAJECTORY_JSON@@": embedded_json(trajectories),
    "@@VISUAL_SCRIPT@@": (docs / "visuals.js").read_text(),
    "@@GALLERY_CARDS@@": "\n".join(gallery),
    "@@HEATMAP_PANELS@@": "\n".join(heat_panels),
    "@@DIVISION_FRAMES@@": "\n".join(division_frames),
    "@@DIVISION_CAPTION@@": escape(
        f'{division["sample"]}: context node {division["context_id"]}, parent {division["parent_id"]}, '
        f'daughters {division["daughter_ids"]}. Fixed crop: Z/Y/X origin {division["crop_origin_zyx"]}, '
        f'shape {division["crop_shape_zyx"]} on the sampled grid (1.625 µm/voxel). '
        f'Each image projects the same {dz}-slice Z slab. Selected from the source graph; not model predictions.'
    ),
    "@@DEPTH_OPTIONS@@": "".join(f'<option value="{i}">{escape(v["title"])}</option>' for i, v in enumerate(visual["depth"])),
    "@@ZOO_OPTIONS@@": "".join(f'<option value="{i}">{escape(v["label"])}</option>' for i, v in enumerate(trajectories["zoo"])),
    "@@RIKEN_OPTIONS@@": "".join(f'<option value="{i}">{escape(v["measurement_id"])} · t={v["t"]}</option>' for i, v in enumerate(trajectories["riken"]["features"])),
}
page = (root / "guide.template.html").read_text()
for key, value in replacements.items():
    page = page.replace(key, value)
if "@@" in page:
    raise RuntimeError("An HTML placeholder was not replaced")
(root / "index.html").write_text(page)
print(f"Wrote {root / 'index.html'} ({len(page.encode()):,} bytes)")

"use strict";
// Data and images are embedded at build time; no network or file fetches.
const visuals = JSON.parse(byId("visual-data").textContent);
const trajectories = JSON.parse(byId("trajectory-data").textContent);
const svgNS = "http://www.w3.org/2000/svg";
const numberText = (value, digits=1) => Number(value.toFixed(digits)).toLocaleString("en-US");
function svgElement(tag, attributes={}, content) {
  const el = document.createElementNS(svgNS, tag);
  for (const [key, value] of Object.entries(attributes)) el.setAttribute(key, value);
  if (content !== undefined) el.textContent = content;
  return el;
}
function fact(parent, label, value) {
  const span = document.createElement("span"), strong = document.createElement("strong");
  strong.textContent = value; span.append(strong, " " + label); parent.append(span);
}
function toggleGroups(selector, visible) {
  document.querySelectorAll(selector).forEach(el => el.style.display = visible ? "" : "none");
}
byId("gallery-label-toggle").addEventListener("change", () => {
  document.querySelectorAll(".gallery-labels").forEach(el => el.style.display = byId("gallery-label-toggle").checked ? "inline" : "none");
});
byId("division-label-toggle").addEventListener("change", () => toggleGroups(".division-marks", byId("division-label-toggle").checked));

function renderDepth(reset=false) {
  const item = visuals.depth[Number(byId("depth-sample").value)];
  const slider = byId("depth-slider"), zmax = item.shape_zyx[0]-1;
  slider.max = zmax;
  if (reset) slider.value = item.default_z;
  const z = Number(slider.value), [_, height, width] = item.shape_zyx;
  const points = item.points_zyx.filter(p => Math.min(zmax, Math.floor(p[0]+.5)) === z);
  const plot = byId("depth-image");
  plot.setAttribute("viewBox", "0 0 " + width + " " + height);
  plot.replaceChildren(svgElement("title", {}, item.title + ", Z=" + z),
    svgElement("image", {href:item.slices[z], width, height}));
  const layer = svgElement("g", {id:"depth-points"});
  if (byId("depth-label-toggle").checked) {
    for (const p of points) layer.append(svgElement("circle", {cx:p[2],cy:p[1],r:2.2,class:"centre-ring"}));
  }
  plot.append(layer);
  byId("depth-title").textContent = item.sample + " · t=" + item.t;
  byId("depth-value").textContent = z + " / " + zmax;
  byId("depth-readout").textContent = points.length + " centre" + (points.length===1 ? "" : "s") +
    " in this plane / " + item.count + " in the volume · Z=" + numberText(z*item.voxel_um_zyx[0],3) + " µm";
  byId("depth-caption").textContent = item.kind==="real" ?
    "Real embryo data with sparse GEFF annotations. Display limits: "+item.contrast.low+"–"+item.contrast.high+" source intensity units." :
    "Synthetic volume with released centre labels. Display limits: "+item.contrast.low+"–"+item.contrast.high+" source intensity units.";
}

// Equal coordinate increments receive equal screen distances within a plot.
// Zoo coordinate units remain uncalibrated; this is not a physical-size map.
function plotAxes(plot, bounds, plane, units) {
  const indices = {xy:[2,1],xz:[2,0],yz:[1,0]}[plane];
  const names = ["Z","Y","X"], lo=indices.map(i=>bounds.min[i]), hi=indices.map(i=>bounds.max[i]);
  const span = hi.map((v,i)=>Math.max(v-lo[i],1));
  const scale = Math.min(460/span[0],310/span[1]);
  const cx=(lo[0]+hi[0])/2, cy=(lo[1]+hi[1])/2;
  const sx=v=>315+(v-cx)*scale, sy=v=>183-(v-cy)*scale;
  const xmin=sx(lo[0]), xmax=sx(hi[0]), ymin=sy(hi[1]), ymax=sy(lo[1]);
  plot.replaceChildren();
  plot.append(svgElement("path",{d:"M"+xmin+" "+ymin+" V"+ymax+" H"+xmax,fill:"none",stroke:"#bbc9bd","stroke-width":1}));
  for (let i=0; i<3; i++) {
    const xv=lo[0]+(hi[0]-lo[0])*i/2, yv=lo[1]+(hi[1]-lo[1])*i/2;
    plot.append(svgElement("text",{x:sx(xv),y:ymax+21,"text-anchor":"middle"},numberText(xv)),
      svgElement("text",{x:xmin-10,y:sy(yv)+4,"text-anchor":"end"},numberText(yv)));
  }
  plot.append(svgElement("text",{x:315,y:406,"text-anchor":"middle",class:"axis-label"},names[indices[0]]+" ("+units+")"),
    svgElement("text",{transform:"translate(17 183) rotate(-90)","text-anchor":"middle",class:"axis-label"},names[indices[1]]+" ("+units+")"));
  const keys=indices.map(i=>["z","y","x"][i]);
  return {sx,sy,x:n=>sx(n[keys[0]]),y:n=>sy(n[keys[1]])};
}

function renderZoo(reset=false) {
  const item=trajectories.zoo[Number(byId("zoo-species").value)], slider=byId("zoo-time");
  slider.min=item.window.start; slider.max=item.window.end;
  if(reset) slider.value=item.window.end;
  const t=Number(slider.value), plane=byId("zoo-plane").value;
  const plot=byId("zoo-plot"), axes=plotAxes(plot,item.bbox_zyx,plane,"source units");
  const nodes=new Map(item.nodes.map(n=>[n.id,n]));
  plot.append(svgElement("title",{},item.label+" trajectories, "+plane.toUpperCase()+", time "+t+", source units"));
  const edges=svgElement("g",{id:"zoo-edges",stroke:"#328570","stroke-width":1.1,"stroke-opacity":.63});
  let visibleEdges=0;
  for(const [u,v] of item.edges) {
    const a=nodes.get(u), b=nodes.get(v);
    if(b.t<=t) {
      edges.append(svgElement("line",{x1:axes.x(a),y1:axes.y(a),x2:axes.x(b),y2:axes.y(b)}));
      visibleEdges++;
    }
  }
  plot.append(edges);
  const current=svgElement("g",{id:"zoo-current",fill:"#174f41",stroke:"#fffefa","stroke-width":.5});
  const points=item.nodes.filter(n=>n.t===t);
  for(const n of points) {
    const dot=svgElement("circle",{cx:axes.x(n),cy:axes.y(n),r:2.8});
    dot.append(svgElement("title",{},"Node "+n.id+" · tracklet "+n.tracklet_id+" · t="+n.t)); current.append(dot);
  }
  plot.append(current);
  const forks=svgElement("g",{id:"zoo-forks",fill:"none",stroke:"#a95c16","stroke-width":1.5});
  for(const id of item.division_parent_ids) {
    const n=nodes.get(id);
    if(n.t<t) forks.append(svgElement("circle",{cx:axes.x(n),cy:axes.y(n),r:4.1}));
  }
  plot.append(forks);
  byId("zoo-time-value").textContent=t+" / "+item.window.end;
  byId("zoo-facts").replaceChildren();
  fact(byId("zoo-facts"),"current nodes",points.length);
  fact(byId("zoo-facts"),"links shown",visibleEdges);
  fact(byId("zoo-facts"),"observations in excerpt",item.nodes.length);
  fact(byId("zoo-facts"),"forks in excerpt",item.division_parent_ids.length);
  byId("zoo-caption").textContent=item.label+" · source frames "+item.window.start+"–"+item.window.end+
    ". "+item.selection_note+" Projected crossings do not prove a collision or an identity switch; the hidden axis still matters.";
}

function renderRiken() {
  const r=trajectories.riken, frame=r.frames[Number(byId("riken-time").value)], plot=byId("riken-plot");
  const axes=plotAxes(plot,r.bbox_zyx,"xy","µm"), group=svgElement("g",{id:"riken-points",fill:"#245a48","fill-opacity":.8});
  for(const n of frame.points) {
    const dot=svgElement("circle",{cx:axes.x(n),cy:axes.y(n),r:2.5});
    dot.append(svgElement("title",{},"Measurement "+n.id+" · t="+frame.t+" · Z="+numberText(n.z)+" µm"));
    group.append(dot);
  }
  plot.append(group);
  byId("riken-time-value").textContent="t="+frame.t+" · "+numberText(frame.elapsed_seconds/60,1)+" min";
  byId("riken-facts").replaceChildren();
  fact(byId("riken-facts"),"point measurements",frame.count);
  fact(byId("riken-facts"),"second frame spacing",r.time_step_seconds);
  fact(byId("riken-facts"),"verified temporal links",0);
  byId("riken-source-note").textContent="Animal C · all 1,312 prepared measurements over 13.5 minutes. "+
    "Elapsed time starts at the first displayed frame (HDF5 object time index 1), not at a known developmental age. "+
    "Feature examples are source measurements at t=0. Fluorescence values are in arbitrary units and cannot directly be compared with Biohub intensities.";
}

function renderMeasurement() {
  const f=trajectories.riken.features[Number(byId("riken-measurement").value)];
  const bboxMin=f.bbox_min_zyx_um, bboxMax=f.bbox_max_zyx_um;
  const fullWidthXY=[1,2].every(i=>Number.isFinite(f.fwhm_min_zyx_um[i])&&Number.isFinite(f.fwhm_max_zyx_um[i]));
  const lo=bboxMin.slice(), hi=bboxMax.slice();
  if(fullWidthXY) for(const i of [1,2]) {
    lo[i]=Math.min(lo[i],f.fwhm_min_zyx_um[i]); hi[i]=Math.max(hi[i],f.fwhm_max_zyx_um[i]);
  }
  const centre=[f.z,f.y,f.x];
  for(const i of [1,2]) {lo[i]=Math.min(lo[i],centre[i]);hi[i]=Math.max(hi[i],centre[i]);}
  const padding=Math.max(hi[1]-lo[1],hi[2]-lo[2])*.12;
  for(const i of [1,2]) {lo[i]-=padding;hi[i]+=padding;}
  const plot=byId("riken-box"), axes=plotAxes(plot,{min:lo,max:hi},"xy","µm");
  function box(min,max,colour,dash) {
    plot.append(svgElement("rect",{x:axes.sx(min[2]),y:axes.sy(max[1]),
      width:axes.sx(max[2])-axes.sx(min[2]),height:axes.sy(min[1])-axes.sy(max[1]),
      fill:"none",stroke:colour,"stroke-width":2,"stroke-dasharray":dash||"none"}));
  }
  box(bboxMin,bboxMax,"#328570");
  if(fullWidthXY) box(f.fwhm_min_zyx_um,f.fwhm_max_zyx_um,"#a95c16","5 4");
  plot.append(svgElement("circle",{cx:axes.sx(f.x),cy:axes.sy(f.y),r:4,fill:"#183d30"}));
  if(!fullWidthXY) plot.append(svgElement("text",{x:315,y:376,"text-anchor":"middle"},"FWHM rectangle unavailable: an XY bound is missing"));
  const list=byId("riken-measurements");list.replaceChildren();
  const dims=values=>[values[2],values[1],values[0]].map(v=>Number.isFinite(v)?numberText(v):"missing").join(" × ")+" µm";
  for(const [label,value] of [
    ["Core box X × Y × Z",dims(f.bbox_size_zyx_um)],["FWHM X × Y × Z",dims(f.fwhm_size_zyx_um)],
    ["Mean fluorescence",Number.isFinite(f.mean_intensity_au)?numberText(f.mean_intensity_au,2)+" a.u.":"missing"],
    ["Centre-of-mass fluorescence",Number.isFinite(f.com_intensity_au)?numberText(f.com_intensity_au,2)+" a.u.":"missing"]
  ]) {
    const block=document.createElement("div"), term=document.createElement("dt"), valueEl=document.createElement("dd");
    term.textContent=label;valueEl.textContent=value;block.append(term,valueEl);list.append(block);
  }
}

byId("depth-sample").addEventListener("change",()=>renderDepth(true));
byId("depth-slider").addEventListener("input",()=>renderDepth());
byId("depth-label-toggle").addEventListener("change",()=>renderDepth());
byId("zoo-species").addEventListener("change",()=>renderZoo(true));
byId("zoo-plane").addEventListener("change",()=>renderZoo());
byId("zoo-time").addEventListener("input",()=>renderZoo());
byId("riken-time").addEventListener("input",renderRiken);
byId("riken-measurement").addEventListener("change",renderMeasurement);
renderDepth(true);renderZoo(true);renderRiken();renderMeasurement();

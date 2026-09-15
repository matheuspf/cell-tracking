/* Native coordinates throughout; physical scale is applied only to display and distances. */
"use strict";
(() => {
const $ = id => document.getElementById(id);
const canvases = [$("raw-canvas"), $("current-canvas"), $("focus-canvas")];
document.querySelectorAll("#comparison-panel button,#comparison-panel select,#comparison-panel input").forEach(control=>{control.disabled=true;});
const colors = {gt:"#f2f5fb", current:"#6bdef0", focus:"#ffb267", selected:"#fff28b", miss:"#ff8895"};
const S = {index:null, seq:null, frames:new Map(), position:0, frame:null, volume:null,
  plane:"xy", projection:"mip", depth:32, current:"pooled", focus:"focus_centroid", radius:"7",
  mode:"matches", zoom:1, cx:.5, cy:.5, gamma:1, opacity:.35, selection:null, playing:false,
  epoch:0, projectionCache:null, busy:false, errorKind:"all", errorScope:"sequence", activeError:null, reviewModel:"pooled"};
const volumeCache = new Map(), pending = new Map();
let timer, previousMethod = "best", caseRequest = 0;
const keyFor = (seq,t) => `${seq.dataset}-t${String(t).padStart(3,"0")}`;
const escaped = value => String(value).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const format = x => x == null ? '<span class="miss">Miss</span>' : `${x.toFixed(2)} µm`;
const point = (method,id,frame=S.frame) => frame?.points[method]?.find(p => p[0] === id);
const pairs = method => S.frame?.matches[method]?.[S.radius] || [];
function axes() { return S.plane === "xy" ? [2,1,0] : S.plane === "xz" ? [2,0,1] : [1,0,2]; }
function fail(error) { stop(); S.busy=false; $("panels").classList.remove("loading"); $("error").hidden=false; $("error").textContent=String(error.message || error); $("status").textContent="Could not load this view"; }
async function json(url) {const response=await fetch(url); if(!response.ok) throw Error(`${url}: HTTP ${response.status}`); return response.json();}
async function binary(url) {
  if(!window.DecompressionStream) throw Error("This viewer needs a current browser with gzip decompression support.");
  const response=await fetch(url); if(!response.ok) throw Error(`${url}: HTTP ${response.status}`);
  return new Response(response.body.pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
}
async function volume(key,shape) {
  if(volumeCache.has(key)) {const value=volumeCache.get(key); volumeCache.delete(key); volumeCache.set(key,value); return value;}
  if(pending.has(key)) return pending.get(key);
  const job=(async()=>{
    const [grayBuffer,labelBuffer]=await Promise.all([binary(`frames/${key}.gray.gz`),binary(`frames/${key}.labels.gz`)]);
    const count=shape.reduce((a,b)=>a*b,1);
    if(grayBuffer.byteLength!==count || labelBuffer.byteLength!==count*4) throw Error(`Truncated image or mask data: ${key}`);
    const value={gray:new Uint8Array(grayBuffer), labels:new Uint32Array(labelBuffer)};
    volumeCache.set(key,value);
    while(volumeCache.size>6) volumeCache.delete(volumeCache.keys().next().value);
    return value;
  })(); pending.set(key,job);
  try{return await job;}finally{pending.delete(key);}
}
function prefetch() {
  const seq=S.seq;
  for(let offset=1;offset<=2;offset++) {
    const i=S.position+offset;
    if(i<seq.times.length) volume(keyFor(seq,seq.times[i]),seq.shape).catch(()=>{});
  }
}
function stop() {S.playing=false; clearTimeout(timer); $("play").textContent="Play"; $("play").setAttribute("aria-label","Play sequence");}
async function playTick() {
  if(!S.playing) return;
  const before=performance.now();
  await showTime((S.position+1)%S.seq.times.length);
  if(S.playing) timer=setTimeout(playTick,Math.max(0,1000/Number($("fps").value)-(performance.now()-before)));
}
function play() {
  if(S.playing) return stop();
  if(!S.seq?.continuous || S.busy) return;
  S.playing=true; $("play").textContent="Pause"; $("play").setAttribute("aria-label","Pause sequence");
  timer=setTimeout(playTick,1000/Number($("fps").value));
}
async function selectSequence(id,initialTime,keepReview=false) {
  if(!keepReview){caseRequest++;$("comparison-view").open=true;}
  stop(); const epoch=++S.epoch; S.busy=true; $("panels").classList.add("loading");
  $("status").textContent="Loading sequence…";
  try {
    const seq=S.index.sequences.find(s=>s.id===id);
    if(!seq) throw Error("Unknown sequence");
    const frameData=await Promise.all(seq.times.map(t=>json(`frames/${keyFor(seq,t)}.json`)));
    const position=Math.max(0,seq.times.indexOf(Number(initialTime)));
    const v=await volume(keyFor(seq,seq.times[position]),seq.shape);
    if(epoch!==S.epoch) return;
    S.seq=seq; S.frames=new Map(frameData.map(f=>[f.t,f])); S.position=position;
    S.frame=frameData[position]; S.volume=v; S.selection=null; S.activeError=null; S.zoom=1; S.cx=.5; S.cy=.5;
    S.depth=Math.floor(seq.shape[axes()[2]]/2); S.projectionCache=null; S.busy=false;
    $("sequence").value=seq.id; $("time").max=seq.times.length-1; $("time").value=position;
    $("play").disabled=!seq.continuous; $("panels").classList.remove("loading"); $("error").hidden=true;
    $("status").textContent=seq.continuous ? "Continuous frames · real images & masks" : "Two snapshots · playback disabled across the gap";
    updateControls(); render(); prefetch();
  } catch(error) {if(epoch===S.epoch) fail(error);}
}
async function showTime(position,keepReview=false) {
  if(!keepReview){caseRequest++;$("comparison-view").open=true;}
  if(!S.seq) return;
  position=Math.max(0,Math.min(S.seq.times.length-1,position));
  const epoch=++S.epoch, t=S.seq.times[position], frame=S.frames.get(t);
  S.busy=true; $("panels").classList.add("loading"); $("status").textContent=`Loading frame ${t}…`;
  try {
    const v=await volume(frame.key,frame.shape);
    if(epoch!==S.epoch) return;
    S.frame=frame; S.volume=v; S.position=position; S.selection=null; S.activeError=null; S.projectionCache=null; S.busy=false;
    $("time").value=position; $("panels").classList.remove("loading"); $("error").hidden=true;
    $("status").textContent=S.seq.continuous ? "Continuous frames · real images & masks" : "Two snapshots · 50-frame gap";
    updateControls(); render(); prefetch();
  } catch(error) {if(epoch===S.epoch) fail(error);}
}
function updateControls() {
  const [,,d]=axes();
  $("depth").max=S.seq.shape[d]-1; S.depth=Math.max(0,Math.min(S.seq.shape[d]-1,S.depth)); $("depth").value=S.depth;
  $("depth").disabled=S.projection==="mip";
  $("depth-label").textContent=`${"ZYX"[d]} ${S.depth} · ${(S.depth*S.seq.spacing[d]).toFixed(2)} µm`;
  $("time-label").textContent=`${S.frame.t}`;
  $("range-label").textContent=S.seq.continuous ? ` / ${S.seq.times[0]}–${S.seq.times.at(-1)}` : " / snapshots 25, 75";
  $("current").disabled=false;
  for(const option of $("current").options)option.disabled=S.mode==="tracks"&&!['best','pooled','selected'].includes(option.value);
  $("current").value=S.current; $("projection").value=S.projection; $("plane").value=S.plane;$("case-plane").value=S.plane;$("case-brightness").value=S.gamma;
  $("gamma-label").textContent=S.gamma.toFixed(1); $("opacity-label").textContent=`${Math.round(S.opacity*100)}%`;
  $("previous").disabled=S.position===0; $("next").disabled=S.position===S.seq.times.length-1;
}
function projection() {
  const [h,v,d]=axes(), shape=S.seq.shape;
  const token=`${S.frame.key}|${S.plane}|${S.projection}|${S.depth}`;
  if(S.projectionCache?.token===token) return S.projectionCache;
  const width=shape[h],height=shape[v],gray=new Uint8Array(width*height),labels=new Uint32Array(width*height);
  const strides=[shape[1]*shape[2],shape[2],1];
  const low=S.projection==="mip" ? 0 : Math.max(0,S.depth-(S.projection==="slab"?2:0));
  const high=S.projection==="mip" ? shape[d]-1 : Math.min(shape[d]-1,S.depth+(S.projection==="slab"?2:0));
  for(let y=0;y<height;y++) for(let x=0;x<width;x++) {
    const base=y*strides[v]+x*strides[h]; let best=-1,winning=base+low*strides[d];
    for(let depth=low;depth<=high;depth++) {const i=base+depth*strides[d],value=S.volume.gray[i];if(value>best){best=value;winning=i;}}
    const j=y*width+x; gray[j]=best; labels[j]=S.volume.labels[winning];
  }
  S.projectionCache={token,width,height,gray,labels,low,high,h,v,d}; return S.projectionCache;
}
function transform(canvas) {
  const [h,v]=axes(), pw=S.seq.shape[h]*S.seq.spacing[h], ph=S.seq.shape[v]*S.seq.spacing[v];
  const scale=Math.min(canvas.width/pw,canvas.height/ph)*S.zoom;
  return {scale,ox:canvas.width/2-S.cx*pw*scale,oy:canvas.height/2-S.cy*ph*scale,pw,ph,h,v};
}
function xy(row,tr) {return [tr.ox+(row[tr.h+1]+.5)*S.seq.spacing[tr.h]*tr.scale,tr.oy+(row[tr.v+1]+.5)*S.seq.spacing[tr.v]*tr.scale];}
function visible(row) {const p=projection(); return row[p.d+1]>=p.low && row[p.d+1]<=p.high;}
function palette(label) {
  const root=S.mode==="tracks" ? S.frame.tracks[S.focus]?.[label] ?? label : label;
  const hue=((root*137.508)%360+360)%360;
  const c=.7, x=c*(1-Math.abs((hue/60)%2-1)), m=.22;
  const rgb=hue<60?[c,x,0]:hue<120?[x,c,0]:hue<180?[0,c,x]:hue<240?[0,x,c]:hue<300?[x,0,c]:[c,0,x];
  return rgb.map(k=>Math.round((k+m)*255));
}
function textures() {
  const p=projection(),raw=document.createElement("canvas"),mask=document.createElement("canvas");
  raw.width=mask.width=p.width;raw.height=mask.height=p.height;
  const a=raw.getContext("2d").createImageData(p.width,p.height), b=mask.getContext("2d").createImageData(p.width,p.height);
  const lut=Array.from({length:256},(_,i)=>Math.round(255*Math.pow(i/255,1/S.gamma))), cmap=new Map();
  const chosen=selectedMask();
  for(let i=0;i<p.gray.length;i++) {
    const j=i*4,value=lut[p.gray[i]]; a.data[j]=a.data[j+1]=a.data[j+2]=value;a.data[j+3]=255;
    const label=p.labels[i];if(!label) continue;
    if(!cmap.has(label)) cmap.set(label,palette(label));
    const x=i%p.width,y=Math.floor(i/p.width);
    const boundary=x===0||y===0||x===p.width-1||y===p.height-1||p.labels[i-1]!==label||p.labels[i+1]!==label||p.labels[i-p.width]!==label||p.labels[i+p.width]!==label;
    const rgb=label===chosen?[255,242,139]:cmap.get(label);
    b.data[j]=rgb[0];b.data[j+1]=rgb[1];b.data[j+2]=rgb[2];
    b.data[j+3]=S.opacity===0 ? 0 : Math.round(255*(boundary?Math.max(.65,S.opacity):S.opacity));
  }
  raw.getContext("2d").putImageData(a,0,0);mask.getContext("2d").putImageData(b,0,0);return {raw,mask};
}
function symbol(ctx,x,y,kind,color,selected=false) {
  const r=selected?8:5;ctx.save();ctx.strokeStyle=color;ctx.lineWidth=selected?2.5:1.7;ctx.beginPath();
  if(kind==="gt"){ctx.moveTo(x,y-r);ctx.lineTo(x+r,y);ctx.lineTo(x,y+r);ctx.lineTo(x-r,y);ctx.closePath();}
  else if(kind==="focus"){ctx.moveTo(x-r,y);ctx.lineTo(x+r,y);ctx.moveTo(x,y-r);ctx.lineTo(x,y+r);}
  else ctx.arc(x,y,r,0,2*Math.PI);
  const weight=ctx.lineWidth;ctx.strokeStyle="rgba(0,0,0,.85)";ctx.lineWidth=weight+2.5;ctx.stroke();
  ctx.strokeStyle=color;ctx.lineWidth=weight;ctx.stroke();if(selected){ctx.beginPath();ctx.arc(x,y,r+5,0,2*Math.PI);ctx.stroke();}ctx.restore();
}
function selectedGT() {
  if(!S.selection) return null;
  if(S.selection.method==="gt") return S.selection.id;
  return pairs(S.selection.method).find(r=>r[0]===S.selection.id)?.[1] ?? null;
}
function selectedMask() {
  if(!S.selection) return null;
  if(S.selection.method.startsWith("focus")) return S.selection.id;
  const gt=selectedGT();
  if(gt!==null) return pairs(S.focus).find(r=>r[1]===gt)?.[0] ?? null;
  const p=point(S.selection.method,S.selection.id);
  if(!p) return null;
  const [z,y,x]=p.slice(1,4),shape=S.seq.shape;
  return S.volume.labels[(z*shape[1]+y)*shape[2]+x] || null;
}
function selectedPoint() {return S.selection ? point(S.selection.method,S.selection.id) : null;}
function drawTrails(ctx,method,tr,color) {
  const times=S.seq.times.slice(Math.max(0,S.position-4),S.position+1), allowed=new Set(times);
  for(const t of times) {
    const frame=S.frames.get(t);
    for(const [previous,a,b] of frame.links[method] || []) {
      if(!allowed.has(previous)) continue;
      const pa=point(method,a,S.frames.get(previous)),pb=point(method,b,frame);
      if(!pa || !pb || (!visible(pa)&&!visible(pb))) continue;
      const [x1,y1]=xy(pa,tr),[x2,y2]=xy(pb,tr);
      ctx.save();ctx.globalAlpha=.3+.7*(times.indexOf(t)+1)/times.length;ctx.strokeStyle=color;ctx.lineWidth=2;
      if(S.index.prediction_reviews?.[method]) {
        const status=frame.model_edge_status?.[method]?.[`${a}:${b}`];
        ctx.strokeStyle=status==="fp"?colors.miss:status==="unknown"?"#8191a8":color;
        if(status==="fp")ctx.lineWidth=3;
      }
      if(method.startsWith("focus")) ctx.setLineDash([5,4]);
      ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();ctx.restore();
    }
  }
}
function currentReview() {return S.index?.prediction_reviews?.[S.reviewModel];}
function allCases() {return currentReview()?.cases[S.radius]||[];}
function selectedError() {return allCases().find(e=>e.id===S.activeError||e.related_flags.includes(S.activeError));}
function drawError(ctx,kind,tr) {
  const event=selectedError();if(!event||S.current!==S.reviewModel)return;
  // Never draw another timepoint's cells on the current microscopy image.
  for(const p of event.scene.points) {
    if(p.t!==S.frame.t)continue;
    if(kind!=="current"&&p.kind!=="annotation")continue;
    if(p.kind==="annotation"&&!$("gt").checked||p.kind==="prediction"&&!$("centers").checked)continue;
    const row=[p.id,...p.zyx];if(!visible(row))continue;
    const [x,y]=xy(row,tr),color=p.role==="missing"?colors.miss:p.kind==="annotation"?colors.gt:colors.current;
    symbol(ctx,x,y,p.kind==="annotation"?"gt":"current",color,true);
    ctx.save();ctx.font='600 15px system-ui';ctx.strokeStyle='#080c12';ctx.lineWidth=4;ctx.fillStyle=color;
    const dy=p.kind==="annotation"?-15:23;ctx.strokeText(p.label,x+15,y+dy);ctx.fillText(p.label,x+15,y+dy);ctx.restore();
  }
}
function drawPanel(canvas,kind,texture) {
  const ctx=canvas.getContext("2d"),tr=transform(canvas),p=projection();
  ctx.fillStyle="#080c12";ctx.fillRect(0,0,canvas.width,canvas.height);ctx.imageSmoothingEnabled=false;
  ctx.drawImage(texture.raw,tr.ox,tr.oy,tr.pw*tr.scale,tr.ph*tr.scale);
  if(kind==="focus" && $("masks").checked) ctx.drawImage(texture.mask,tr.ox,tr.oy,tr.pw*tr.scale,tr.ph*tr.scale);
  const method=kind==="raw" ? "gt" : kind==="current" ? S.current : S.focus;
  if(S.mode==="tracks" && $("lines").checked) drawTrails(ctx,method,tr,kind==="raw"?colors.gt:colors[kind]);
  const matched=new Map(pairs(method).map(r=>[r[0],r]));
  const gselected=selectedGT();
  if(kind!=="raw" && S.mode==="matches" && $("lines").checked) {
    for(const [id,gid] of pairs(method)) {
      const a=point(method,id),b=point("gt",gid);
      if(!visible(a)||!visible(b)) continue;
      const [x1,y1]=xy(a,tr),[x2,y2]=xy(b,tr);ctx.save();ctx.strokeStyle=gselected===gid?colors.selected:colors[kind];ctx.lineWidth=gselected===gid?2.5:1.6;ctx.globalAlpha=.8;
      ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();ctx.restore();
    }
  }
  if(kind!=="raw" && $("centers").checked) for(const row of S.frame.points[method]) {
    if(!visible(row)) continue;
    const selected=(S.selection?.method===method && S.selection.id===row[0]) || (gselected!==null&&matched.get(row[0])?.[1]===gselected);
    const [x,y]=xy(row,tr);symbol(ctx,x,y,kind,selected?colors.selected:colors[kind],selected);
  }
  if($("gt").checked) for(const row of S.frame.points.gt) {
    if(!visible(row)) continue;
    const isMiss=kind!=="raw"&&!pairs(method).some(r=>r[1]===row[0]);
    const [x,y]=xy(row,tr);symbol(ctx,x,y,"gt",gselected===row[0]?colors.selected:isMiss?colors.miss:colors.gt,gselected===row[0]);
  }
  drawError(ctx,kind,tr);
  // A physical scale bar stays correct in every plane and at every zoom level.
  const length=S.zoom>3?5:10,pixels=length*tr.scale;
  ctx.fillStyle="rgba(0,0,0,.75)";ctx.fillRect(15,canvas.height-48,pixels+22,36);
  ctx.strokeStyle="#fff";ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(25,canvas.height-20);ctx.lineTo(25+pixels,canvas.height-20);ctx.stroke();
  ctx.fillStyle="#fff";ctx.font="16px system-ui";ctx.fillText(`${length} µm`,25,canvas.height-30);
  ctx.font="15px system-ui";ctx.fillText(`${"ZYX"[p.h]} → / ${"ZYX"[p.v]} ↓`,canvas.width-105,canvas.height-24);
}
function renderTable() {
  const a=new Map(pairs(S.current).map(r=>[r[1],r[2]])),b=new Map(pairs(S.focus).map(r=>[r[1],r[2]]));
  const selected=selectedGT();
  $("assignments").innerHTML=S.frame.points.gt.length ? S.frame.points.gt.map(p=>{
    const d1=a.get(p[0]),d2=b.get(p[0]),delta=d1==null||d2==null ? "—" : `${d2-d1>0?"+":""}${(d2-d1).toFixed(2)} µm`;
    return `<tr${selected===p[0]?' class="selected"':""}><td><button data-node="${p[0]}" aria-label="Inspect annotated node ${p[0]}">${p[0]}</button></td><td>${format(d1)}</td><td>${format(d2)}</td><td>${delta}</td></tr>`;
  }).join("") : '<tr><td colspan="4">No sparse annotations in this frame. Predictions and masks remain inspectable.</td></tr>';
  $("assignments").querySelectorAll("button").forEach(button=>button.addEventListener("click",()=>{stop();S.activeError=null;S.selection={method:"gt",id:Number(button.dataset.node)};centerSelection();}));
}
function renderInspector() {
  const p=selectedPoint(),g=selectedGT(),mask=selectedMask();
  $("center-selection").disabled=!p;
  if(!p) {$("inspector").innerHTML='<p>Select an annotation, predicted center, or mask.</p><p>White diamonds are sparse labels. Circles and crosses use rounded native-voxel centers.</p>';return;}
  const label=S.selection.method==="gt"?"Annotated node":S.selection.method.startsWith("focus")?"FOCUS instance":"Current prediction";
  const p1=g==null?null:pairs(S.current).find(r=>r[1]===g),p2=g==null?null:pairs(S.focus).find(r=>r[1]===g);
  let html=`<p class="selection-title">${label} ${p[0]} · frame ${S.frame.t}</p><dl><dt>Submitted Z, Y, X</dt><dd>${p.slice(1,4).join(", ")} voxels</dd><dt>Exact center Z, Y, X</dt><dd>${p.slice(4,7).map(n=>n.toFixed(3)).join(", ")}</dd>`;
  if(g!==null) html+=`<dt>Annotated node</dt><dd>${g}</dd><dt>Current match</dt><dd>${p1?`ID ${p1[0]} · ${format(p1[2])}`:format(null)}</dd><dt>FOCUS match</dt><dd>${p2?`mask ${p2[0]} · ${format(p2[2])}`:format(null)}</dd>`;
  else html+='<dt>Annotation match</dt><dd>No assigned annotation. This does not establish a false positive.</dd>';
  if(p1)html+=`<dt>Current exact Z, Y, X</dt><dd>${point(S.current,p1[0]).slice(4,7).map(v=>v.toFixed(3)).join(", ")} voxels</dd>`;
  if(S.index.prediction_reviews?.[S.current] && g!==null) {
    const nearest=S.frame.model_nearest?.[S.current]?.find(r=>r[0]===g);
    if(nearest)html+=`<dt>Nearest ${escaped(S.index.prediction_reviews[S.current].name)} center</dt><dd>ID ${nearest[1]} · ${format(nearest[2])}${p1?'':' · not an assigned match'}</dd>`;
  }
  if(p2)html+=`<dt>FOCUS exact Z, Y, X</dt><dd>${point(S.focus,p2[0]).slice(4,7).map(v=>v.toFixed(3)).join(", ")} voxels</dd>`;
  if(mask) {
    const members=S.frame.members[mask],physicalVolume=S.frame.sizes[mask]*S.seq.spacing.reduce((a,b)=>a*b,1);
    html+=`<dt>FOCUS instance</dt><dd>${mask} · ${S.frame.sizes[mask]} voxels · ${physicalVolume.toFixed(1)} µm³</dd><dt>Centers inside mask</dt><dd>${members.pooled?.length??0} C4_m6 · ${members.best?.length??0} P0 · ${members.detector.length} detector · ${members.gt.length} annotated</dd>`;
    if(members.gt.length>1) html+=`<dt>Inspect possible merge</dt><dd>${members.gt.length} annotated centers lie inside this instance.</dd>`;
  }
  html+='</dl>';$("inspector").innerHTML=html;
}
function render() {
  if(!S.frame || !S.volume) return;
  if($("comparison-view").open){const texture=textures();canvases.forEach((c,i)=>drawPanel(c,["raw","current","focus"][i],texture));}
  const n=S.frame.points.gt.length,a=pairs(S.current).length,b=pairs(S.focus).length;
  $("gt-count").textContent=`${n} annotations`;
  $("current-title").textContent=S.index.methods[S.current];
  $("current-count").textContent=`${S.frame.points[S.current].length} centers`;
  $("focus-count").textContent=`${S.frame.points[S.focus].length} centers`;
  $("current-foot").textContent=S.mode==="tracks"?"Saved tracker edges":`${a}/${n} annotated matches`;
  $("focus-foot").textContent=S.mode==="tracks"?"Dashed = preview association":`${b}/${n} annotated matches`;
  $("raw-foot").textContent=S.mode==="tracks"?"Saved annotation edges":"Native microscopy";
  $("current-recall").textContent=n?`${(a/n*100).toFixed(1)}%`:"n/a";
  $("focus-recall").textContent=n?`${(b/n*100).toFixed(1)}%`:"n/a";
  $("mask-count").textContent=Object.keys(S.frame.sizes).length;
  const p=projection(),axis="ZYX"[p.d];
  $("view-label").textContent=`${S.plane.toUpperCase()} · ${S.projection==="mip"?"full-depth maximum":`${axis} ${p.low}–${p.high}`} · ${S.seq.spacing.join(" × ")} µm/voxel (ZYX)`;
  $("zoom-label").textContent=`${S.zoom.toFixed(1)}×`;
  $("mode-note").textContent=S.mode==="tracks" ? `Trails show up to four preceding frames. White: annotation graph. Cyan: ${S.index.prediction_reviews?.[S.current]?`${S.index.prediction_reviews[S.current].name} links confirmed by annotations; red: incorrect links; grey: unevaluable links`:"saved v3 tracker"}. Dashed orange: FOCUS preview association (7 µm, one-to-one, no division model). The focused error comparison above shows each timepoint separately.` : `Assignments are optimal one-to-one 3D matches within ${S.radius} µm, recomputed separately for each method and threshold. Red diamonds are unmatched annotations. Lines appear only when both endpoints lie in the displayed depth range.`;
  renderTable();renderInspector();renderErrors();writeHash();
}
function scopedCases() {return allCases().filter(e=>S.errorScope==="all"||(e.dataset===S.seq.dataset&&S.seq.times.includes(e.t)));}
function reviewEvents() {
  const order=['link_missing','link_wrong','link_detection','center_missing','center_conflict','division_missing','division_extra','center_offset'];
  return scopedCases().filter(e=>S.errorKind==="all"||e.category===S.errorKind)
    .sort((a,b)=>order.indexOf(a.category)-order.indexOf(b.category)||a.dataset.localeCompare(b.dataset)||a.t-b.t||a.id.localeCompare(b.id));
}
function setCategory(category) {
  caseRequest++;stop();S.errorKind=category;S.activeError=null;S.selection=null;
  const first=reviewEvents()[0];if(first)inspectError(first);else render();
}
function renderErrors() {
  const review=currentReview();if(!review)return;
  const metrics=review.datasets[S.seq.dataset],events=reviewEvents(),active=selectedError(),scoped=scopedCases();
  $("review-title").textContent=`${review.name} · understand a failure`;
  $("error-model").value=S.reviewModel;
  $("best-scope").textContent=`${S.index.frame_keys.length} exported image frames · ${Object.keys(review.datasets).length} clips`;
  $("best-overall").textContent=review.full_evaluation?`${review.name} · total score ${review.full_evaluation.score.toFixed(6)} on all ${review.full_evaluation.n} evaluation clips. ${review.selection_note}`:'';
  $("best-metrics").textContent=`${S.seq.dataset} · complete ${metrics.frames_total}-frame clip: adjusted edge Jaccard ${metrics.adj_edge_jaccard.toFixed(5)} · links ${metrics.edge_tp} correct / ${metrics.edge_fp} incorrect / ${metrics.edge_fn} missed · divisions ${metrics.division_tp} correct / ${metrics.division_fp} incorrect / ${metrics.division_fn} missed.`;
  $("error-scope").options[1].textContent=`All ${S.index.frame_keys.length} exported frames`;
  $("error-kind").value=S.errorKind;$("error-scope").value=S.errorScope;
  const position=events.findIndex(e=>e.id===active?.id);
  $("error-count").textContent=position<0?`${events.length} cases`:`Case ${position+1} of ${events.length}`;
  $("error-previous").disabled=$("error-next").disabled=!events.length;
  const counts={};for(const e of scoped)counts[e.category]=(counts[e.category]||0)+1;
  const groupTitles={centers:'1 · Find the cell center',links:'2 · Connect cells across frames',divisions:'3 · Recover the division'};
  const token=JSON.stringify([S.reviewModel,S.seq.id,S.radius,S.errorKind,S.errorScope,S.activeError]);
  if($("error-categories").dataset.token!==token) {
    $("error-categories").dataset.token=token;
    $("error-categories").innerHTML=Object.entries(groupTitles).map(([group,title])=>`<article class="category-group ${group}"><h3>${title}</h3>${Object.entries(review.categories).filter(([,c])=>c.group===group).map(([id,c])=>`<button data-category="${id}" aria-pressed="${S.errorKind===id}" title="${escaped(c.description)}"><span>${escaped(c.short)}</span><strong>${counts[id]||0}</strong></button>`).join('')}</article>`).join('');
    $("error-categories").querySelectorAll('button').forEach(b=>b.onclick=()=>setCategory(b.dataset.category));
  }
  if($("error-list").dataset.token!==token) {
    $("error-list").dataset.token=token;
    $("error-list").innerHTML=events.length?events.map((e,i)=>`<button class="error-item ${review.categories[e.category].group}${e.id===active?.id?' selected':''}" data-error="${escaped(e.id)}" aria-pressed="${e.id===active?.id}"><span class="case-number">${i+1}</span><span>${escaped(e.title)}${e.distance_um==null?'':` · ${e.distance_um.toFixed(2)} µm`}<small>${e.category.startsWith('center_')?`Frame ${e.t}`:`Frames ${e.scene.times.join(' → ')}`} · ${escaped(e.dataset)}</small></span></button>`).join(''):'<p class="muted">No cases of this type in these exported images. Try another category, radius or sequence.</p>';
    $("error-list").querySelectorAll("button").forEach(b=>b.onclick=()=>inspectError(events.find(e=>e.id===b.dataset.error)));
  }
  if(!active) {
    $("error-detail").innerHTML='<div class="case-empty"><h3>Choose the failure you want to understand</h3><p>Start with “Missing connection” to see cells that were detected but not linked. “Missing center” inspects detection in one frame.</p></div>';
    window.errorCaseView.clear();return;
  }
  const group=review.categories[active.category].group,center=group==='centers';
  const cells=center?active.scene.cells.slice(0,1):active.scene.cells.filter(c=>active.scene.expected.some(e=>e.source===c.annotation||e.target===c.annotation));
  const status=c=>`<div class="case-step ${c.prediction?'found':'failed'}"><span>Cell ${c.label} · frame ${c.t}</span><strong>${c.prediction?'✓ Center matched':'× No matched center'}</strong><small>${c.prediction?`${c.distance_um.toFixed(2)} µm from annotation`:`No assignment within ${active.radius_um} µm`}</small></div>`;
  const focus=active.scene.focus_connection, names=new Map(active.scene.points.map(p=>[p.key,p.label]));
  const missingPair=focus?`${names.get(focus.source)} → ${names.get(focus.target)} missing`:'Expected edge absent';
  const delta=active.scene.center_probe?.delta_um;
  const displacement=delta?`<p class="case-displacement">${active.scene.center_probe.assigned?'Matched':'Nearest'} center − annotation: ${delta.map((v,i)=>`<span>Δ${'ZYX'[i]} ${v>=0?'+':''}${v.toFixed(2)} µm</span>`).join(' · ')}. Distance uses all three axes.</p>`:'';
  const verdict=center?(active.category==='center_offset'?`<div class="case-step failed"><span>Center position</span><strong>${active.distance_um.toFixed(2)} µm offset</strong><small>Above the 3 µm diagnostic threshold</small></div>`:''):`<div class="case-step failed"><span>${group==='divisions'?'Division':'Connection'}</span><strong>× ${active.category==='link_detection'?'Endpoint missing':active.category==='link_missing'?missingPair:active.category==='link_wrong'?'Incorrect edge':active.category==='division_extra'?'Split rejected':'Split not recovered'}</strong><small>${active.category==='link_detection'?'Inspect center detection first':'See expected and actual below'}</small></div>`;
  $("error-detail").innerHTML=`<div class="case-heading ${group}"><p class="case-stage">${center?'CENTER DETECTION · SAME FRAME':group==='links'?'TEMPORAL LINKING · BETWEEN FRAMES':'CELL DIVISION'}</p><h3>${escaped(active.title)}</h3><p>${escaped(active.explanation)}</p></div><div class="case-steps">${cells.map(status).join('')}${verdict}</div>${displacement}<details class="case-identifiers"><summary>Cell IDs and related scoring flags</summary><p>${cells.map(c=>`${c.label}: annotation ${c.annotation.slice(2)}${c.prediction?`, prediction ${c.prediction.slice(2)}`:', no assigned prediction'}`).map(escaped).join(' · ')}</p><p>${active.related_flags.map(escaped).join(' · ')}</p></details>`;
  window.errorCaseView.update({event:{...active,pipeline:S.reviewModel},shape:S.seq.shape,spacing:S.seq.spacing,plane:S.plane,gamma:S.gamma,
    local:$("case-projection").value==='local',isolate:$("case-isolate").checked,
    available:t=>S.index.frame_keys.includes(keyFor({dataset:active.dataset},t)),
    load:async t=>{
      const key=keyFor({dataset:active.dataset},t),frame=S.seq.dataset===active.dataset&&S.frames.has(t)?S.frames.get(t):await json(`frames/${key}.json`);
      return {key,frame,volume:await volume(key,frame.shape)};
    }});
}
async function inspectError(event,time=event?.t) {
  if(!event)return;stop();const request=++caseRequest;
  const tracking=!event.category.startsWith('center_'),method=S.reviewModel;
  const displacement=event.scene.center_probe?.delta_um?.map(Math.abs);
  S.plane=displacement&&displacement[0]>Math.max(displacement[1],displacement[2])?(displacement[2]>=displacement[1]?'xz':'yz'):'xy';
  S.projectionCache=null;
  S.current=method;previousMethod=method;S.mode=tracking?"tracks":"matches";$("mode").value=S.mode;
  if(tracking){S.radius="7";$("radius").value="7";}
  for(const control of ['centers','gt','lines'])$(control).checked=true;
  $("comparison-view").open=false;
  if(S.seq.dataset!==event.dataset||!S.seq.times.includes(time)) {
    const seq=S.index.sequences.find(s=>s.dataset===event.dataset&&s.times.includes(time));
    if(!seq)return;
    await selectSequence(seq.id,time,true);
  } else if(S.frame.t!==time) await showTime(S.seq.times.indexOf(time),true);
  if(request!==caseRequest||S.busy||S.frame.t!==time||S.seq.dataset!==event.dataset||S.reviewModel!==method)return;
  S.activeError=event.id;S.selection={method:event.method,id:event.node};
  if(!selectedPoint())S.selection=null;
  centerSelection();updateControls();render();
}
function centerSelection() {
  const p=selectedPoint();if(!p) return;
  const [h,v,d]=axes();S.depth=p[d+1];S.cx=(p[h+1]+.5)/S.seq.shape[h];S.cy=(p[v+1]+.5)/S.seq.shape[v];
  S.zoom=3;if(S.projection==="mip")S.projection="slab";S.projectionCache=null;updateControls();render();
}
function pointerPoint(event,canvas) {
  const rect=canvas.getBoundingClientRect(),tr=transform(canvas);
  return {x:(event.clientX-rect.left)*canvas.width/rect.width,y:(event.clientY-rect.top)*canvas.height/rect.height,tr};
}
function pick(event,canvas,kind) {
  stop();const loc=pointerPoint(event,canvas),method=kind==="raw"?"gt":kind==="current"?S.current:S.focus;
  const candidates=[];
  if($("gt").checked) candidates.push(...S.frame.points.gt.map(row=>({method:"gt",row})));
  if(kind!=="raw" && $("centers").checked) candidates.push(...S.frame.points[method].map(row=>({method,row})));
  let best=null,dist=18;
  for(const candidate of candidates) {
    if(!visible(candidate.row))continue;const [x,y]=xy(candidate.row,loc.tr),distance=Math.hypot(x-loc.x,y-loc.y);
    if(distance<dist){dist=distance;best={method:candidate.method,id:candidate.row[0]};}
  }
  if(!best && kind==="focus" && $("masks").checked) {
    const p=projection(),x=Math.floor((loc.x-loc.tr.ox)/(S.seq.spacing[p.h]*loc.tr.scale)),y=Math.floor((loc.y-loc.tr.oy)/(S.seq.spacing[p.v]*loc.tr.scale));
    if(x>=0&&y>=0&&x<p.width&&y<p.height){const label=p.labels[y*p.width+x];if(label)best={method:S.focus,id:label};}
  }
  S.selection=best;S.activeError=null;render();
}
canvases.forEach((canvas,index)=>{
  let drag=null;
  canvas.addEventListener("pointerdown",e=>{if(!S.frame||S.busy)return;canvas.setPointerCapture(e.pointerId);drag={x:e.clientX,y:e.clientY,cx:S.cx,cy:S.cy,moved:false};});
  canvas.addEventListener("pointermove",e=>{
    if(!drag)return;const rect=canvas.getBoundingClientRect(),tr=transform(canvas),dx=(e.clientX-drag.x)*canvas.width/rect.width,dy=(e.clientY-drag.y)*canvas.height/rect.height;
    if(Math.hypot(dx,dy)>4)drag.moved=true;
    if(drag.moved){S.cx=Math.max(0,Math.min(1,drag.cx-dx/(tr.pw*tr.scale)));S.cy=Math.max(0,Math.min(1,drag.cy-dy/(tr.ph*tr.scale)));render();}
  });
  canvas.addEventListener("pointerup",e=>{if(drag&&!drag.moved)pick(e,canvas,["raw","current","focus"][index]);drag=null;});
  canvas.addEventListener("pointercancel",()=>{drag=null;});
  canvas.addEventListener("wheel",e=>{
    if(!S.frame||S.busy)return;e.preventDefault();const loc=pointerPoint(e,canvas),tr=loc.tr;
    const u=(loc.x-tr.ox)/(tr.pw*tr.scale),v=(loc.y-tr.oy)/(tr.ph*tr.scale),next=Math.max(1,Math.min(10,S.zoom*Math.exp(-e.deltaY*.0015)));
    const ratio=S.zoom/next;S.cx=Math.max(0,Math.min(1,u+(S.cx-u)*ratio));S.cy=Math.max(0,Math.min(1,v+(S.cy-v)*ratio));S.zoom=next;render();
  },{passive:false});
});
function writeHash() {
  if($("comparison-panel").hidden)return;
  if(['detection','report','tracking'].includes(new URLSearchParams(location.hash.slice(1)).get('tab')))return;
  if(!S.seq)return;
  const query=new URLSearchParams({tab:'comparison',sequence:S.seq.id,t:S.frame.t,plane:S.plane,projection:S.projection,depth:S.depth,current:S.current,focus:S.focus,radius:S.radius,mode:S.mode});
  for(const k of ["zoom","cx","cy","gamma","opacity"])query.set(k,String(S[k]));
  for(const k of ["masks","centers","gt","lines"])query.set(k,$(k).checked?"1":"0");
  if(S.selection)query.set("selection",`${S.selection.method}:${S.selection.id}`);
  query.set('reviewModel',S.reviewModel);query.set('errorKind',S.errorKind);query.set('errorScope',S.errorScope);
  if(S.activeError)query.set('error',S.activeError);
  history.replaceState(null,"",`#${query}`);
}
$("comparison-view").ontoggle=()=>{if(S.frame)render();};
$("case-plane").onchange=()=>{$("plane").value=$("case-plane").value;$("plane").onchange();};
$("case-brightness").oninput=()=>{S.gamma=Number($("case-brightness").value);$("gamma").value=S.gamma;updateControls();render();};
$("case-isolate").onchange=render;$("case-projection").onchange=render;
$("case-clear").onclick=()=>{caseRequest++;S.activeError=null;S.selection=null;$("comparison-view").open=true;render();$("comparison-view").scrollIntoView({block:'start'});};
$("play").onclick=play;
$("previous").onclick=()=>{caseRequest++;stop();showTime(S.position-1);};$("next").onclick=()=>{caseRequest++;stop();showTime(S.position+1);};
$("time").oninput=()=>{caseRequest++;stop();showTime(Number($("time").value));};
$("sequence").onchange=()=>selectSequence($("sequence").value);
$("mode").onchange=()=>{
  S.mode=$("mode").value;S.selection=null;S.activeError=null;$("comparison-view").open=true;
  if(S.mode==="tracks"){previousMethod=S.current;if(!['best','pooled','selected'].includes(S.current))S.current="best";}else S.current=previousMethod;
  updateControls();render();
};
for(const id of ["current","focus","radius"]){$(id).onchange=()=>{caseRequest++;S[id]=$(id).value;if(id==="current"){previousMethod=S.current;if(S.index.prediction_reviews?.[S.current])S.reviewModel=S.current;}S.selection=null;S.activeError=null;render();};}
$("error-model").onchange=()=>{caseRequest++;S.reviewModel=$("error-model").value;S.current=S.reviewModel;previousMethod=S.current;S.activeError=null;S.selection=null;updateControls();render();};
$("error-kind").onchange=()=>setCategory($("error-kind").value);
$("error-scope").onchange=()=>{caseRequest++;S.errorScope=$("error-scope").value;S.activeError=null;render();};
for(const [id,delta] of [['error-previous',-1],['error-next',1]])$(id).onclick=()=>{
  const events=reviewEvents();if(!events.length)return;
  const current=events.findIndex(e=>e.id===S.activeError);
  inspectError(events[current<0?(delta>0?0:events.length-1):(current+delta+events.length)%events.length]);
};
$("plane").onchange=()=>{S.plane=$("plane").value;S.depth=Math.floor(S.seq.shape[axes()[2]]/2);S.cx=.5;S.cy=.5;S.zoom=1;S.projectionCache=null;updateControls();render();};
$("projection").onchange=()=>{S.projection=$("projection").value;S.projectionCache=null;updateControls();render();};
$("depth").oninput=()=>{S.depth=Number($("depth").value);S.projectionCache=null;updateControls();render();};
for(const id of ["gamma","opacity"]){$(id).oninput=()=>{S[id]=Number($(id).value);updateControls();render();};}
for(const id of ["masks","centers","gt","lines"]){$(id).onchange=render;}
$("reset").onclick=()=>{S.zoom=1;S.cx=.5;S.cy=.5;render();};$("center-selection").onclick=centerSelection;
$("copy-view").onclick=async()=>{try{await navigator.clipboard.writeText(location.href);$("copy-status").textContent="View link copied";}catch{$("copy-status").textContent="Copy the address from your browser; it contains this frame and controls.";}};
document.addEventListener("keydown",event=>{
  if($("comparison-panel").hidden)return;
  if(["INPUT","SELECT","TEXTAREA","BUTTON","SUMMARY"].includes(document.activeElement?.tagName)||!S.frame)return;
  if(event.code==="Space"){event.preventDefault();play();}
  if(event.key==="ArrowRight"||event.key==="ArrowLeft"){event.preventDefault();stop();showTime(S.position+(event.key==="ArrowRight"?1:-1));}
  if(event.key==="ArrowUp"||event.key==="ArrowDown"){event.preventDefault();S.depth+=event.key==="ArrowUp"?1:-1;if(S.projection==="mip")S.projection="slice";S.projectionCache=null;updateControls();render();}
});
async function init() {
  const params=new URLSearchParams(location.hash.slice(1));
  S.index=await json("index.json");
  $("sequence").innerHTML=S.index.sequences.map(s=>`<option value="${escaped(s.id)}">${escaped(s.title)}</option>`).join("");
  $("provenance").innerHTML=`<p>${escaped(S.index.limitations)}</p><p><b>Center matching.</b> ${escaped(S.index.matching)} All overlays use submitted integer centers; exact mask-derived coordinates appear in the inspector. Table errors and recall cover the full frame regardless of the current slice. This is not the competition tracking score.</p><p><b>Mask rendering.</b> Filled regions and outlines are actual FOCUS instance labels. A projection shows the label at the brightest displayed image voxel along each ray, so overlapping objects can be hidden. Use individual slices to examine boundaries. All three panels use identical brightness, depth, zoom, and physical aspect ratio.</p><p><b>FOCUS center definitions.</b> ${Object.values(S.index.center_rules).map(escaped).join(" ")}</p><p><b>Temporal comparison.</b> ${escaped(S.index.focus_tracking)} C4_m6, P0 and the selected v3 tracker use their complete saved prediction graphs. Their full-clip scoring is distinct from the exported image windows. Neither raw detector proposals nor V5 watershed centroids have a separate tracking result in this viewer. Colors in FOCUS temporal mode follow preview associations, not biological identity.</p><p><b>Selection.</b> ${escaped(S.index.selection)} Playback speed is frames per second of the display, not a biological frame interval. Snapshot pairs are not interpolated.</p><p><b>Keyboard.</b> Space: play/pause. Left/right: frame. Up/down: depth. Focus a canvas or the page before using shortcuts.</p><p><b>Sources.</b> Native competition Zarr and GEFF; cached pre-ILP detections, C4_m6, P0 and selected v3 graphs, V5 regions; frozen FOCUS nuclei checkpoint. Per-frame JSON contains source hashes, inference settings, native mask IDs, and every assignment.</p>`;
  for(const name of ["plane","projection","current","focus","radius","mode"]){
    const value=params.get(name);if(value && Array.from($(name).options).some(o=>o.value===value)){S[name]=value;$(name).value=value;}
  }
  if(S.mode==="tracks"&&!['best','pooled','selected'].includes(S.current))S.current="best";
  previousMethod=S.current;
  for(const [name,id] of [['errorKind','error-kind'],['errorScope','error-scope'],['reviewModel','error-model']]) {
    if(Array.from($(id).options).some(o=>o.value===params.get(name)))S[name]=params.get(name);
  }
  const id=S.index.sequences.some(s=>s.id===params.get("sequence"))?params.get("sequence"):S.index.sequences[0].id;
  await selectSequence(id,params.get("t"));
  if(!S.frame)return;
  if(params.has("depth")&&Number.isFinite(Number(params.get("depth"))))S.depth=Math.round(Number(params.get("depth")));
  for(const [k,low,high] of [["zoom",1,10],["cx",0,1],["cy",0,1],["gamma",.5,2.5],["opacity",0,.8]]) {
    if(params.has(k)&&Number.isFinite(Number(params.get(k))))S[k]=Math.max(low,Math.min(high,Number(params.get(k))));
  }
  for(const k of ["masks","centers","gt","lines"])if(params.has(k))$(k).checked=params.get(k)==="1";
  for(const k of ["gamma","opacity"])$(k).value=S[k];
  if(params.has("selection")) {
    const [method,id]=params.get("selection").split(":");
    if(point(method,Number(id)))S.selection={method,id:Number(id)};
  }
  if(!params.has('reviewModel')&&S.index.prediction_reviews?.[S.current])S.reviewModel=S.current;
  const savedError=allCases().find(e=>e.id===params.get('error')||e.related_flags.includes(params.get('error')));
  if(savedError?.dataset===S.seq.dataset){S.activeError=savedError.id;$("comparison-view").open=false;}
  document.querySelectorAll("#comparison-panel button,#comparison-panel select,#comparison-panel input").forEach(control=>{control.disabled=false;});
  $("play").disabled=!S.seq.continuous;
  updateControls();render();
  window.centerComparison={get state(){return S;},projection,point,officialPairs:pairs,stop};
  if(!params.size&&reviewEvents().length)await inspectError(reviewEvents()[0]);
}
init().catch(fail);
})();

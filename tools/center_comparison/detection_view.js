/* An exhaustive, paginated review over frozen predictions and native image data. */
"use strict";
(() => {
const $=id=>document.getElementById(id), params=new URLSearchParams(location.hash.slice(1));
const D={catalog:null,model:"pooled",filter:"all",dataset:"",pool:"final",offset:0,limit:40,
  frameFilter:"",caseLoading:false,method:"final",targetGt:null,selectedPred:null,ranked:[],
  rows:[],total:0,payload:null,frame:null,volume:null,key:null,plane:"xy",projection:"local",depth:0,
  center:[32,128,128],field:40,gamma:1,playing:false,position:0,caseEpoch:0,frameEpoch:0,listEpoch:0,projectionCache:null};
const canvases=[$("d-final-canvas")];
const fullCanvases=[$("d-final-full")];
const titles={missing:"Missed center",proposal_only:"Detector candidate present; final center missing",conflict:"Center assignment conflict",
  recovered:"Detector miss",offset:"Center offset",crowded:"Ambiguous centers",matched:"Matched center",
  association:"Missing connection",association_ambiguous:"Missing connection · ambiguous centers",wrong_link:"Wrong connection",
  wrong_link_ambiguous:"Wrong connection · ambiguous centers",unlabeled:"Unlabeled prediction"};
const palettes={gt:"#ffd166",pred:"#42e4f5",crop:"#e5edf8",outside:"#ff8b99"};
const volumes=new Map(),pending=new Map();let timer,comparisonHash="",activeTab="comparison",initializing=false;
const storageKey="biohub-detection-assessments-v1";
let assessments={};try{assessments=JSON.parse(localStorage.getItem(storageKey)||"{}");}catch{}
const esc=x=>String(x??"").replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pretty=n=>Number(n).toLocaleString();
const distance=n=>n==null?"No candidate":`${n.toFixed(2)} µm`;
const axes=()=>D.plane==="xy"?[2,1,0]:D.plane==="xz"?[2,0,1]:[1,0,2];
const controlIds=["d-reset","d-jump","d-jump-go","d-model","d-filter","d-clip","d-pool","d-frame-filter","d-page","d-page-go","d-page-prev","d-page-next","d-prev","d-next","d-back","d-play","d-forward","d-time","d-case-frame","d-plane","d-projection","d-depth","d-width","d-brightness","d-centers","d-labels","d-lines","d-recenter","d-verdict","d-notes","d-save","d-copy"];
const caseControls=["d-back","d-play","d-forward","d-time","d-case-frame","d-plane","d-projection","d-depth","d-width","d-brightness","d-centers","d-labels","d-lines","d-recenter","d-verdict","d-notes","d-save","d-copy"];
controlIds.push('d-pred-source','d-gt-choice','d-pred-choice','d-fit-pair');
caseControls.push('d-gt-choice','d-pred-choice','d-fit-pair');
function error(e){stop();D.caseLoading=false;$("d-error").hidden=false;$("d-error").textContent=e.message||String(e);$("d-loading").textContent="Could not load this view";$("d-images").classList.remove("loading");}
async function api(path,query={}) {
  const r=await fetch(`/api/detection/${path}?${new URLSearchParams(query)}`);
  const data=await r.json().catch(()=>({error:"Restart the viewer server to enable the new detection API."}));
  if(!r.ok)throw Error(data.error||`HTTP ${r.status}`);return data;
}
async function getVolume(dataset,t,shape) {
  const key=`${dataset}:${t}`;
  if(volumes.has(key)){const result=volumes.get(key);volumes.delete(key);volumes.set(key,result);return result;}
  if(pending.has(key))return pending.get(key);
  const job=(async()=>{
    const r=await fetch(`/api/detection/frame?${new URLSearchParams({dataset,t})}`);
    if(!r.ok){const e=await r.json();throw Error(e.error);}
    if(!window.DecompressionStream)throw Error("This viewer needs a browser supporting gzip DecompressionStream.");
    const buffer=await new Response(r.body.pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
    if(buffer.byteLength!==shape.reduce((a,b)=>a*b,1))throw Error("The native frame is truncated or has the wrong dimensions.");
    const value=new Uint8Array(buffer);volumes.set(key,value);while(volumes.size>6)volumes.delete(volumes.keys().next().value);return value;
  })();pending.set(key,job);try{return await job;}finally{pending.delete(key);}
}
function reviewed(row){const record=assessments[row.key];return !!record && !!(record.verdict||record.notes);}
function syncHash() {
  if(activeTab!=="detection")return;
  const q=new URLSearchParams({tab:"detection",review:"2",dmodel:D.model,dfilter:D.filter,dclip:D.dataset,dpool:D.pool,doffset:D.offset,
    dplane:D.plane,dprojection:D.projection,ddepth:D.depth,dwidth:D.field,dgamma:D.gamma,dcenter:D.center.join(",")});
  if(D.key)q.set("dcase",D.key);if(D.frame)q.set("dt",D.frame.t);
  if(D.frameFilter!=="")q.set("dframefilter",D.frameFilter);
  q.set('dmethod',D.method);
  if(D.frame){q.set('dgt',D.targetGt??'');q.set('dpred',D.selectedPred??'');}
  history.replaceState(null,"",`#${q}`);
}
function saveAssessment(silent=false) {
  if(!D.payload)return;
  const verdict=$("d-verdict").value,notes=$("d-notes").value;
  const existing=assessments[D.key];
  if(!verdict&&!notes&&!existing)return;
  if(existing?.verdict===verdict&&existing?.notes===notes)return;
  assessments[D.key]={case_key:D.key,dataset:D.payload.case.dataset,pipeline:D.payload.name,
    frame:D.payload.case.t,automatic_group:D.payload.case.kind,verdict,notes,
    source:D.payload.provenance,updated:new Date().toISOString(),view:location.hash};
  try{localStorage.setItem(storageKey,JSON.stringify(assessments));$("d-save-status").textContent="Saved in this browser";}
  catch{$("d-save-status").textContent="Browser storage unavailable. Export assessments to keep your notes.";}
  if(!silent)renderList();
}
function restoreAssessment() {
  const record=assessments[D.key];
  $("d-verdict").value=record?.verdict||"";$("d-notes").value=record?.notes||"";
  $("d-save-status").textContent=record?"Saved assessment loaded":"Not reviewed";
  if(record && JSON.stringify(record.source)!==JSON.stringify(D.payload.provenance))$("d-save-status").textContent="Saved judgment comes from different prediction sources; inspect again.";
}
function renderList() {
  $("d-total").textContent=`${pretty(D.total)} examples`;
  $("d-range").textContent=D.total?`${pretty(D.offset+1)}–${pretty(Math.min(D.total,D.offset+D.rows.length))}`:"";
  $("d-list").innerHTML=D.rows.length?D.rows.map((row,i)=>{
    return `<button role="listitem" class="d-case-item${D.key===row.key?' selected':''}" data-key="${esc(row.key)}"><strong>${pretty(D.offset+i+1)}. ${esc(titles[row.kind])}</strong><small>${esc(row.dataset)} · t${row.t}${reviewed(row)?' · reviewed':''}</small></button>`;
  }).join(""):'<p class="muted">No examples. Use All datasets to reset.</p>';
  $("d-list").querySelectorAll("button").forEach(b=>b.onclick=()=>selectCase(b.dataset.key));
  $("d-page-prev").disabled=D.offset===0;$("d-page-next").disabled=D.offset+D.limit>=D.total;
  $("d-page").max=Math.max(1,Math.ceil(D.total/D.limit));$("d-page").value=Math.floor(D.offset/D.limit)+1;
  const index=D.rows.findIndex(r=>r.key===D.key);
  $("d-progress").textContent=D.total?`${pretty(D.offset+Math.max(index,0)+1)} / ${pretty(D.total)}`:"0 examples";
  $("d-jump").max=Math.max(1,D.total);$("d-jump").value=D.offset+Math.max(index,0)+1;
  const active=$("d-list").querySelector('.selected');if(active){const list=$("d-list"),top=active.offsetTop-list.offsetTop;if(top<list.scrollTop)list.scrollTop=top;else if(top+active.offsetHeight>list.scrollTop+list.clientHeight)list.scrollTop=top+active.offsetHeight-list.clientHeight;}
  $("d-prev").disabled=D.total===0||(index===0&&D.offset===0);$("d-next").disabled=D.total===0||(index===D.rows.length-1&&D.offset+D.rows.length>=D.total);
}
function scopeText() {
  const descriptions={
    all:"All flagged centers and connections, including ambiguous cases. No clip sampling.",
    detection:"Every flagged annotated center: missing proposals, downstream gaps, offsets and identity conflicts. Association cases have their own group.",
    missing:"No detector candidate within 7 µm. Includes cases the final pipeline recovered, so a later rescue does not hide a detector gap.",
    selection:"A detector candidate lies within 7 µm, but no final center does. Selection, movement and incorrect candidate identity remain possible; this does not establish the cause.",
    offset:"Assigned final centers at least 3 µm from their annotation. These are localization candidates, including cases with competing identities.",
    association:"Missing connections whose two endpoints have assigned centers, plus every scored incorrect connection. Uncertain endpoint identity is marked explicitly.",
    ambiguous:"Every labeled center or scored connection flagged for competing candidates, ambiguous identity or a proposal/final discrepancy. Unlabeled predictions are available in the separate unmatched-predictions group.",
    unlabeled:"Every prediction without an official annotation assignment, including frames with no labels. Unknown truth status: sparse annotations do not make these false positives.",
    annotations:"Every annotated center, including matched controls. This lets you inspect unflagged cases too; the automated groups are not an exhaustive biological diagnosis."};
  $("d-scope").textContent=descriptions[D.filter]+" Matching is 3D at 7 µm. Filter groups can overlap.";
  $("d-pool-label").hidden=D.filter!=="unlabeled";
  $("d-unknown-note").hidden=D.filter!=="unlabeled";
  if(D.catalog){
    $("d-coverage").textContent=D.dataset?`${D.dataset}${D.frameFilter!==''?' · frame '+D.frameFilter:''}`:`All ${D.catalog.datasets} clips${D.frameFilter!==''?' · frame '+D.frameFilter:''}`;
    $("d-reset").hidden=!D.dataset&&D.frameFilter==='';
    document.querySelectorAll('#d-groups button').forEach(b=>{b.setAttribute('aria-pressed',String(b.dataset.filter===D.filter&&!D.dataset&&D.frameFilter===''));b.querySelector('b').textContent=pretty(b.dataset.filter==='unlabeled'&&D.pool==='raw'?D.catalog.raw_unmatched:D.catalog.counts[D.model][b.dataset.filter]);});
  }
}
async function loadCases(preferredKey=null,selectionPosition=0,restoreView=false) {
  saveAssessment(true);stop();const epoch=++D.listEpoch;++D.caseEpoch;++D.frameEpoch;
  D.caseLoading=true;caseControls.forEach(id=>$(id).disabled=true);
  $("d-total").textContent="Loading cases…";$("d-error").hidden=true;scopeText();
  try {
    const data=await api("cases",{model:D.model,filter:D.filter,dataset:D.dataset,pool:D.pool,offset:D.offset,limit:D.limit,t:D.frameFilter,key:preferredKey||''});
    if(epoch!==D.listEpoch)return;
    D.rows=data.rows;D.total=data.total;D.offset=data.offset;renderList();syncHash();
    if(preferredKey&&D.rows.some(r=>r.key===preferredKey))await selectCase(preferredKey,restoreView);
    else if(D.rows.length)await selectCase(D.rows[selectionPosition<0?D.rows.length-1:selectionPosition].key);
    else {
      D.payload=null;D.key=null;D.frame=null;D.volume=null;D.caseLoading=false;D.targetGt=null;D.selectedPred=null;D.ranked=[];
      caseControls.forEach(id=>$(id).disabled=true);
      $("d-title").textContent="No cases in this group";$("d-kind").textContent="";$("d-location").textContent="";
      for(const id of ["d-evidence","d-center-status","d-connections","d-neighbors","d-image-caption","d-frame-count","d-zoom-status","d-selection-readout","d-nearby-picks","d-gt-choice","d-pred-choice","d-neighbor-caption","d-picked","d-time-label","d-save-status"])$(id).textContent="";
      $("d-verdict").value="";$("d-notes").value="";
      [...canvases,...fullCanvases].forEach(c=>{const ctx=c.getContext("2d");ctx.fillStyle="#080c12";ctx.fillRect(0,0,c.width,c.height);});
      $("d-images").classList.remove("loading");$("d-loading").textContent="Choose another filter";syncHash();
    }
  }catch(e){if(epoch===D.listEpoch)error(e);}
}
function explain(c) {
  const raw=distance(c.raw_distance),near=distance(c.final_nearest);
  return ({
    missing:`No detector candidate or final center lies within 7 µm of this annotation. Nearest detector candidate: ${raw}; final: ${near}. Inspect for a missed or badly localized cell. This is a detection candidate, not an automatic biological verdict.`,
    proposal_only:`A detector candidate is ${raw} away, but the closest final center is ${near}. A center was proposed nearby; removal, displacement or incorrect proposal identity could explain the final gap.`,
    conflict:`The nearest final center is ${near} away, yet optimal one-to-one matching leaves this annotation unmatched. Check which annotation received that center and whether two cells were merged or confused.`,
    recovered:`The detector has no proposal within 7 µm, but the final pipeline has an assigned center ${distance(c.distance)} away. Inspect the detector candidate gap and the later center separately.`,
    offset:`The assigned final center is ${distance(c.distance)} from the annotation. The 3 µm flag is a localization diagnostic. Nearby competing centers or an imperfect annotation can also explain the displacement.`,
    crowded:"Multiple nearby centers or annotations, or a non-nearest assignment, make identity uncertain. Inspect all surrounding proposals and the assignment table; a match alone does not establish that the correct cell was found.",
    matched:`The final center is assigned ${distance(c.distance)} from the annotation. This is an unflagged control available for your own inspection.`,
    association:"Both annotated endpoints have assigned final centers, but the expected connection is absent. That is evidence for an association problem, conditional on those center identities being correct.",
    association_ambiguous:"Both endpoints have assigned centers, but the expected link is absent and at least one center is displaced or has competing identities. Detection/localization and association cannot be cleanly separated automatically.",
    wrong_link:"The official scorer counts this connection as incorrect. Inspect the source and target in their own frames; assigned endpoints do not prove correct biological identity.",
    wrong_link_ambiguous:"This connection is scored incorrect, and an endpoint is unassigned, displaced, or has competing identities. Inspect both detection evidence and the chosen connection.",
    unlabeled:"This prediction has no ground-truth assignment. It may be a real unlabeled cell, a duplicate, a displaced center, or a false detection. The sparse annotations cannot distinguish these cases; inspect it yourself."
  })[c.kind];
}
function renderCase() {
  const p=D.payload,c=p.case;
  $("d-kind").textContent=(c.kind==='unlabeled'?'Unknown':c.entity==='gt'?'Detection':'Association')+(c.ambiguous&&c.kind!=='unlabeled'?' · ambiguous':'');
  $("d-title").textContent=titles[c.kind]+(c.kind==='offset'?` · ${distance(c.distance)}`:'');
  const caseGT=p.frames.find(f=>f.t===c.t)?.points.gt.findIndex(r=>r[0]===c.node)??-1;
  $("d-location").textContent=`${c.dataset} · frame ${c.t}${c.entity==='gt'&&caseGT>=0?' · case GT'+(caseGT+1):''}`;
  $("d-evidence").textContent=explain(c);
  $("d-center-status").innerHTML=c.entity==='gt'?
    `<span>Detector candidates within 7 µm <b>${c.raw_count}</b></span><span>Final centers within 7 µm <b>${c.final_count}</b></span><span>Detector assignment <b>${c.raw_matched?'Present':'Absent'}</b></span><span>Final assignment <b>${c.matched?'Present':'Absent'}</b></span>${c.ambiguous?'<span class="ambiguous-flag">Identity / cause remains ambiguous</span>':''}`:
    `<span>${c.kind==='unlabeled'?'No annotation assignment · truth status unknown':'Inspect expected and actual connections below. Each image displays its own timepoint.'}</span>`;
  const expected=p.expected.map(e=>`<button data-time="${e.source[1]}">GT ${e.source[0]} · t${e.source[1]}</button> → <button data-time="${e.target[1]}">GT ${e.target[0]} · t${e.target[1]}</button> <strong class="${e.recovered?'correct':'miss'}">${e.recovered?'Recovered':'Missing'}</strong>`);
  const actual=p.actual.map(e=>`<button data-time="${e.source[1]}">P ${e.source[0]} · t${e.source[1]}</button> → <button data-time="${e.target[1]}">P ${e.target[0]} · t${e.target[1]}</button> <strong class="${e.status===1?'correct':e.status===-1?'miss':''}">${e.status===1?'Scored correct':e.status===-1?'Scored incorrect':'Unevaluable'}</strong>`);
  $("d-connections").innerHTML=(expected.length||actual.length)?`<details><summary>Connections · ${expected.filter(e=>e.includes('>Missing<')).length} missing</summary><div class="d-edge-columns"><div><h4>Ground truth</h4>${expected.length?expected.map(x=>`<p>${x}</p>`).join(''):'<p>None</p>'}</div><div><h4>Predicted</h4>${actual.length?actual.map(x=>`<p>${x}</p>`).join(''):'<p>None</p>'}</div></div></details>`:"";
  $("d-connections").querySelectorAll("button").forEach(b=>b.onclick=()=>{stop();showFrame(p.frames.findIndex(f=>f.t===Number(b.dataset.time)));});
  renderList();restoreAssessment();
}
async function selectCase(key,restore=false) {
  saveAssessment(true);stop();const epoch=++D.caseEpoch;++D.frameEpoch;
  D.caseLoading=true;caseControls.forEach(id=>$(id).disabled=true);
  $("d-loading").textContent="Loading case and native image…";$("d-images").classList.add("loading");$("d-error").hidden=true;
  try {
    const payload=await api("case",{key});
    if(epoch!==D.caseEpoch)return;
    const defaultPosition=payload.frames.findIndex(f=>f.t===payload.case.t);
    const requested=restore&&params.has('dt')?payload.frames.findIndex(f=>f.t===Number(params.get('dt'))):-1;
    const position=requested>=0?requested:defaultPosition;
    const gray=await getVolume(payload.case.dataset,payload.frames[position].t,payload.clip.shape.slice(1));
    if(epoch!==D.caseEpoch)return;
    ++D.frameEpoch;D.caseLoading=false;
    D.payload=payload;D.key=key;D.frame=payload.frames[position];D.position=position;D.volume=gray;
    D.model=payload.case.model;$("d-model").value=D.model;
    if(payload.case.entity==='raw_prediction')D.method='raw';
    if(payload.case.entity==='prediction')D.method='final';
    autoSelect();
    D.center=[...payload.anchor];D.field=40;D.projection="local";D.plane="xy";
    const probe=payload.neighbors.final.find(n=>n.assigned_gt===payload.case.node)||payload.neighbors.final[0];
    if(probe&&payload.case.entity==='gt'){
      const delta=probe.zyx.map((x,i)=>Math.abs((x-payload.anchor[i])*payload.clip.spacing[i]));
      if(delta[0]>Math.max(delta[1],delta[2]))D.plane=delta[2]>=delta[1]?"xz":"yz";
    }
    D.depth=Math.round(D.center[axes()[2]]);
    fitSelection();
    if(restore){
      for(const [name,id,q] of [['plane','d-plane','dplane'],['projection','d-projection','dprojection']]){
        const v=params.get(q);if([...$(id).options].some(o=>o.value===v))D[name]=v;
      }
      for(const [k,q,low,high] of [['depth','ddepth',0,255],['field','dwidth',12,150],['gamma','dgamma',.5,2.5]])if(params.has(q)&&Number.isFinite(Number(params.get(q))))D[k]=Math.max(low,Math.min(high,Number(params.get(q))));
      const center=(params.get('dcenter')||'').split(',').map(Number);if(center.length===3&&center.every(Number.isFinite))D.center=center;
      if(params.has('dgt')){
        const id=params.get('dgt')===''?null:Number(params.get('dgt'));
        if(id===null||D.frame.points.gt.some(r=>r[0]===id)){D.targetGt=id;rankPredictions();}
      }
      if(params.has('dpred')&&D.frame.points[D.method].some(r=>r[0]===Number(params.get('dpred'))))D.selectedPred=Number(params.get('dpred'));
    }
    D.projectionCache=null;D.fullProjectionCache=null;$("d-images").classList.remove("loading");$("d-loading").textContent="";
    caseControls.forEach(id=>$(id).disabled=false);
    renderCase();render();prefetch();
  }catch(e){if(epoch===D.caseEpoch)error(e);}
}
function prefetch(){const p=D.payload,f=p?.frames[D.position+1];if(f)getVolume(p.case.dataset,f.t,p.clip.shape.slice(1)).catch(()=>{});}
function stop(){D.playing=false;clearTimeout(timer);$("d-play").textContent="Play";}
async function showFrame(position) {
  if(D.caseLoading||!D.payload||position<0||position>=D.payload.frames.length)return;
  const epoch=++D.frameEpoch,caseEpoch=D.caseEpoch,p=D.payload,f=p.frames[position];
  $("d-loading").textContent=`Loading t${f.t}…`;$("d-images").classList.add("loading");
  try{
    const gray=await getVolume(p.case.dataset,f.t,p.clip.shape.slice(1));
    if(epoch!==D.frameEpoch||caseEpoch!==D.caseEpoch)return;
    D.frame=f;D.volume=gray;D.position=position;D.projectionCache=null;D.fullProjectionCache=null;autoSelect();
    $("d-images").classList.remove("loading");$("d-loading").textContent="";render();prefetch();
  }catch(e){if(epoch===D.frameEpoch&&caseEpoch===D.caseEpoch)error(e);}
}
function continuous(){return D.payload?.frames.every((f,i,a)=>i===0||f.t===a[i-1].t+1);}
async function tick(){if(!D.playing)return;await showFrame((D.position+1)%D.payload.frames.length);if(D.playing)timer=setTimeout(tick,500);}
function play(){if(D.playing)return stop();if(D.caseLoading||!D.payload||!continuous())return;D.playing=true;$("d-play").textContent="Pause";timer=setTimeout(tick,500);}
function targetRow(){return D.frame?.points.gt.find(r=>r[0]===D.targetGt)||null;}
function predictionRow(){return D.frame?.points[D.method].find(r=>r[0]===D.selectedPred)||null;}
function centerComparison(gt,pred,spacing,assignedGT=null){
  const distance=gt&&pred?Math.hypot(...gt.slice(2).map((v,i)=>(v-pred[i+2])*spacing[i])):null;
  return {distance,inside:distance!==null&&distance<=7,assignedGT,
    assignment:assignedGT===null?'none':gt&&assignedGT===gt[0]?'this':'other'};
}
function spatialDistance(a,b){return centerComparison(a,b,D.payload.clip.spacing).distance;}
function rankPredictions(){
  const gt=targetRow(),anchor=gt||[0,D.frame.t,...D.payload.anchor];
  D.ranked=D.frame.points[D.method].map(row=>({row,d:spatialDistance(row,anchor)})).sort((a,b)=>a.d-b.d||a.row[0]-b.row[0]);
}
function autoSelect(){
  const frame=D.frame,p=D.payload;
  const related=[...p.focus.gt,...p.expected.flatMap(e=>[e.source[0],e.target[0]])];
  const candidates=frame.points.gt.filter(r=>related.includes(r[0]));
  const gt=candidates.find(r=>p.focus.gt.includes(r[0]))||candidates[0];
  D.targetGt=gt?.[0]??null;
  rankPredictions();
  const assigned=frame.matches[D.method].find(r=>r[1]===D.targetGt);
  const primary=p.case.entity==='raw_prediction'&&D.method==='raw'||['prediction','pred_edge'].includes(p.case.entity)&&D.method==='final';
  const eventPrediction=primary?frame.points[D.method].find(r=>r[0]===(p.case.other??p.case.node))||frame.points[D.method].find(r=>r[0]===p.case.node):null;
  D.selectedPred=eventPrediction?.[0]??assigned?.[0]??D.ranked[0]?.row[0]??null;
  D.selectionUiKey=null;
}
function pairInfo(){
  const gt=targetRow(),pred=predictionRow(),match=D.frame?.matches[D.method].find(r=>r[0]===D.selectedPred);
  return {gt,pred,...centerComparison(gt,pred,D.payload.clip.spacing,match?.[1]??null),
    assignedLabel:match?'GT'+(D.frame.points.gt.findIndex(r=>r[0]===match[1])+1):null,
    gtLabel:gt?'GT'+(D.frame.points.gt.findIndex(r=>r[0]===gt[0])+1):'GT',
    predLabel:pred?'P'+(D.ranked.findIndex(r=>r.row[0]===pred[0])+1):'Prediction'};
}
function fitSelection(){
  const gt=targetRow(),pred=predictionRow(),match=D.frame.matches[D.method].find(r=>r[0]===D.selectedPred);
  const assigned=D.frame.points.gt.find(r=>r[0]===match?.[1]),rows=[gt,pred,assigned].filter(Boolean);
  if(!rows.length)return;
  const [h,v,d]=axes(),spacing=D.payload.clip.spacing;
  // Keep the complete 7 µm GT circle and the selected prediction in the crop.
  const bounds=[h,v].map(axis=>{const values=rows.map(r=>r[axis+2]*spacing[axis]);if(gt)values.push(gt[axis+2]*spacing[axis]-9,gt[axis+2]*spacing[axis]+9);return [Math.min(...values),Math.max(...values)];});
  D.center[h]=(bounds[0][0]+bounds[0][1])/2/spacing[h];D.center[v]=(bounds[1][0]+bounds[1][1])/2/spacing[v];
  D.depth=Math.round((gt||pred)[d+2]);D.center[d]=D.depth;
  D.field=Math.max(40,Math.min(150,Math.max(...bounds.map(b=>b[1]-b[0]))+8));
  D.projection='local';D.projectionCache=null;
}
function choosePrediction(id){
  if(D.caseLoading||!D.frame)return;
  const row=D.frame.points[D.method].find(r=>r[0]===id);if(!row)return;
  stop();D.selectedPred=id;D.selectionUiKey=null;fitSelection();render();
}
function chooseGroundTruth(id){
  if(D.caseLoading||!D.frame)return;
  if(id!==null&&!D.frame.points.gt.some(r=>r[0]===id))return;
  stop();D.targetGt=id;rankPredictions();
  const assigned=D.frame.matches[D.method].find(r=>r[1]===id);
  D.selectedPred=assigned?.[0]??D.ranked[0]?.row[0]??null;
  D.selectionUiKey=null;fitSelection();render();
}
function renderSelection(){
  const key=`${D.key}:${D.frame.t}:${D.method}:${D.targetGt}:${D.selectedPred}`;
  const info=pairInfo();
  if(D.selectionUiKey===key)return info;
  D.selectionUiKey=key;
  $("d-pred-source").value=D.method==='raw'?'raw':'final:'+D.model;
  $("d-gt-choice").innerHTML='<option value="">No GT selected</option>'+D.frame.points.gt.map((r,i)=>`<option value="${r[0]}">GT${i+1}${D.payload.focus.gt.includes(r[0])?' · case':''}</option>`).join('');
  $("d-gt-choice").value=D.targetGt===null?'':String(D.targetGt);
  $("d-pred-choice").innerHTML=D.ranked.length?D.ranked.map((r,i)=>`<option value="${r.row[0]}">P${i+1}${info.gt?' · '+distance(r.d):''}</option>`).join(''):'<option value="">No predictions</option>';
  $("d-pred-choice").value=D.selectedPred===null?'':String(D.selectedPred);
  $("d-pred-choice").disabled=!D.ranked.length;
  const candidates=D.ranked.slice(0,Math.max(6,info.gt?D.ranked.filter(r=>r.d<=7).length:0));
  const selected=D.ranked.find(r=>r.row[0]===D.selectedPred);if(selected&&!candidates.includes(selected))candidates.push(selected);
  $("d-nearby-picks").innerHTML='<span>Nearby predictions</span>'+candidates.map(r=>`<button data-pred="${r.row[0]}" aria-pressed="${r.row[0]===D.selectedPred}" title="Source ID ${r.row[0]} · Z,Y,X ${r.row.slice(2).join(', ')}">P${D.ranked.indexOf(r)+1}${info.gt?' <small>'+distance(r.d)+'</small>':''}</button>`).join('');
  $("d-nearby-picks").querySelectorAll('button').forEach(b=>b.onclick=()=>choosePrediction(Number(b.dataset.pred)));
  const assignment=info.assignment==='this'?'Matched to this GT':info.assignment==='other'?'Matched to '+info.assignedLabel:'Unassigned';
  $("d-selection-readout").innerHTML=info.gt&&info.pred?
    `<b class="d-key-gt">${info.gtLabel}</b><span>↔</span><b class="d-key-pred">${info.predLabel}</b><strong>${distance(info.distance)} <small>3D</small></strong><span class="d-gate ${info.distance<=7?'inside':'outside'}">${info.distance<=7?'Inside':'Outside'} 7 µm</span><span class="d-assignment ${info.assignment}">${assignment}</span>`:
    info.pred?`<b class="d-key-pred">${info.predLabel}</b><span>No GT selected in this frame</span>`:info.gt?`<b class="d-key-gt">${info.gtLabel}</b><span>No predictions in this frame</span>`:'No center selected';
  $("d-picked").textContent=`${info.gt?'GT source ID '+info.gt[0]+' · ':''}${info.pred?'Prediction source ID '+info.pred[0]+' · '+assignment:''}`;
  return info;
}
async function selectPredictionSource(){
  if(!D.catalog)return;stop();const value=$("d-pred-source").value,oldModel=D.model,oldPool=D.pool,target=D.targetGt,time=D.frame?.t;
  D.method=value==='raw'?'raw':'final';if(value!=='raw')D.model=value.split(':')[1];
  $("d-model").value=D.model;
  D.pool=D.method==='raw'?'raw':'final';$("d-pool").value=D.pool;scopeText();
  if(oldModel!==D.model||(D.filter==='unlabeled'&&oldPool!==D.pool)){
    const key=oldModel!==D.model&&D.key&&D.filter!=='unlabeled'?D.model+D.key.slice(D.key.indexOf(':')):null;
    D.offset=key?D.offset:0;await loadCases(key);
    if(key&&D.key===key){
      const index=D.payload.frames.findIndex(f=>f.t===time);if(index>=0&&D.frame.t!==time)await showFrame(index);
      if(D.key===key&&(target===null||D.frame.points.gt.some(r=>r[0]===target)))chooseGroundTruth(target);
    }return;
  }
  if(D.frame)chooseGroundTruth(target);
}
function projection(full=false) {
  const [h,v,d]=axes(),shape=D.payload.clip.shape.slice(1),mode=full?'mip':D.projection;
  const key=`${D.key}:${D.frame.t}:${D.plane}:${mode}:${full?'':D.depth+':'+D.targetGt+':'+D.method+':'+D.selectedPred}:${D.gamma}`,cache=full?'fullProjectionCache':'projectionCache';
  if(D[cache]?.key===key)return D[cache];
  const width=shape[h],height=shape[v];
  let low=Math.max(0,D.depth-(mode==='slab'?2:0)),high=Math.min(shape[d]-1,D.depth+(mode==='slab'?2:0));
  if(mode==='mip'){low=0;high=shape[d]-1;}
  if(mode==='local'){
    const depths=[D.depth];
    for(const row of [targetRow(),predictionRow()].filter(Boolean))depths.push(row[d+2]);
    if(targetRow())for(const candidate of D.ranked)if(candidate.d<=7)depths.push(candidate.row[d+2]);
    const assigned=D.frame.points.gt.find(r=>r[0]===pairInfo().assignedGT);if(assigned)depths.push(assigned[d+2]);
    low=Math.max(0,Math.floor(Math.min(...depths))-2);high=Math.min(shape[d]-1,Math.ceil(Math.max(...depths))+2);
  }
  const gray=new Uint8Array(width*height),strides=[shape[1]*shape[2],shape[2],1];
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    const start=y*strides[v]+x*strides[h];let best=0;for(let depth=low;depth<=high;depth++)best=Math.max(best,D.volume[start+depth*strides[d]]);gray[y*width+x]=best;
  }
  const texture=document.createElement('canvas');texture.width=width;texture.height=height;
  const ctx=texture.getContext('2d'),pixels=ctx.createImageData(width,height),lut=Array.from({length:256},(_,i)=>Math.round(255*Math.pow(i/255,1/D.gamma)));
  for(let i=0;i<gray.length;i++){pixels.data[4*i]=pixels.data[4*i+1]=pixels.data[4*i+2]=lut[gray[i]];pixels.data[4*i+3]=255;}ctx.putImageData(pixels,0,0);
  return D[cache]={key,gray,texture,width,height,h,v,d,low,high};
}
function transform(canvas,full=false){
  const p=projection(full),spacing=D.payload.clip.spacing;
  const field=full?Math.max(p.width*spacing[p.h],p.height*spacing[p.v]):D.field,scale=canvas.width/field;
  const center=full?D.payload.clip.shape.slice(1).map(n=>(n-1)/2):D.center;
  return {h:p.h,v:p.v,field,scale,ox:canvas.width/2-(center[p.h]+.5)*spacing[p.h]*scale,oy:canvas.height/2-(center[p.v]+.5)*spacing[p.v]*scale};
}
function coords(row,tr){const s=D.payload.clip.spacing;return [tr.ox+(row[tr.h+2]+.5)*s[tr.h]*tr.scale,tr.oy+(row[tr.v+2]+.5)*s[tr.v]*tr.scale];}
function visible(row,p=projection()){return row[p.d+2]>=p.low&&row[p.d+2]<=p.high;}
function paintLabel(ctx,text,x,y,color,size=18){
  ctx.save();ctx.font=`600 ${size}px system-ui`;const width=ctx.measureText(text).width;
  x=Math.max(5,Math.min(ctx.canvas.width-width-8,x));y=Math.max(size+6,Math.min(ctx.canvas.height-6,y));
  ctx.fillStyle='rgba(5,10,16,.92)';ctx.fillRect(x-4,y-size-2,width+8,size+8);ctx.fillStyle=color;ctx.fillText(text,x,y);ctx.restore();
}
function onCanvas(ctx,x,y){return x>=0&&y>=0&&x<=ctx.canvas.width&&y<=ctx.canvas.height;}
function marker(ctx,x,y,method,selected,label,full=false){
  const color=method==='gt'?palettes.gt:palettes.pred,radius=selected?10:full?3.5:5.5;
  ctx.save();ctx.globalAlpha=selected?1:full?.65:.82;ctx.beginPath();
  if(method==='gt'){ctx.moveTo(x,y-radius);ctx.lineTo(x+radius,y);ctx.lineTo(x,y+radius);ctx.lineTo(x-radius,y);ctx.closePath();}
  else ctx.arc(x,y,radius,0,Math.PI*2);
  ctx.strokeStyle='#03080e';ctx.lineWidth=selected?6:3;ctx.stroke();ctx.strokeStyle=color;ctx.lineWidth=selected?3:1.8;ctx.stroke();
  if(method==='gt'){ctx.fillStyle=color;ctx.fillRect(x-1.5,y-1.5,3,3);}
  if(selected&&method!=='gt'){ctx.beginPath();ctx.arc(x,y,15,0,Math.PI*2);ctx.strokeStyle=color;ctx.lineWidth=2;ctx.stroke();}
  ctx.restore();
  if(label&&onCanvas(ctx,x,y))paintLabel(ctx,label,method==='gt'?x-58:x+17,method==='gt'?y-15:y+24,color,full?14:19);
}
function ringGeometry(gt,tr){if(!gt)return null;const [x,y]=coords(gt,tr);return {x,y,radius:7*tr.scale};}
function draw(canvas,method,full=false) {
  const ctx=canvas.getContext('2d'),p=projection(full),tr=transform(canvas,full),s=D.payload.clip.spacing,info=pairInfo();
  ctx.fillStyle='#080c12';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.imageSmoothingEnabled=false;
  ctx.drawImage(p.texture,tr.ox,tr.oy,p.width*s[p.h]*tr.scale,p.height*s[p.v]*tr.scale);
  if(full){
    const [x,y]=coords([0,D.frame.t,...D.center],tr),size=D.field*tr.scale;
    ctx.strokeStyle='#080c12';ctx.lineWidth=5;ctx.strokeRect(x-size/2,y-size/2,size,size);
    ctx.strokeStyle=palettes.crop;ctx.lineWidth=2;ctx.strokeRect(x-size/2,y-size/2,size,size);
  }
  // The ring is the 7 µm sphere's projection, centered on actual ground truth.
  if($("d-labels").checked&&info.gt&&visible(info.gt,p)){
    const ring=ringGeometry(info.gt,tr);ctx.save();ctx.beginPath();ctx.arc(ring.x,ring.y,ring.radius,0,2*Math.PI);
    ctx.fillStyle='rgba(255,209,102,.035)';ctx.fill();ctx.setLineDash([9,6]);
    ctx.strokeStyle='#090d12';ctx.lineWidth=5;ctx.stroke();ctx.strokeStyle=palettes.gt;ctx.lineWidth=2.5;ctx.stroke();ctx.restore();
    if(onCanvas(ctx,ring.x,ring.y))paintLabel(ctx,'7 µm',ring.x-22,ring.y-ring.radius-9,palettes.gt,full?16:20);
  }
  if($("d-lines").checked&&$("d-labels").checked&&$("d-centers").checked&&info.gt&&info.pred&&visible(info.gt,p)&&visible(info.pred,p)){
    const [x1,y1]=coords(info.gt,tr),[x2,y2]=coords(info.pred,tr);
    ctx.save();ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);if(info.assignment!=='this')ctx.setLineDash([6,4]);
    ctx.strokeStyle='#080c12';ctx.lineWidth=5;ctx.stroke();ctx.strokeStyle=palettes.pred;ctx.lineWidth=2;ctx.stroke();ctx.restore();
  }
  if($("d-centers").checked)for(const row of D.frame.points[method])if(row[0]!==D.selectedPred&&visible(row,p)){
    const [x,y]=coords(row,tr),rank=D.ranked.findIndex(r=>r.row[0]===row[0]);
    marker(ctx,x,y,'pred',false,!full&&rank<6?'P'+(rank+1):'',full);
  }
  if($("d-labels").checked)for(const row of D.frame.points.gt)if(row[0]!==D.targetGt&&visible(row,p)){
    const [x,y]=coords(row,tr);marker(ctx,x,y,'gt',false,'',full);
  }
  if($("d-lines").checked&&$("d-labels").checked&&$("d-centers").checked&&info.assignment==='other'&&info.pred&&visible(info.pred,p)){
    const assigned=D.frame.points.gt.find(r=>r[0]===info.assignedGT);
    if(assigned&&visible(assigned,p)){
      const [ax,ay]=coords(assigned,tr),[px,py]=coords(info.pred,tr);
      ctx.beginPath();ctx.moveTo(px,py);ctx.lineTo(ax,ay);ctx.strokeStyle='#080c12';ctx.lineWidth=5;ctx.stroke();ctx.strokeStyle=palettes.pred;ctx.lineWidth=2;ctx.stroke();
      marker(ctx,ax,ay,'gt',false,info.assignedLabel,full);
    }
  }
  // Selected centers keep their source colors, including exact overlaps.
  if($("d-centers").checked&&info.pred&&visible(info.pred,p)){const [x,y]=coords(info.pred,tr);marker(ctx,x,y,'pred',true,info.predLabel,full);}
  if($("d-labels").checked&&info.gt&&visible(info.gt,p)){const [x,y]=coords(info.gt,tr);marker(ctx,x,y,'gt',true,info.gtLabel,full);}
  const length=full?20:D.field<30?5:10,bar=length*tr.scale;ctx.fillStyle='rgba(0,0,0,.8)';ctx.fillRect(12,canvas.height-50,bar+20,40);
  ctx.strokeStyle='white';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(22,canvas.height-22);ctx.lineTo(22+bar,canvas.height-22);ctx.stroke();ctx.fillStyle='white';ctx.font='19px system-ui';ctx.fillText(`${length} µm`,22,canvas.height-32);
  ctx.fillText(`${'ZYX'[p.h]} → / ${'ZYX'[p.v]} ↓`,canvas.width-125,canvas.height-22);
}
function renderNeighbors() {
  const anchor=targetRow()?.slice(2)||D.payload.anchor,s=D.payload.clip.spacing;
  $("d-neighbor-caption").textContent=`Frame ${D.frame.t}. Distances from the selected GT or case location (${anchor.join(', ')} in Z,Y,X). The closest eight centers and every center within 7 µm are included. Nearest and assigned are distinct.`;
  const table=method=>{
    const map=new Map((D.frame.matches[method]||[]).map(r=>[r[0],r[1]]));
    const rows=D.frame.points[method].map(r=>({row:r,d:Math.hypot(...r.slice(2).map((v,i)=>(v-anchor[i])*s[i]))})).sort((a,b)=>a.d-b.d);
    const shown=rows.slice(0,Math.max(8,rows.filter(r=>r.d<=7).length));
    return `<div><h4>${method==='raw'?'Detector candidates':'Final centers'}</h4><table><thead><tr><th>ID</th><th>Distance</th><th>Assigned GT</th></tr></thead><tbody>${shown.map(({row,d})=>`<tr><td>${row[0]}</td><td>${distance(d)}</td><td>${map.get(row[0])??'None'}</td></tr>`).join('')||'<tr><td colspan="3">No predictions in this frame</td></tr>'}</tbody></table></div>`;
  };$("d-neighbors").innerHTML=table('raw')+table('final');
}
function render() {
  if(!D.payload||!D.frame)return;
  const shape=D.payload.clip.shape.slice(1),d=axes()[2];D.depth=Math.max(0,Math.min(shape[d]-1,Math.round(D.depth)));
  $("d-plane").value=D.plane;$("d-projection").value=D.projection;$("d-depth").max=shape[d]-1;$("d-depth").value=D.depth;$("d-depth").disabled=['mip','local'].includes(D.projection);
  $("d-depth-label").textContent=`${'ZYX'[d]} ${D.depth} · ${(D.depth*D.payload.clip.spacing[d]).toFixed(2)} µm`;
  $("d-time").max=D.payload.frames.length-1;$("d-time").value=D.position;
  $("d-time-label").textContent=`t${D.frame.t}`;
  $("d-back").disabled=D.position===0;$("d-forward").disabled=D.position===D.payload.frames.length-1;$("d-play").disabled=!continuous();
  $("d-brightness").value=D.gamma;$("d-width").value=String(D.field);
  const info=renderSelection();
  draw(canvases[0],D.method);draw(fullCanvases[0],D.method,true);
  $("d-frame-count").textContent=`${D.frame.points.gt.length} GT · ${D.frame.points[D.method].length} predictions`;
  $("d-zoom-status").textContent=info.gt?`${D.ranked.filter(r=>r.d<=7).length} predictions within 7 µm`:'Select a GT to measure';
  const p=projection();$("d-image-caption").textContent=`Frame ${D.frame.t} · ${D.plane.toUpperCase()} · Full: maximum projection · Zoom: ${'ZYX'[d]} ${p.low}–${p.high} · distances in 3D`;
  renderNeighbors();syncHash();
}
function adjacent(delta) {
  const index=D.rows.findIndex(r=>r.key===D.key),next=index+delta;
  if(next>=0&&next<D.rows.length)return selectCase(D.rows[next].key);
  if(delta>0&&D.offset+D.limit<D.total){D.offset+=D.limit;return loadCases();}
  if(delta<0&&D.offset>0){D.offset=Math.max(0,D.offset-D.limit);return loadCases(null,-1);}
}
for(const [id,key] of [['d-filter','filter'],['d-clip','dataset'],['d-pool','pool']])$(id).onchange=()=>{D[key]=$(id).value;if(id==='d-pool')D.method=D.pool==='raw'?'raw':'final';D.offset=0;loadCases();};
$("d-model").onchange=()=>{$("d-pred-source").value='final:'+$("d-model").value;selectPredictionSource();};
$("d-pred-source").onchange=selectPredictionSource;
$("d-gt-choice").onchange=()=>chooseGroundTruth($("d-gt-choice").value===''?null:Number($("d-gt-choice").value));
$("d-pred-choice").onchange=()=>choosePrediction(Number($("d-pred-choice").value));
$("d-fit-pair").onclick=()=>{if(D.frame){fitSelection();render();}};
document.querySelectorAll('#d-groups button').forEach(b=>b.onclick=()=>{if(!D.catalog)return;D.filter=b.dataset.filter;D.dataset="";D.frameFilter="";$("d-clip").value="";$("d-frame-filter").value="";$("d-filter").value=D.filter;D.offset=0;loadCases();});
$("d-reset").onclick=()=>{D.dataset="";D.frameFilter="";D.filter="all";D.offset=0;$("d-clip").value="";$("d-frame-filter").value="";$("d-filter").value="all";loadCases();};
$("d-jump-go").onclick=()=>{if(!D.total)return;const number=Math.max(1,Math.min(D.total,Math.floor(Number($("d-jump").value)||1)))-1;D.offset=Math.floor(number/D.limit)*D.limit;loadCases(null,number%D.limit);};
$("d-jump").onkeydown=e=>{if(e.key==='Enter')$("d-jump-go").click();};
$("d-page-prev").onclick=()=>{D.offset=Math.max(0,D.offset-D.limit);loadCases();};$("d-page-next").onclick=()=>{D.offset+=D.limit;loadCases();};
$("d-page-go").onclick=()=>{const page=Math.max(1,Math.min(Math.ceil(D.total/D.limit)||1,Number($("d-page").value)||1));D.offset=(page-1)*D.limit;loadCases();};
$("d-page").onkeydown=e=>{if(e.key==='Enter')$("d-page-go").click();};
$("d-frame-filter").onchange=()=>{const value=$("d-frame-filter").value;if(value!==''&&(!Number.isInteger(Number(value))||Number(value)<0||Number(value)>99))return;D.frameFilter=value;D.offset=0;loadCases();};
$("d-prev").onclick=()=>adjacent(-1);$("d-next").onclick=()=>adjacent(1);
$("d-play").onclick=play;$("d-back").onclick=()=>{stop();showFrame(D.position-1);};$("d-forward").onclick=()=>{stop();showFrame(D.position+1);};
$("d-time").oninput=()=>{stop();showFrame(Number($("d-time").value));};$("d-case-frame").onclick=()=>{stop();if(D.payload)showFrame(D.payload.frames.findIndex(f=>f.t===D.payload.case.t));};
$("d-plane").onchange=()=>{D.plane=$("d-plane").value;D.depth=Math.round(D.center[axes()[2]]);D.projectionCache=null;render();};
$("d-projection").onchange=()=>{D.projection=$("d-projection").value;D.projectionCache=null;render();};
$("d-depth").oninput=()=>{D.depth=Number($("d-depth").value);D.projectionCache=null;render();};
$("d-width").onchange=()=>{D.field=Number($("d-width").value);if(D.field===104)D.center=D.payload.clip.shape.slice(1).map(n=>(n-1)/2);render();};
$("d-brightness").oninput=()=>{D.gamma=Number($("d-brightness").value);D.projectionCache=null;render();};
for(const id of ['d-centers','d-labels','d-lines'])$(id).onchange=render;
$("d-recenter").onclick=()=>{if(D.payload){D.center=[...D.payload.anchor];D.depth=Math.round(D.center[axes()[2]]);D.field=40;D.projectionCache=null;render();}};
$("d-save").onclick=()=>saveAssessment();
$("d-verdict").onchange=()=>saveAssessment();$("d-notes").onchange=()=>saveAssessment();
$("d-export").onclick=()=>{
  saveAssessment();const blob=new Blob([JSON.stringify({schema:1,exported:new Date().toISOString(),interpretation:'Human assessments; automatic flags are not confirmed diagnoses.',assessments:Object.values(assessments)},null,2)],{type:'application/json'});
  const a=document.createElement('a'),url=URL.createObjectURL(blob);a.href=url;a.download='cell-detection-assessments.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
$("d-copy").onclick=async()=>{syncHash();try{await navigator.clipboard.writeText(location.href);$("d-save-status").textContent="Case link copied";}catch{$("d-save-status").textContent="The browser address contains this case and view; copy it to reopen.";}};
function hitCenter(canvas,event,full=false){
  const rect=canvas.getBoundingClientRect(),tr=transform(canvas,full),p=projection(full),x=event.clientX-rect.left,y=event.clientY-rect.top;
  const candidates=[];
  if($("d-centers").checked)for(const row of D.frame.points[D.method])candidates.push({row,method:'pred'});
  if($("d-labels").checked)for(const row of D.frame.points.gt)candidates.push({row,method:'gt'});
  let picked=null,best=13;
  for(const c of candidates){if(!visible(c.row,p))continue;const [cx,cy]=coords(c.row,tr),distance=Math.hypot(cx*rect.width/canvas.width-x,cy*rect.height/canvas.height-y);
    // Prediction wins an exact overlap; GT also has its explicit picker.
    if(distance<best){best=distance;picked=c;}
  }return picked;
}
for(const [canvas,full] of [[canvases[0],false],[fullCanvases[0],true]]){
  let drag=null;
  canvas.addEventListener('pointerdown',e=>{if(!D.frame||D.caseLoading)return;canvas.setPointerCapture(e.pointerId);drag={x:e.clientX,y:e.clientY,center:[...D.center],moved:false};});
  canvas.addEventListener('pointermove',e=>{
    if(!D.frame||D.caseLoading)return;
    canvas.style.cursor=hitCenter(canvas,e,full)?'pointer':full?'crosshair':'grab';
    if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(Math.hypot(dx,dy)>4)drag.moved=true;if(full)return;
    if(drag.moved){const [h,v]=axes(),rect=canvas.getBoundingClientRect(),s=D.payload.clip.spacing;D.center[h]=drag.center[h]-dx/rect.width*D.field/s[h];D.center[v]=drag.center[v]-dy/rect.height*D.field/s[v];render();}
  });
  canvas.addEventListener('pointerup',e=>{
    if(drag&&!drag.moved&&D.frame&&!D.caseLoading){stop();const picked=hitCenter(canvas,e,full);
      if(picked){if(picked.method==='gt')chooseGroundTruth(picked.row[0]);else choosePrediction(picked.row[0]);}
      else if(full){const box=canvas.getBoundingClientRect(),tr=transform(canvas,true),s=D.payload.clip.spacing;
        D.center[tr.h]=((e.clientX-box.left)*canvas.width/box.width-tr.ox)/tr.scale/s[tr.h]-.5;
        D.center[tr.v]=((e.clientY-box.top)*canvas.height/box.height-tr.oy)/tr.scale/s[tr.v]-.5;render();
      }
    }drag=null;
  });canvas.addEventListener('pointercancel',()=>drag=null);
  if(!full)canvas.addEventListener('wheel',e=>{if(!D.frame||D.caseLoading)return;e.preventDefault();D.field=Math.max(12,Math.min(150,D.field*Math.exp(e.deltaY*.0015)));render();},{passive:false});
}
document.addEventListener('keydown',e=>{
  if(activeTab!=='detection'||!D.payload||['INPUT','SELECT','TEXTAREA','BUTTON','SUMMARY'].includes(document.activeElement?.tagName))return;
  if(e.code==='Space'){e.preventDefault();play();}else if(e.key==='ArrowRight'||e.key==='ArrowLeft'){e.preventDefault();stop();adjacent(e.key==='ArrowRight'?1:-1);}
});
window.addEventListener('beforeunload',()=>saveAssessment(true));
async function initialize() {
  if(initializing)return;initializing=true;
  controlIds.forEach(id=>$(id).disabled=true);
  try{
    D.catalog=await api('catalog');
    $("d-coverage").textContent=`All ${D.catalog.datasets} clips`;
    $("d-clip").innerHTML='<option value="">All '+D.catalog.datasets+' clips</option>'+['44b6','6bba'].map(embryo=>`<optgroup label="Embryo ${embryo}">${D.catalog.clips.filter(c=>c.embryo===embryo).map(c=>`<option value="${c.dataset}">${c.dataset}</option>`).join('')}</optgroup>`).join('');
    // Old URLs defaulted to one clip. Only new explicit review links restore filters.
    if(params.get('review')==='2'){
      for(const [k,id,q] of [['model','d-model','dmodel'],['filter','d-filter','dfilter'],['dataset','d-clip','dclip'],['pool','d-pool','dpool']])if([...$(id).options].some(o=>o.value===params.get(q))){D[k]=params.get(q);$(id).value=D[k];}
      if(params.has('doffset'))D.offset=Math.max(0,Math.min(100000000,Number(params.get('doffset'))||0));
      if(params.has('dframefilter')&&/^\d{1,2}$/.test(params.get('dframefilter'))){D.frameFilter=params.get('dframefilter');$("d-frame-filter").value=D.frameFilter;}
      if(['raw','final'].includes(params.get('dmethod')))D.method=params.get('dmethod');
      else D.method=D.pool==='raw'?'raw':'final';
      D.pool=D.method==='raw'?'raw':'final';
    }
    $("d-filter").value=D.filter;
    $("d-provenance").textContent=`${D.catalog.complete_training_population?'Complete training population':'Partial development index'}: ${pretty(D.catalog.datasets)} clips, ${pretty(D.catalog.frames)} timepoints, ${pretty(D.catalog.annotations)} sparse GT nodes. Final pipelines: ${Object.values(D.catalog.models).join(', ')}. Every prediction file was checked against its evaluation receipt and the full-clip official scoring counts were reproduced. Native images are read on demand at the selected timepoint; no image is substituted from the 42-frame FOCUS panel. The original comparison remains in the other tab.`;
    controlIds.forEach(id=>$(id).disabled=false);
    await loadCases(params.get('review')==='2'?params.get('dcase'):null,0,true);
    window.detectionReview={get state(){return D;},projection,transform,coords,ringGeometry,pairInfo,centerComparison,hitCenter,get assessments(){return assessments;}};
  }catch(e){error(e);}finally{initializing=false;}
}
function tab(name) {
  if(activeTab==='comparison')comparisonHash=location.hash;
  if(name!=='comparison')window.centerComparison?.stop();
  if(name!=='detection'){saveAssessment(true);stop();}
  if(!['report','tracking'].includes(name))window.pipelineReview?.deactivate();
  activeTab=name;
  for(const n of ['report','detection','tracking','comparison']){$(n+'-panel').hidden=name!==n;$("tab-"+n).setAttribute('aria-selected',String(name===n));$("tab-"+n).tabIndex=name===n?0:-1;}
  $("status").parentElement.hidden=name!=='comparison';
  if(name==='detection'){syncHash();if(!D.catalog)initialize();}else if(name==='comparison') {
    const q=new URLSearchParams(comparisonHash.includes('tab=detection')?'':comparisonHash.slice(1));q.set('tab','comparison');history.replaceState(null,'','#'+q);
  }else window.pipelineReview?.activate(name);
}
window.switchReviewTab=tab;
for(const name of ['report','tracking','comparison','detection'])$("tab-"+name).onclick=()=>tab(name);
$("open-all-errors").onclick=()=>{tab('detection');if(D.catalog)$("d-reset").onclick();};
document.querySelector('.main-tabs').addEventListener('keydown',e=>{if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();const names=['report','detection','tracking','comparison'];const name=names[(names.indexOf(activeTab)+(e.key==='ArrowRight'?1:3))%4];tab(name);$("tab-"+name).focus();}});
tab(['report','comparison','tracking'].includes(params.get('tab'))?params.get('tab'):'detection');
})();

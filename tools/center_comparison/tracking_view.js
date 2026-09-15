/* Full-population report and temporal error review. No model inference in the UI. */
"use strict";
(() => {
const $=id=>document.getElementById(id), initial=new URLSearchParams(location.hash.slice(1));
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=n=>Number(n).toLocaleString(), pct=(n,d)=>d?`${(100*n/d).toFixed(1)}%`:'—';
const decimal=n=>Number(n).toFixed(6), delta=n=>`${n>=0?'+':''}${Number(n).toFixed(6)}`;
const titles={edge_endpoint:'Unmatched endpoint',edge_link:'Matched cells, missing link',edge_fp:'Incorrect connection',division_fn:'Missed division',division_fp:'Incorrect division'};
const T={report:null,active:null,model:initial.get('tmodel')||initial.get('rmodel')||'pooled',filter:initial.get('tfilter')||'all',stage:initial.get('tstage')||'',
  embryo:initial.get('tembryo')||'',dataset:initial.get('tclip')||'',offset:Number(initial.get('toffset'))||0,limit:40,total:0,rows:[],key:initial.get('tcase'),
  payload:null,plane:initial.get('tplane')||'xy',projection:initial.get('tprojection')||'local',width:initial.get('twidth')||'auto',gamma:1,
  busy:false,epoch:0,volumes:new Map(),imageEpoch:0};
let reportJob=null,initialized=false;
const pending=new Map(),cache=new Map(),storageKey='biohub-tracking-assessments-v1';
let notes={};try{notes=JSON.parse(localStorage.getItem(storageKey)||'{}');}catch{}
async function api(path,params={}){const r=await fetch('/api/tracking/'+path+'?'+new URLSearchParams(params));const data=await r.json();if(!r.ok)throw Error(data.error||`HTTP ${r.status}`);return data;}
function fail(id,error){$(id).hidden=false;$(id).textContent=error.message||String(error);}
function hash(){
  if(!['tracking','report'].includes(T.active))return;
  const q=T.active==='report'?new URLSearchParams({tab:'report',rmodel:$('r-model').value,rembryo:$('r-embryo').value}):
    new URLSearchParams({tab:'tracking',tmodel:T.model,tfilter:T.filter,tstage:T.stage,tembryo:T.embryo,tclip:T.dataset,toffset:T.offset,tplane:T.plane,tprojection:T.projection,twidth:T.width});
  if(T.active==='tracking'&&T.key)q.set('tcase',T.key);
  history.replaceState(null,'','#'+q);
}
async function loadReport(){if(T.report)return T.report;if(!reportJob)reportJob=api('report').then(r=>T.report=r).catch(e=>{reportJob=null;throw e;});return reportJob;}
const linkButton=(label,filter,extra={})=>`<button class="tr-text-link" data-review="${filter}" ${Object.entries(extra).map(([k,v])=>`data-${k}="${esc(v)}"`).join(' ')}>${esc(label)} →</button>`;
function bindReportLinks(){document.querySelectorAll('[data-review]').forEach(b=>b.onclick=()=>openReview(b.dataset.review,b.dataset.clip||'',b.dataset.stage||''));}
function renderReport(){
  const model=$('r-model').value, scope=$('r-embryo').value, r=T.report.models[model], s=r.summaries[scope], c=s.counts;
  const missing=s.annotations-s.matched_nodes, divTotal=c.division_tp+c.division_fn;
  const nErrors=c.edge_fn+c.edge_fp, lossMax=Math.max(...s.scenarios.map(x=>x.delta),0.001);
  const stages=s.stages, e=s.categories;
  const candidates=s.centers.proposal_only||0;
  const priority=scope==='44b6'?`In 44b6, the edge term has more accounting headroom than the division term. Across all 199 clips, division decisions have the larger gap. Observation selection and identity affect both missing and incorrect links.`:`Division decisions have the largest single-term accounting gap. Observation selection and identity affect both missing and incorrect links.`;
  const clips=r.clips.filter(x=>scope==='all'||x.embryo===scope).sort((a,b)=>(b.edge_fn+b.edge_fp)-(a.edge_fn+a.edge_fp));
  $('r-content').innerHTML=`
    <section class="tr-lead"><div><h3>Focus next: division decisions, then observation identity</h3><p>${priority}</p><p>${num(c.division_fn)} of ${num(divTotal)} annotated divisions are missed. ${num(e.edge_endpoint||0)} of ${num(c.edge_fn)} missed links have an unmatched endpoint; another ${num(e.edge_link||0)} fail with both centers available.</p></div><div class="tr-scope-note"><strong>Local diagnostic evidence</strong><p>${esc(T.report.scope)}</p></div></section>
    <section class="tr-score-strip" aria-label="Score breakdown"><div><span>Combined score · ${esc(r.name)}</span><strong>${decimal(s.score)}</strong><small>${esc(r.status)}</small></div><div><span>Adjusted edge contribution</span><strong>${decimal(s.adj_edge_jaccard)}</strong><small>Raw edge Jaccard ${decimal(s.edge_jaccard)}</small></div><div><span>Division contribution · maximum 0.1</span><strong>${decimal(0.1*s.division_jaccard)}</strong><small>${num(c.division_tp)} TP / ${num(c.division_fp)} FP / ${num(c.division_fn)} FN</small></div><div><span>Annotated center recall</span><strong>${pct(s.matched_nodes,s.annotations)}</strong><small>${num(s.matched_nodes)} / ${num(s.annotations)} sparse annotations</small></div></section>
    <div class="tr-report-columns"><section><h3>Errors along the pipeline</h3><p class="tr-caption">Link FN groups partition all missed edges. Incorrect links and division flags can overlap them.</p>
    <div class="tr-stage-row"><div><strong>1. Propose and select observations</strong><p>${num(missing)} unmatched centers; ${num(candidates)} have a detector candidate within 7 µm but no nearby final center (${pct(candidates,missing)}).</p></div><div><b>${num(e.edge_endpoint||0)}</b><small>blocked links · ${pct(e.edge_endpoint||0,c.edge_fn)} of FN</small>${linkButton('Inspect endpoints','edge_endpoint')}</div></div>
    <div class="tr-stage-row"><div><strong>2. Connect the right observations</strong><p>${num(e.edge_link||0)} missed links have both endpoints matched. ${num((stages.competing||0)+(stages.unmatched||0))} of ${num(c.edge_fp)} incorrect links touch an unmatched prediction; ${num(stages.competing||0)} use one within 7 µm of the expected, already assigned cell. Proximity flags identity competition; it does not prove a duplicate.</p></div><div><b>${num(c.edge_fp+(e.edge_link||0))}</b><small>association-related edge flags</small>${linkButton('Inspect connections','edge_fp')}</div></div>
    <div class="tr-stage-row"><div><strong>3. Recover parent → daughter events</strong><p>Division recall ${pct(c.division_tp,divTotal)}; precision ${pct(c.division_tp,c.division_tp+c.division_fp)}. ${num(stages.division_no_fork||0)} misses have local parent and daughter matches but no parent-side predicted fork.</p></div><div><b>${num(c.division_fn+c.division_fp)}</b><small>division error flags</small>${linkButton('Inspect divisions','division_fn')}</div></div>
    <p class="tr-caption">Node-count adjustment contributes ${delta(s.node_adjustment)} relative to raw edge Jaccard in this scope. This is the aggregate effect of coarse clip-level count estimates, not a separate set of known false cells.</p>
    </section><section><h3>Score impact of idealized repairs</h3><p class="tr-caption">Change one group of counts at a time; recompute the official weighted score. Bars show change in combined-score units.</p><div class="tr-scenario-axis"><span>0</span><span>Δ score ${lossMax.toFixed(3)}</span></div>
    ${[...s.scenarios].sort((a,b)=>b.delta-a.delta).map(x=>`<div class="tr-scenario"><div><span>${esc(x.title)}</span><strong>${delta(x.delta)}</strong></div><div class="tr-bar-track"><span style="width:${100*x.delta/lossMax}%"></span></div></div>`).join('')}
    <p class="tr-caution">${esc(T.report.scenario_note)}</p></section></div>
    <section class="tr-breakdown"><h3>Evidence behind the priorities</h3><div class="tr-report-columns">
      <div><h4>Missed divisions · ${num(c.division_fn)} total</h4>${[['division_no_fork','Local matches present, no parent-side fork'],['division_observation','Insufficient matches in the local window'],['division_topology','Fork exists, invalid branches'],['division_pairing','Valid fork lost event pairing']].filter(([k])=>stages[k]).map(([k,label])=>`<div class="tr-breakdown-row"><span>${esc(label)}</span><strong>${num(stages[k])} · ${pct(stages[k],c.division_fn)}</strong>${linkButton('Inspect','division_fn',{stage:k})}</div>`).join('')}
      <h4>Incorrect divisions · ${num(c.division_fp)} total</h4>${[['division_extra','Evaluable fork without a recovered division'],['division_cross','Branches contradict separate annotated lineages'],['division_merge','Merged daughter branches']].filter(([k])=>stages[k]).map(([k,label])=>`<div class="tr-breakdown-row"><span>${esc(label)}</span><strong>${num(stages[k])}</strong>${linkButton('Inspect','division_fp',{stage:k})}</div>`).join('')}</div>
      <div><h4>Endpoint-blocked links · ${num(e.edge_endpoint||0)} total</h4>${[['selection','Nearby detector proposal; final observation unavailable'],['proposal_gap','At least one missing endpoint has no nearby proposal'],['assignment','Nearby final centers assigned elsewhere']].filter(([k])=>stages[k]).map(([k,label])=>`<div class="tr-breakdown-row"><span>${esc(label)}</span><strong>${num(stages[k])} · ${pct(stages[k],e.edge_endpoint)}</strong>${linkButton('Inspect','edge_endpoint',{stage:k})}</div>`).join('')}<p class="tr-caption">These groups partition missed links by their unmatched endpoint evidence. If a link has different failures at its two endpoints, no-proposal takes priority, then assignment conflict, then final-selection gap. The labels localize the failure; they do not establish its biological cause.</p></div>
    </div></section>
    <section class="tr-action-section"><h3>What to test next</h3><ol class="tr-actions">
      <li><strong>Train and calibrate division versus continuation + birth on fixed incumbent observations.</strong><p>Start with missed divisions that have local matches but no fork; then check the false-fork branches. Compare event scores and acceptance gates while preserving correct continuations. Optimize the combined metric, with a matched no-new-event control. ${linkButton('Open matched-window misses','division_fn',{stage:'division_no_fork'})}</p></li>
      <li><strong>Audit observation selection and ambiguous identity before replacing the detector.</strong><p>${num(candidates)} nearby-proposal/final-gap cases and ${num(stages.competing||0)} competing-center false links are directly inspectable. Follow each track across time; distinguish rejected observations, displaced centers and nearby real cells. ${linkButton('Open competing observations','edge_fp',{stage:'competing'})}</p></li>
      <li><strong>Improve continuation scores and decoding where both endpoints exist.</strong><p>Separate missing candidate links from rejected links using the candidate bank for the exact pipeline in the next experiment. The endpoint audit here identifies where linking can matter; it does not identify the decoder's reason. ${linkButton('Open linking failures','edge_link')}</p></li>
    </ol><p class="tr-caution"><strong>Validation gate:</strong> retain P0 as the adopted reference and C4_m6 as the pooled-score comparator. Lock embryo splits and audit every upstream checkpoint's exposure before interpreting a new training result as generalization. Existing local results have repeated embryo reuse.</p></section>
    <div class="tr-report-columns"><section><h3>Best available complete solutions</h3><div class="tr-table-wrap"><table><thead><tr><th>Pipeline</th><th>Score in this scope</th><th>Decision</th></tr></thead><tbody>${Object.values(T.report.models).map(m=>`<tr><td>${esc(m.name)}</td><td>${decimal(m.summaries[scope].score)}</td><td>${esc(m.status)}</td></tr>`).join('')}</tbody></table></div><p class="tr-caption">C0 / selected v3: 0.934802374261 across all 199 clips. C4_m6 was not adopted: its 6bba score regressed versus C0 and the required replication did not qualify. Newer Cellpose/ultrack experiments cover six reused clips and do not replace a complete 199-clip evaluation.</p>
    <h3>By embryo · ${esc(r.name)}</h3><div class="tr-table-wrap"><table><thead><tr><th>Embryo</th><th>Clips</th><th>Score</th><th>Edge FN / FP</th><th>Division FN / FP</th></tr></thead><tbody>${['44b6','6bba'].map(k=>{const v=r.summaries[k];return `<tr><td>${k}</td><td>${v.clips}</td><td>${decimal(v.score)}</td><td>${num(v.counts.edge_fn)} / ${num(v.counts.edge_fp)}</td><td>${v.counts.division_fn} / ${v.counts.division_fp}</td></tr>`;}).join('')}</tbody></table></div></section>
    <section><h3>Where link errors concentrate</h3><p class="tr-caption">Top ten clips by edge FP + FN in the selected scope; ${num(nErrors)} edge error flags overall. Long or densely annotated tracks contribute more observations.</p><div class="tr-table-wrap"><table><thead><tr><th>Clip</th><th>FN</th><th>FP</th><th>Share of error flags</th></tr></thead><tbody>${clips.slice(0,10).map(x=>`<tr><td>${linkButton(x.dataset,'all',{clip:x.dataset})}</td><td>${num(x.edge_fn)}</td><td>${num(x.edge_fp)}</td><td>${pct(x.edge_fn+x.edge_fp,nErrors)}</td></tr>`).join('')}</tbody></table></div></section></div>
    <section class="tr-methods"><h3>Evidence and interpretation</h3><p>${esc(T.report.source)} ${esc(T.report.counting)}</p><p>Division verdicts use independently matched local windows around the parent and daughters, including one-frame timing tolerance. Whole-clip center matching can differ from these local matches. Edge verdicts use whole-clip matching. Unlabeled predictions are unknown; displaced-but-matched centers are diagnostics, not extra score penalties.</p><p>Generated ${esc(T.report.created_utc)} · ${num(T.report.datasets)} complete clips · ${num(T.report.frames)} frames · metric revision <code>${esc(T.report.metric_revision)}</code>.</p><p><a href="https://github.com/royerlab/kaggle-cell-tracking-competition/blob/${esc(T.report.metric_revision)}/metrics.md" target="_blank" rel="noopener">Pinned official metric</a> · ${num(T.report.provenance.length)} verified graph evaluations · per-clip counts and source hashes in Download evidence.</p></section>`;
  bindReportLinks(); hash();
}
async function openReview(filter='all',dataset='',stage=''){
  T.model=$('r-model').value;T.embryo=$('r-embryo').value==='all'?'':$('r-embryo').value;T.filter=filter;T.dataset=dataset;T.stage=stage;T.offset=0;T.key=null;
  if(initialized)syncControls();window.switchReviewTab('tracking');
  if(initialized)await loadCases();
}
function syncControls(){
  for(const [id,value] of [['t-model',T.model],['t-filter',T.filter],['t-stage',T.stage],['t-embryo',T.embryo]])$(id).value=value;
  const clips=T.report.models[T.model].clips.filter(c=>!T.embryo||c.embryo===T.embryo);
  $('t-clip').innerHTML='<option value="">All clips</option>'+clips.map(c=>`<option>${esc(c.dataset)}</option>`).join('');
  if(!clips.some(c=>c.dataset===T.dataset))T.dataset='';$('t-clip').value=T.dataset;
  $('t-plane').value=T.plane;$('t-projection').value=T.projection;$('t-width').value=T.width;
}
function renderList(){
  $('t-total').textContent=`${num(T.total)} error flags`;$('t-range').textContent=T.total?`${num(T.offset+1)}–${num(T.offset+T.rows.length)}`:'';
  $('t-page-label').textContent=T.total?`Page ${Math.floor(T.offset/T.limit)+1} / ${Math.ceil(T.total/T.limit)}`:'0 pages';
  $('t-list').innerHTML=T.rows.map(r=>`<button role="listitem" class="tr-case-item ${r.key===T.key?'selected':''}" data-key="${esc(r.key)}"><strong>${esc(titles[r.category])}</strong><small>${esc(r.dataset)} · t=${r.t}${notes[r.key]?' · reviewed':''}</small><span>${esc(T.report.stages[r.stage])}</span></button>`).join('');
  document.querySelectorAll('.tr-case-item').forEach(b=>b.onclick=()=>selectCase(b.dataset.key));
  $('t-page-prev').disabled=T.offset===0||T.busy;$('t-page-next').disabled=T.offset+T.rows.length>=T.total||T.busy;
  const index=T.offset+T.rows.findIndex(r=>r.key===T.key);
  $('t-progress').textContent=T.total?`${num(index+1)} / ${num(T.total)}`:'0 / 0';$('t-jump').max=Math.max(1,T.total);$('t-jump').value=Math.max(1,index+1);
  $('t-prev').disabled=!T.total||index===0||T.busy;$('t-next').disabled=!T.total||index>=T.total-1||T.busy;
}
async function loadCases(preferred=T.key){
  saveNotes(true);const epoch=++T.epoch;T.busy=true;T.imageEpoch++;$('t-error').hidden=true;$('t-case').hidden=true;$('t-empty').hidden=true;
  renderList();
  try{
    const data=await api('cases',{model:T.model,filter:T.filter,stage:T.stage,embryo:T.embryo,dataset:T.dataset,offset:T.offset,limit:T.limit,...(preferred?{key:preferred}:{})});
    if(epoch!==T.epoch)return;Object.assign(T,data);T.busy=false;T.payload=null;T.key=null;renderList();
    if(data.rows.length)await selectCase(data.rows.some(r=>r.key===preferred)?preferred:data.rows[0].key);else{$('t-empty').hidden=false;hash();}
  }catch(e){if(epoch===T.epoch){T.busy=false;fail('t-error',e);renderList();}}
}
async function jump(index){
  if(!T.total||T.busy)return;index=Math.max(0,Math.min(T.total-1,index));
  if(index>=T.offset&&index<T.offset+T.rows.length)return selectCase(T.rows[index-T.offset].key);
  T.offset=Math.floor(index/T.limit)*T.limit;await loadCases(null);
  if(index!==T.offset&&T.rows[index-T.offset])await selectCase(T.rows[index-T.offset].key);
}
async function selectCase(key){
  saveNotes(true);const epoch=++T.epoch;T.busy=true;T.imageEpoch++;T.key=key;$('t-case').hidden=true;$('t-error').hidden=true;renderList();
  try{const payload=await api('case',{key});if(epoch!==T.epoch)return;T.payload=payload;T.busy=false;renderList();renderCase();hash();await renderImages();}
  catch(e){if(epoch===T.epoch){T.busy=false;fail('t-error',e);renderList();}}
}
function graph(kind){
  const scene=T.payload.scene, expected=kind==='expected', points=scene.points.filter(p=>p.kind===(expected?'annotation':'prediction'));
  const times=scene.times, edges=scene[kind], byTime=times.map(t=>points.filter(p=>p.t===t));
  const width=Math.max(360,times.length*145),height=Math.max(155,Math.max(...byTime.map(p=>p.length),1)*52+65), positions=new Map();
  byTime.forEach((list,i)=>list.forEach((p,j)=>positions.set(p.key,[60+i*(width-110)/Math.max(1,times.length-1),60+j*52])));
  const color=expected?'#ffd166':'#42e4f5';
  return `<article><h4>${expected?'Expected from annotations':'Actually predicted'}</h4><div class="tr-graph-scroll"><svg role="img" aria-label="${expected?'Expected annotated':'Actual predicted'} temporal graph" viewBox="0 0 ${width} ${height}" style="min-width:${width}px"><defs><marker id="arrow-${kind}" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#acb9cb"/></marker></defs>${times.map((t,i)=>`<text class="tr-time-label" x="${60+i*(width-110)/Math.max(1,times.length-1)}" y="19" text-anchor="middle">t=${t}</text>`).join('')}${edges.map(e=>{const a=positions.get(e.source),b=positions.get(e.target);if(!a||!b)return '';const bad=expected?!e.present:e.status==='fp';return `<path d="M${a[0]+11},${a[1]} L${b[0]-13},${b[1]}" stroke="${bad?'#ff8895':'#88d9af'}" stroke-width="${e.focus?3:2}" fill="none" ${expected&&!e.present?'stroke-dasharray="5 4"':''} marker-end="url(#arrow-${kind})"/>`;}).join('')}${points.map(p=>{const [x,y]=positions.get(p.key);return `<g>${expected?`<path d="M${x},${y-7} l7,7 -7,7 -7,-7 Z"`:`<circle cx="${x}" cy="${y}" r="8"`} fill="#101925" stroke="${color}" stroke-width="2"/><text x="${x}" y="${y+24}" text-anchor="middle">${esc(p.label)}</text><title>${esc(p.key)} · t=${p.t}</title></g>`;}).join('')}</svg></div><div class="tr-edge-list">${edges.length?edges.map(e=>{const a=scene.points.find(p=>p.key===e.source),b=scene.points.find(p=>p.key===e.target);const verdict=expected?(e.present?'Linked':'Missing'):(e.status==='tp'?'Correct':e.status==='fp'?'Incorrect':'Unevaluated');return `<p class="${expected?!e.present?'tr-bad':'tr-good':e.status==='fp'?'tr-bad':e.status==='tp'?'tr-good':''}">${esc(a?.label)} → ${esc(b?.label)} <strong>${verdict}</strong>${e.focus?' · selected error':''}</p>`;}).join(''):'<p>No connections in this scene.</p>'}</div></article>`;
}
function renderCase(){
  const p=T.payload,c=p.case;$('t-case').hidden=false;$('t-title').textContent=titles[c.category];$('t-location').textContent=`${p.name} · ${c.dataset} · t=${c.t}`;$('t-explanation').textContent=p.explanation;
  const e=p.evidence;
  $('t-evidence').textContent=c.category==='edge_endpoint'?`Unmatched annotation endpoint${e.missing_gt.length===1?'':'s'}: ${e.missing_gt.join(', ')}. Inspect their detector candidates below.`:
    c.category==='edge_link'?'Both observations have official center matches. The predicted graph does not connect that assigned pair.':
    c.category==='edge_fp'?(e.competing.length?e.competing.map(x=>`Unmatched prediction ${x.prediction} is ${x.distance_um.toFixed(2)} µm from the expected annotation; prediction ${x.assigned_prediction} owns its assignment.`).join(' '):'This edge is an official false positive under the sparse annotation rules.'):
    c.category==='division_fn'?'The official local-window division scorer did not recover this annotated parent and both daughter lineages. This is not inferred from a missing immediate edge alone.':'The official scorer counted this predicted fork as a false division. Its relevant annotated and predicted branches are shown below.';
  $('t-graphs').innerHTML=graph('expected')+graph('actual');
  $('t-matching').textContent=`${p.matching}. Dashed red expected links are absent; solid green links are present. Actual edge verdicts use the whole-clip edge metric. “Unevaluated” means sparse labels cannot judge that edge.`;
  $('t-centers').innerHTML=p.center_checks.map(x=>`<tr><td>${esc(x.label)} · t=${x.t}</td><td>${x.matched?`Matched · ${x.final_distance_um.toFixed(2)} µm`:'No matched center'}</td><td>${x.raw_distance_um==null?'No candidate':x.raw_distance_um.toFixed(2)+' µm'}</td><td><a href="${detectionLink(x)}" target="_blank" rel="noopener">Inspect ${esc(x.label)} ↗</a></td></tr>`).join('');
  $('t-source').innerHTML=`<p>Case: <code>${esc(c.key)}</code></p><p>Overlay assignments: ${esc(p.matching)}. Center checks above use whole-clip matching; division-window assignments can differ.</p><ul>${p.scene.points.map(x=>`<li>${esc(x.label)} = ${esc(x.key)} · t=${x.t}, z/y/x = ${x.zyx.join(' / ')}</li>`).join('')}</ul><pre>${esc(JSON.stringify({evidence:e,provenance:p.provenance},null,2))}</pre>`;
  const note=notes[c.key]||{};$('t-verdict').value=note.verdict||'';$('t-notes').value=note.notes||'';$('t-note-status').textContent=note.updated?'Saved in this browser':'';
}
function detectionLink(x){return '/#'+new URLSearchParams({tab:'detection',review:'2',dmodel:T.payload.case.model,dfilter:'annotations',dclip:T.payload.case.dataset,dcase:x.detection_key});}
async function volume(dataset,t,shape){
  const key=`${dataset}:${t}`;if(cache.has(key))return cache.get(key);if(pending.has(key))return pending.get(key);
  const job=(async()=>{const r=await fetch('/api/detection/frame?'+new URLSearchParams({dataset,t}));if(!r.ok){const e=await r.json();throw Error(e.error);}
    const data=new Uint8Array(await new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());if(data.length!==shape.reduce((a,b)=>a*b,1))throw Error('Native frame dimensions do not match the clip.');
    cache.set(key,data);while(cache.size>8)cache.delete(cache.keys().next().value);return data;})();pending.set(key,job);try{return await job;}finally{pending.delete(key);}
}
const axes=()=>T.plane==='xy'?[2,1,0]:T.plane==='xz'?[2,0,1]:[1,0,2];
function frameGeometry(t){
  const p=T.payload,spacing=p.clip.spacing,shape=p.clip.shape.slice(1),points=p.scene.points,local=points.filter(x=>x.t===t),[a,b,h]=axes();
  const physical=points.map(x=>x.zyx.map((v,i)=>v*spacing[i]));
  const lows=[0,1,2].map(i=>Math.min(...physical.map(x=>x[i]))),highs=[0,1,2].map(i=>Math.max(...physical.map(x=>x[i])));
  const center=lows.map((v,i)=>(v+highs[i])/2),field=T.width==='auto'?Math.max(30,highs[a]-lows[a]+16,highs[b]-lows[b]+16):Number(T.width);
  const depth=T.projection==='mip'?[0,shape[h]-1]:[Math.max(0,Math.floor(Math.min(...local.map(x=>x.zyx[h]))-2)),Math.min(shape[h]-1,Math.ceil(Math.max(...local.map(x=>x.zyx[h]))+2))];
  return {a,b,h,shape,spacing,center,field,depth,points:local};
}
function drawImage(canvas,t,data){
  const g=frameGeometry(t),{a,b,h,shape,spacing,center,field,depth}=g,N=440;
  canvas.width=N;canvas.height=N;const ctx=canvas.getContext('2d'),img=ctx.createImageData(N,N),[Z,Y,X]=shape;
  const q=[0,0,0];
  for(let y=0;y<N;y++)for(let x=0;x<N;x++){
    q[a]=Math.round((center[a]+(x/(N-1)-.5)*field)/spacing[a]);q[b]=Math.round((center[b]+(y/(N-1)-.5)*field)/spacing[b]);let v=0;
    if(q[a]>=0&&q[a]<shape[a]&&q[b]>=0&&q[b]<shape[b])for(let d=depth[0];d<=depth[1];d++){q[h]=d;v=Math.max(v,data[(q[0]*Y+q[1])*X+q[2]]);}
    v=Math.min(255,Math.round(v*T.gamma));const i=(y*N+x)*4;img.data[i]=img.data[i+1]=img.data[i+2]=v;img.data[i+3]=255;
  }
  ctx.putImageData(img,0,0);
  if($('t-overlays').checked)for(const p of g.points){
    const x=(p.zyx[a]*spacing[a]-center[a]+field/2)/field*(N-1),y=(p.zyx[b]*spacing[b]-center[b]+field/2)/field*(N-1),gt=p.kind==='annotation';
    ctx.strokeStyle=gt?'#ffd166':'#42e4f5';ctx.lineWidth=2;ctx.beginPath();
    if(gt){ctx.moveTo(x,y-8);ctx.lineTo(x+8,y);ctx.lineTo(x,y+8);ctx.lineTo(x-8,y);ctx.closePath();}else ctx.arc(x,y,5,0,2*Math.PI);ctx.stroke();
    ctx.font='bold 13px system-ui';const ty=gt?y-13:y+19;ctx.lineWidth=3;ctx.strokeStyle='#080c12';ctx.strokeText(p.label,x+9,ty);ctx.fillStyle=gt?'#ffd166':'#42e4f5';ctx.fillText(p.label,x+9,ty);
  }
  ctx.fillStyle='#f0f3f8';ctx.fillRect(16,N-25,10/field*(N-1),3);ctx.font='12px system-ui';ctx.fillText('10 µm',16,N-32);
  canvas.dataset.frame=String(t);canvas.dataset.case=T.key;canvas.dataset.plane=T.plane;
  canvas.parentElement.querySelector('.tr-image-depth').textContent=`${'ZYX'[h]} ${depth[0]}–${depth[1]} · ${field.toFixed(1)} µm crop · ${g.points.length} event markers`;
}
async function renderImages(){
  if(!T.payload)return;const epoch=++T.imageEpoch,p=T.payload;const times=p.scene.times;
  $('t-loading').textContent='Loading event frames…';$('t-images').innerHTML=times.map(t=>`<article><h4>Frame ${t}${t===p.case.t?' · flagged time':''}</h4><canvas role="img" aria-label="Microscopy and event cells at frame ${t}" width="440" height="440"></canvas><p class="tr-image-depth">Loading exact frame…</p></article>`).join('');
  const slots=[...$('t-images').children];T.volumes=new Map();
  await Promise.all(times.map(async(t,i)=>{try{const data=await volume(p.case.dataset,t,p.clip.shape.slice(1));if(epoch!==T.imageEpoch)return;T.volumes.set(t,data);drawImage(slots[i].querySelector('canvas'),t,data);}catch(e){if(epoch===T.imageEpoch){slots[i].querySelector('canvas').hidden=true;slots[i].querySelector('p').textContent=`Frame unavailable: ${e.message}`;fail('t-error',e);}}}));
  if(epoch===T.imageEpoch)$('t-loading').textContent=`${T.volumes.size} / ${times.length} exact frames loaded`;
}
function redraw(){if(!T.payload)return;[...$('t-images').querySelectorAll('canvas')].forEach((c,i)=>{const t=T.payload.scene.times[i],v=T.volumes.get(t);if(v)drawImage(c,t,v);});hash();}
function saveNotes(silent=false){
  if(!T.payload||T.busy||$('t-case').hidden)return;const key=T.payload.case.key,verdict=$('t-verdict').value,text=$('t-notes').value,old=notes[key];
  if(!old&&!verdict&&!text)return;if(old?.verdict===verdict&&old?.notes===text)return;
  notes[key]={key,pipeline:T.payload.name,verdict,notes:text,updated:new Date().toISOString(),provenance:T.payload.provenance};
  try{localStorage.setItem(storageKey,JSON.stringify(notes));$('t-note-status').textContent='Saved in this browser';}catch{$('t-note-status').textContent='Storage unavailable; export notes to keep them.';}
  if(!silent)renderList();
}
async function activate(tab){
  if(T.active==='tracking'&&tab!=='tracking')saveNotes(true);T.active=tab;hash();
  try{await loadReport();if(T.active!==tab)return;
    if(tab==='report'){if(initial.get('rmodel')){$('r-model').value=initial.get('rmodel');initial.delete('rmodel');}if(initial.get('rembryo')){$('r-embryo').value=initial.get('rembryo');initial.delete('rembryo');}renderReport();}
    else if(tab==='tracking'&&!initialized){
      if(!T.report.models[T.model])T.model='pooled';if(!['all',...Object.keys(titles)].includes(T.filter))T.filter='all';if(!T.report.stages[T.stage])T.stage='';
      if(!['xy','xz','yz'].includes(T.plane))T.plane='xy';if(!['auto','30','50','104'].includes(T.width))T.width='auto';if(!['local','mip'].includes(T.projection))T.projection='local';
      $('t-stage').innerHTML='<option value="">All evidence groups</option>'+Object.entries(T.report.stages).map(([k,v])=>`<option value="${k}">${esc(v)}</option>`).join('');
      initialized=true;syncControls();await loadCases();
    }
  }catch(e){fail(tab==='report'?'r-error':'t-error',e);}
}
$('r-model').onchange=$('r-embryo').onchange=()=>{if(T.report)renderReport();};
for(const [id,prop] of [['t-model','model'],['t-filter','filter'],['t-stage','stage'],['t-embryo','embryo'],['t-clip','dataset']])$(id).onchange=()=>{saveNotes(true);T[prop]=$(id).value;T.offset=0;T.key=null;if(id==='t-filter')T.stage='';syncControls();loadCases(null);};
$('t-reset').onclick=()=>{T.filter='all';T.stage='';T.embryo='';T.dataset='';T.offset=0;syncControls();loadCases(null);};
$('t-report').onclick=()=>{$('r-model').value=T.model;$('r-embryo').value=T.embryo||'all';window.switchReviewTab('report');};
$('t-prev').onclick=()=>jump(T.offset+T.rows.findIndex(r=>r.key===T.key)-1);$('t-next').onclick=()=>jump(T.offset+T.rows.findIndex(r=>r.key===T.key)+1);
$('t-go').onclick=()=>jump(Number($('t-jump').value)-1);$('t-jump').onkeydown=e=>{if(e.key==='Enter')$('t-go').click();};
$('t-page-prev').onclick=()=>jump(T.offset-T.limit);$('t-page-next').onclick=()=>jump(T.offset+T.limit);
for(const [id,prop] of [['t-plane','plane'],['t-projection','projection'],['t-width','width']])$(id).onchange=()=>{T[prop]=$(id).value;redraw();};
$('t-fit').onclick=()=>{T.width='auto';$('t-width').value='auto';redraw();};$('t-brightness').oninput=()=>{T.gamma=Number($('t-brightness').value);redraw();};$('t-overlays').onchange=redraw;
$('t-copy').onclick=async()=>{hash();try{await navigator.clipboard.writeText(location.href);$('t-note-status').textContent='Case link copied';}catch{$('t-note-status').textContent='Copy this page’s URL to share the case.';}};
$('t-save').onclick=()=>saveNotes();$('t-verdict').onchange=()=>saveNotes();$('t-notes').onchange=()=>saveNotes();
$('t-export').onclick=()=>{saveNotes();const url=URL.createObjectURL(new Blob([JSON.stringify(notes,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='tracking-review-notes.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
window.pipelineReview={activate,deactivate(){saveNotes(true);T.active=null;},get state(){return T;},frameGeometry,drawImage};
})();

"""Offline, label-blind viewers for independent image census and candidate audits."""
from __future__ import annotations

import base64
import io
import json

import numpy as np
import pandas as pd
from PIL import Image
import zarr

from .common import DATA,OUT,SEED,load_graph,now,read_json,sha,write_json


def candidate_pack():
    root=OUT/'blinded_candidate_audit';root.mkdir(exist_ok=True)
    design=root/'candidate_design_and_blank_labels.csv'
    if design.exists():return
    rng=np.random.default_rng(SEED+420);pools={};counts={}
    for name in read_json(OUT/'fold_manifest.json')['expected_samples']:
        b=load_graph(OUT/'baseline/clean'/f'{name}.npz');f=b['features']
        quality=f[:,5]>=.05;persistent=f[:,10]>=3
        for high in [False,True]:
            for long in [False,True]:
                ix=np.flatnonzero((quality==high)&(persistent==long))
                key=(name.split('_')[0],int(high),int(long));counts[key]=counts.get(key,0)+len(ix)
                if not len(ix):continue
                priority=rng.random(len(ix));pick=np.argsort(priority)[:6]
                pool=pools.setdefault(key,[])
                pool.extend((float(priority[j]),name,b['nodes'][ix[j]].tolist()) for j in pick)
                pools[key]=sorted(pool,key=lambda x:x[0])[:6]
    chosen=[(key,item) for key,pool in sorted(pools.items()) for item in pool]
    rng.shuffle(chosen);rows=[]
    for i,(key,(_,name,node)) in enumerate(chosen):
        candidate_id,t,z,y,x=node
        arr=zarr.open_group(DATA/'train'/f'{name}.zarr',mode='r')['0'];im=arr[t]
        center=np.array([z,y,x]);halo=np.array([8,32,32])
        lo=np.maximum(center-halo,0);hi=np.minimum(center+halo+1,im.shape)
        crop=im[tuple(slice(a,b) for a,b in zip(lo,hi))];audit_id=f'candidate_{i:03d}'
        np.savez_compressed(root/(audit_id+'.npz'),image=crop,center=center-lo)
        rows.append(dict(audit_id=audit_id,dataset=name,candidate_id=candidate_id,t=t,z=z,y=y,x=x,
                         embryo=key[0],response_at_least_005=key[1],persistence_at_least_3=key[2],
                         sampling_population=counts[key],inclusion_probability=len(pools[key])/counts[key],
                         real_cell=None,ambiguous=None,second_review=None))
    pd.DataFrame(rows).to_csv(design,index=False)
    write_json(root/'sampling.json',dict(created=now(),samples=len(rows),seed=SEED+420,
               design='Uniform without replacement within embryo x image-response x predicted-persistence strata; six per nonempty stratum',
               labels_used=False,selector_keep_decisions_used=False,manual_labels_received=0,
               role='Post-lock audit material only; never selector training or outcome-dependent tuning'))
    (root/'README.md').write_text('Blinded candidate precision audit\n\nOpen viewer.html offline. Decide whether the marked predicted center corresponds to a real cell, a false detection, or an ambiguous object. The viewer hides sparse GT, model keep decisions and quality strata. Browse z slices for context. Fill the separate CSV or export labels from the viewer. These labels are never used for selector fitting. A precision audit cannot count cells missed by the detector. No manual judgments have been supplied.\n')


def encode_stack(path,kind):
    with np.load(path) as f:
        image=f['image'];mark=f['center'].tolist() if kind=='candidate' else f['core_start'].tolist()
        shape=None if kind=='candidate' else f['core_shape'].tolist()
    q=max(float(np.quantile(image,.999)),1.)
    pixels=np.clip(image/q*255,0,255).astype(np.uint8);frames=[]
    for frame in pixels:
        stream=io.BytesIO();Image.fromarray(frame).save(stream,format='PNG')
        frames.append('data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode())
    return dict(id=path.stem,kind=kind,frames=frames,mark=mark,shape=shape,
                width=image.shape[2],height=image.shape[1],display_upper_quantile=q)


def run(args):
    candidate_pack()
    for directory,kind in [('blinded_census','census'),('blinded_candidate_audit','candidate')]:
        root=OUT/directory;payload=[encode_stack(p,kind) for p in sorted(root.glob('*.npz'))]
        data=json.dumps(payload,separators=(',',':')).replace('</','<\\/')
        (root/'viewer.html').write_text(HTML.replace('__DATA__',data).replace('__KIND__',kind))
    audit=read_json(OUT/'quality_audit.json')
    audit.update(candidate_precision_audit='48 stratified predicted-center image crops; no human labels supplied',
                 candidate_audit_pack='blinded_candidate_audit',offline_viewers=True,
                 viewers_created=now(),candidate_audit_added_after_lock_without_labels=True)
    write_json(OUT/'quality_audit.json',audit)
    write_json(OUT/'audit_viewer_receipt.json',dict(created=now(),manual_labels_fabricated=False,
               files={str(p.relative_to(OUT)):sha(p) for d in ['blinded_census','blinded_candidate_audit']
                      for p in (OUT/d).glob('viewer.html')}))
    print('Exported two offline viewers: 48 census ROIs and 48 stratified candidate crops; no manual labels.',flush=True)


HTML='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Blinded __KIND__ audit</title>
<style>body{background:#17232e;color:#ecf1f5;font:16px/1.5 system-ui;max-width:1000px;margin:auto;padding:24px}main{display:grid;grid-template-columns:1.2fr 1fr;gap:24px}canvas{width:100%;image-rendering:pixelated;background:#000}button,select,input{font:inherit;padding:6px;margin:4px}label{display:block;margin:14px 0}input[type=range]{width:90%}.muted{color:#b6c6d0}@media(max-width:700px){main{display:block}}</style>
<h1>Blinded __KIND__ audit</h1><p>Sparse GT, selector scores, keep decisions and sampling strata are hidden. Raw stacks remain in the adjacent NPZ files. Display uses each crop's 99.9th intensity percentile; original pixels are preserved.</p>
<button id="prev">Previous</button><select id="sample"></select><button id="next">Next</button>
<main><section><canvas id="image" width="512" height="512"></canvas><label>Z slice <span id="zlabel"></span><input id="z" type="range" min="0"></label></section><section><h2 id="id"></h2><p id="instructions"></p><p class="muted">Voxel spacing: Z 1.625 µm, Y/X 0.40625 µm. The core boundary or predicted center is shown in gold. Use the surrounding image for context.</p><label id="answer-label">Assessment <input id="answer" placeholder="Count, or cell / false / uncertain"></label><label>Ambiguous objects or notes <input id="ambiguous"></label><label>Second reviewer <input id="reviewer"></label><button id="download">Export entered labels</button><p class="muted">Labels stay in this tab until exported. Blank fields are missing judgments, never zero counts. The automated study does not infer these answers.</p></section></main>
<script type="application/json" id="data">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent),$=x=>document.getElementById(x),answers={};let current=0,token=0;
D.forEach((r,i)=>$('sample').add(new Option(r.id,i)));
function save(){answers[D[current].id]={assessment:$('answer').value,ambiguous:$('ambiguous').value,second_review:$('reviewer').value}}
function draw(){const r=D[current],z=+$('z').value,k=++token,im=new Image();$('zlabel').textContent=`${z+1} / ${r.frames.length}`;im.onload=()=>{if(k!==token)return;const c=$('image'),ctx=c.getContext('2d');c.width=r.width*8;c.height=r.height*8;ctx.imageSmoothingEnabled=false;ctx.drawImage(im,0,0,c.width,c.height);ctx.strokeStyle='#ffd477';ctx.lineWidth=2;const [mz,my,mx]=r.mark;if(r.kind==='census'&&z>=mz&&z<mz+r.shape[0])ctx.strokeRect(mx*8,my*8,r.shape[2]*8,r.shape[1]*8);if(r.kind==='candidate'&&z===mz){ctx.beginPath();ctx.moveTo(mx*8-12,my*8);ctx.lineTo(mx*8+12,my*8);ctx.moveTo(mx*8,my*8-12);ctx.lineTo(mx*8,my*8+12);ctx.stroke()}};im.src=r.frames[z]}
function select(i){save();current=(i+D.length)%D.length;const r=D[current],a=answers[r.id]||{};$('sample').value=current;$('id').textContent=r.id;$('z').max=r.frames.length-1;$('z').value=r.kind==='candidate'?r.mark[0]:Math.min(r.frames.length-1,r.mark[0]+Math.floor(r.shape[0]/2));$('answer').value=a.assessment||'';$('ambiguous').value=a.ambiguous||'';$('reviewer').value=a.second_review||'';$('instructions').textContent=r.kind==='census'?'Count every visible cell center strictly inside the gold 3D core. Include a center only once across z slices. Mark ambiguous centers separately.':'Judge whether the marked predicted center corresponds to a real cell. Browse nearby slices; use cell, false, or uncertain.';draw()}
$('sample').onchange=()=>select(+$('sample').value);$('prev').onclick=()=>select(current-1);$('next').onclick=()=>select(current+1);$('z').oninput=draw;$('download').onclick=()=>{save();const b=new Blob([JSON.stringify(answers,null,2)],{type:'application/json'}),u=URL.createObjectURL(b),a=document.createElement('a');a.href=u;a.download='blinded-audit-labels.json';a.click();URL.revokeObjectURL(u)};select(0);
</script></html>'''

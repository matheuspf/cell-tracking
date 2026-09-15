/* Focused failure images: one real timepoint per image, with a shared physical crop. */
"use strict";
window.errorCaseView = (() => {
  const $ = id => document.getElementById(id);
  const esc = value => String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const ink = {annotation:'#ffffff', prediction:'#64e2f3', missing:'#ff879b', offset:'#ffcb69', wrong:'#ed9cff'};
  let token=0, signature='', current=null;

  function crop(scene,shape,spacing,plane) {
    const axes=plane==='xy'?[2,1,0]:plane==='xz'?[2,0,1]:[1,0,2], [h,v,d]=axes;
    const positions=scene.points.map(p=>p.zyx);
    const range=axis=>positions.length?[Math.min(...positions.map(p=>p[axis]*spacing[axis])),Math.max(...positions.map(p=>p[axis]*spacing[axis]))]:[0,shape[axis]*spacing[axis]];
    const [xmin,xmax]=range(h),[ymin,ymax]=range(v);
    const side=Math.max(26,xmax-xmin+16,ymax-ymin+16);
    const u0=Math.floor(((xmin+xmax)/2-side*.65)/spacing[h]);
    const v0=Math.floor(((ymin+ymax)/2-side*.5)/spacing[v]);
    return {h,v,d,u0,v0,width:Math.ceil(side*1.3/spacing[h]),height:Math.ceil(side/spacing[v])};
  }
  function project(volume,scene,time,shape,geometry,local) {
    const {h,v,d,u0,v0,width,height}=geometry;
    const points=scene.points.filter(p=>p.t===time);
    const depths=points.map(p=>p.zyx[d]);
    const low=local&&depths.length?Math.max(0,Math.floor(Math.min(...depths))-2):0;
    const high=local&&depths.length?Math.min(shape[d]-1,Math.ceil(Math.max(...depths))+2):shape[d]-1;
    const gray=new Uint8Array(width*height),strides=[shape[1]*shape[2],shape[2],1];
    for(let y=0;y<height;y++)for(let x=0;x<width;x++) {
      if(x+u0<0||x+u0>=shape[h]||y+v0<0||y+v0>=shape[v])continue;
      const start=(x+u0)*strides[h]+(y+v0)*strides[v];let value=0;
      for(let z=low;z<=high;z++)value=Math.max(value,volume.gray[start+z*strides[d]]);
      gray[y*width+x]=value;
    }
    return {gray,low,high,...geometry};
  }
  function marker(ctx,x,y,kind,color,label) {
    const r=kind==='annotation'?10:7;
    ctx.save();ctx.strokeStyle='#071018';ctx.lineWidth=6;ctx.beginPath();
    if(kind==='annotation'){ctx.moveTo(x,y-r);ctx.lineTo(x+r,y);ctx.lineTo(x,y+r);ctx.lineTo(x-r,y);ctx.closePath();}
    else ctx.arc(x,y,r,0,Math.PI*2);
    ctx.stroke();ctx.strokeStyle=color;ctx.lineWidth=2.5;ctx.stroke();
    ctx.font='600 18px system-ui';ctx.lineWidth=4;ctx.strokeStyle='#071018';ctx.fillStyle=color;
    const dy=kind==='annotation'?-13:22;
    const lx=Math.max(8,Math.min(ctx.canvas.width-ctx.measureText(label).width-8,x+13));
    const ly=Math.max(20,Math.min(ctx.canvas.height-10,y+dy));
    ctx.strokeText(label,lx,ly);ctx.fillText(label,lx,ly);ctx.restore();
  }
  function draw(canvas,tile,event,time,spacing,gamma,side,frame,isolate) {
    const ctx=canvas.getContext('2d'),{width,height,h,v,u0,v0}=tile;
    const bitmap=document.createElement('canvas');bitmap.width=width;bitmap.height=height;
    const bctx=bitmap.getContext('2d'),pixels=bctx.createImageData(width,height);
    for(let i=0;i<tile.gray.length;i++) {
      const value=Math.round(255*Math.pow(tile.gray[i]/255,1/gamma));
      pixels.data.set([value,value,value,255],i*4);
    }
    bctx.putImageData(pixels,0,0);
    const scale=Math.min(canvas.width/(width*spacing[h]),canvas.height/(height*spacing[v]));
    const ox=(canvas.width-width*spacing[h]*scale)/2,oy=(canvas.height-height*spacing[v]*scale)/2;
    const xy=p=>[ox+(p.zyx[h]-u0+.5)*spacing[h]*scale,oy+(p.zyx[v]-v0+.5)*spacing[v]*scale];
    ctx.fillStyle='#070c12';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.imageSmoothingEnabled=false;
    ctx.drawImage(bitmap,ox,oy,width*spacing[h]*scale,height*spacing[v]*scale);
    if(!isolate&&frame) {
      ctx.save();ctx.globalAlpha=.35;ctx.strokeStyle=ink.prediction;ctx.lineWidth=1;
      for(const row of frame.points[event.pipeline]||[]) {
        if(row[tile.d+1]<tile.low||row[tile.d+1]>tile.high)continue;
        const [x,y]=xy({zyx:row.slice(1,4)});ctx.beginPath();ctx.arc(x,y,3,0,Math.PI*2);ctx.stroke();
      }ctx.restore();
    }
    const points=event.scene.points.filter(p=>p.t===time&&(side!=='annotation'||p.kind==='annotation'));
    if(side!=='annotation')for(const cell of event.scene.cells.filter(c=>c.t===time)) {
      const a=points.find(p=>p.key===cell.annotation),b=points.find(p=>p.key===cell.prediction);
      if(!a)continue;
      const [x,y]=xy(a);ctx.save();
      if(b) {
        const [bx,by]=xy(b);ctx.strokeStyle=cell.distance_um>=3?ink.offset:ink.prediction;ctx.lineWidth=2;ctx.setLineDash([3,4]);
        ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(bx,by);ctx.stroke();
      } else {
        // Circle is a physical radius in the displayed plane, not a 3D sphere
        // projection. Assignment decisions remain full 3D, as the legend states.
        ctx.strokeStyle=ink.missing;ctx.lineWidth=1.8;ctx.setLineDash([7,5]);
        ctx.beginPath();ctx.arc(x,y,event.radius_um*scale,0,Math.PI*2);ctx.stroke();
        ctx.fillStyle=ink.missing;ctx.font='600 14px system-ui';ctx.strokeStyle='#071018';ctx.lineWidth=4;ctx.setLineDash([]);
        // Put the verdict outside the radius so it cannot cover a nearby
        // prediction label or the annotated center itself.
        const labelX=Math.max(8,Math.min(canvas.width-ctx.measureText('NO MATCH').width-8,x-35));
        const labelY=Math.max(20,y-event.radius_um*scale-10);
        ctx.strokeText('NO MATCH',labelX,labelY);ctx.fillText('NO MATCH',labelX,labelY);
      }ctx.restore();
    }
    // Annotation labels above, prediction labels below: centers can coincide
    // without making the two labels unreadable.
    for(const p of points) {
      const [x,y]=xy(p),missing=p.kind==='annotation'&&p.role==='missing';
      const wrong=event.scene.actual.some(e=>e.status==='fp'&&(e.source===p.key||e.target===p.key));
      marker(ctx,x,y,p.kind,missing?ink.missing:p.kind==='annotation'?ink.annotation:wrong?ink.wrong:ink.prediction,p.label);
    }
    const length=5,bar=length*scale;
    ctx.fillStyle='rgba(5,10,16,.9)';ctx.fillRect(10,canvas.height-41,bar+20,32);
    ctx.strokeStyle='white';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(20,canvas.height-16);ctx.lineTo(20+bar,canvas.height-16);ctx.stroke();
    ctx.fillStyle='white';ctx.font='13px system-ui';ctx.fillText('5 µm',20,canvas.height-25);
  }
  function nodeBadge(p,kind) {return `<span class="flow-node ${kind}">${p.kind==='annotation'?'◇':'○'} ${esc(p.label)} <small>frame ${p.t}</small></span>`;}
  function connections(event) {
    const lookup=new Map(event.scene.points.map(p=>[p.key,p]));
    if(event.category.startsWith('center_')) {
      const cell=event.scene.cells[0],a=lookup.get(cell.annotation),b=lookup.get(cell.prediction);
      return `<div class="connection-row"><strong>Center assignment<br><small>Same frame, ${event.radius_um} µm radius</small></strong><div class="flows">${nodeBadge(a,'expected')}<span class="flow-arrow ${b?'':'broken'}">${b?'↔':'×'}</span>${b?nodeBadge(b,'actual'):'<span class="no-connection">No assigned center</span>'}<span>${b?`${cell.distance_um.toFixed(2)} µm offset`:event.nearest_um==null?'No prediction in this frame':`Nearest: ${event.nearest_um.toFixed(2)} µm`}</span></div></div>`;
    }
    const row=(title,edges,type)=>`<div class="connection-row"><strong>${title}</strong><div class="flows">${edges.length?edges.map(e=>`<div class="flow-path">${nodeBadge(lookup.get(e.source),type)}<span class="flow-arrow ${e.status==='fp'?'wrong':e.status==='expected'&&!e.recovered?'broken':''}">${e.status==='expected'?'⇢':'→'}</span>${nodeBadge(lookup.get(e.target),type)}${e.status==='fp'?'<span class="flow-result wrong">Incorrect</span>':e.status==='unknown'?'<span class="flow-result">Unscored</span>':e.status==='tp'?'<span class="flow-result correct">✓ Correct</span>':e.status==='expected'?e.recovered?'<span class="flow-result correct">✓ Recovered</span>':'<span class="flow-result missing">× Missing</span>':''}</div>`).join(''):'<span class="no-connection">× No predicted connection between these cells</span>'}</div></div>`;
    return row('Expected from annotations',event.scene.expected,'expected')+row('Actually predicted',event.scene.actual,'actual');
  }
  async function update(options) {
    const {event,shape,spacing,plane,gamma,isolate,local}=options;
    if(!event){clear();return;}
    const next=JSON.stringify([event.pipeline,event.id,plane,gamma,isolate,local]);
    if(next===signature)return;
    signature=next;const epoch=++token;current=null;
    $('case-visual').hidden=false;$('case-loading').textContent='Loading the relevant frames…';
    const center=event.category.startsWith('center_');
    const views=center?[{t:event.t,side:'annotation',title:'Annotated center'},{t:event.t,side:'prediction',title:'Model result · same frame'}]
      :event.scene.times.map((t,i)=>({t,side:'both',title:i===0?'Earlier frame':'Following frame'}));
    const geometry=crop(event.scene,shape,spacing,plane);
    $('case-images').innerHTML=views.map(view=>`<article class="case-frame"><header><strong>${view.title}</strong><span>Frame ${view.t}</span></header><div class="case-image-slot" data-case-time="${view.t}"><p class="case-placeholder">${options.available(view.t)?'Loading image…':'Image not exported for this frame. The graph relationship is still shown below.'}</p></div><footer></footer></article>`).join('');
    $('case-connections').innerHTML=connections(event);
    $('case-legend').innerHTML='<span>◇ Annotation A, B…</span><span class="prediction-legend">○ Pred A = matched center for A</span><span class="missing-legend">Dashed ring / NO MATCH = no assigned center</span><span class="wrong-legend">Purple centers belong to an incorrect link</span><span>Matching uses 3D distance; ring shows the radius in this plane.</span>';
    try {
      const frames=await Promise.all(views.map(async view=>options.available(view.t)?await options.load(view.t):null));
      if(epoch!==token)return;
      const tiles=[];
      views.forEach((view,i)=>{
        const slot=$('case-images').children[i];
        if(!frames[i])return;
        const tile=project(frames[i].volume,event.scene,view.t,shape,geometry,local);
        const canvas=document.createElement('canvas');canvas.width=520;canvas.height=400;
        canvas.dataset.frame=String(view.t);canvas.dataset.imageKey=frames[i].key;
        canvas.setAttribute('role','img');canvas.setAttribute('aria-label',`${view.title}, frame ${view.t}, ${event.title}`);
        draw(canvas,tile,event,view.t,spacing,gamma,view.side,frames[i].frame,isolate);
        slot.querySelector('.case-image-slot').replaceChildren(canvas);
        slot.querySelector('footer').textContent=`${plane.toUpperCase()} · ${'ZYX'[geometry.d]} ${tile.low}–${tile.high} · ${local?'depth around these cells':'full-depth maximum'} · same physical crop`;
        tiles.push({t:view.t,key:frames[i].key,side:view.side,...tile});
      });
      current={id:event.id,pipeline:event.pipeline,tiles};
      $('case-loading').textContent='';
    } catch(error) {
      if(epoch!==token)return;
      $('case-loading').textContent=`Could not load this comparison: ${error.message}`;
    }
  }
  function clear(){token++;signature='';current=null;$('case-visual').hidden=true;$('case-loading').textContent='';}
  return {update,clear,get current(){return current;}};
})();

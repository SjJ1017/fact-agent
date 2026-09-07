(()=>{
const N=MENTOR_DATA.network;
function byTask(rs,key){const g={};for(const r of rs)(g[r.case_id]??=[]).push(r[key]);return avg(Object.values(g).map(avg))}
const cLabel={'full-generic':'全信息 · generic','full-specialist':'全信息 · specialist','split-generic':'分信息 · generic','split-specialist-aligned':'分信息 · specialist'};
function io(){const rr=+$('io-round').value;let rows=[];for(const c of cells)for(const a of 'ABC'){
 const rs=D.turns.filter(r=>r.condition==c&&r.agent==a&&r.round==rr),inp=rs.filter(r=>r.scope=='input'),out=rs.filter(r=>r.scope=='output');
 rows.push([cLabel[c],a,...'BCXGU'.split('').map(t=>{const i=avg(inp.map(r=>r[t+'_share'])),o=avg(out.map(r=>r[t+'_share']));return `<span class="io-value" style="color:${colors[t]}">${pct(i)} → <b>${pct(o)}</b></span><div class="small">${i!==null&&o!==null?pp(o-i):'无初始输入'}</div>`})]);
} $('io-table').innerHTML=table(['条件','Agent',...Object.values(names)],rows);}
$('io-round').addEventListener('change',io);io();
const geoRows=[];for(const c of cells)for(const scope of ['input','output'])geoRows.push([cLabel[c],scope=='input'?'输入 JSD':'输出 JSD',...[1,2,3].map(rr=>{const v=avg(N.geometry.filter(r=>r.condition==c&&r.scope==scope&&r.round==rr).map(r=>r.js));return v===null?'—':v.toFixed(3)})]);
$('geometry').innerHTML=table(['条件','对象','R1','R2','R3'],geoRows);
$('redundancy').innerHTML=table(['信息 / 主题','候选来源数','generic','specialist','配对差 [95% CI]'],N.contrasts.filter(r=>r.topic=='X'&&r.metric!='multi_among_related').map(r=>[r.info+' / 双侧',r.metric=='one_sender'?'仅一位':'两位',pct(r.generic),pct(r.specialist),`${pp(r.mean)} [${pp(r.ci95[0])}, ${pp(r.ci95[1])}]`]));
const gs=$('graph-condition');gs.innerHTML=cells.map(c=>`<option value="${c}">${cLabel[c]}</option>`).join('');gs.value='split-specialist-aligned';
$('graph-case').innerHTML+=[...new Set(N.graphs.map(g=>g.case_id))].map(c=>`<option>${c}</option>`).join('');
$('graph-kind').innerHTML+='<option value="peer_exclusive">仅此发送者关联的输出</option>';
const positions={A:[220,185],B:[760,185],C:[490,455]};
const defs=id=>`<defs><marker id="${id}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10z" fill="#1f7a6b"/></marker><marker id="${id}-gray" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10z" fill="#90959c"/></marker></defs>`;
function ring(x,y,r,props,label,sub){let offset=0;const length=2*Math.PI*r;let s=`<circle cx="${x}" cy="${y}" r="${r}" fill="var(--panel)" stroke="var(--edge)" stroke-width="14"/>`;
 for(const t of 'BCXGU'){const v=props[t]||0;if(v){s+=`<circle cx="${x}" cy="${y}" r="${r}" fill="none" stroke="${colors[t]}" stroke-width="14" stroke-dasharray="${v*length} ${length}" stroke-dashoffset="${-offset*length}" transform="rotate(-90 ${x} ${y})"><title>${names[t]} ${pct(v)}</title></circle>`;offset+=v;}}
 return s+`<text x="${x}" y="${y+5}" font-size="26" fill="var(--ink)" text-anchor="middle">${label}</text><text x="${x}" y="${y+r+34}" font-size="15" fill="var(--dim)" text-anchor="middle">${sub}</text>`;}
function curved(a,b,n,id,label,color='#1f7a6b',radius=57){const [x1,y1]=a,[x2,y2]=b,dx=x2-x1,dy=y2-y1,len=Math.hypot(dx,dy),ux=dx/len,uy=dy/len,nx=-uy,ny=ux;
 const sx=x1+ux*radius,sy=y1+uy*radius,ex=x2-ux*(radius+6),ey=y2-uy*(radius+6),cx=(sx+ex)/2+nx*50,cy=(sy+ey)/2+ny*50;
 const w=Math.min(13,1.2+n*.45),tx=(sx+2*cx+ex)/4+nx*15,ty=(sy+2*cy+ey)/4+ny*15;
 return `<path d="M${sx} ${sy} Q${cx} ${cy} ${ex} ${ey}" fill="none" stroke="${color}" stroke-width="${w}" opacity=".8" marker-end="url(#${id})"><title>${esc(label)}：${n.toFixed(1)} 条/场</title></path><text x="${tx}" y="${ty}" text-anchor="middle" dominant-baseline="middle" font-size="16" font-weight="600" fill="var(--ink)" stroke="var(--panel)" stroke-width="5" paint-order="stroke">${n.toFixed(1)}</text>`;
}
function selectedGraphs(){return N.graphs.filter(g=>g.condition==gs.value&&($('graph-case').value=='*'||g.case_id==$('graph-case').value))}
function edgeMean(g,source,target,kind,topic){
 if(kind=='peer_exclusive') {const a=source.split('|')[0],b=target.split('|')[0],rr=+target.split('|')[1];const rows=N.rows.filter(r=>g.some(x=>x.execution_id==r.execution_id)&&r.agent==b&&r.round==rr&&r.topic==topic);return avg(rows.map(r=>r.n*(r['exclusive_'+a]||0)))||0;}
 return avg(g.map(x=>x.edges.filter(e=>e.source==source&&e.target==target&&e.kind==kind&&(topic=='*'||e.topic==topic)).reduce((s,e)=>s+e.n,0)))||0;
}
function nodeSummary(g,a,rr){const ns=g.map(x=>x.nodes.find(n=>n.id==a+'|'+rr)).filter(Boolean),n=avg(ns.map(x=>x.n))||0,props={};for(const t of 'BCXGU')props[t]=avg(ns.filter(x=>x.n).map(x=>(x.topics[t]||0)/x.n))||0;return {n,props,unmatched:avg(ns.map(x=>x.unmatched))||0}}
function drawGraph(){const gg=selectedGraphs(),rr=+$('graph-round').value,kind=$('graph-kind').value,topic=$('graph-topic').value;let svg=`<svg viewBox="0 0 980 700" role="img" aria-label="IDRBench 有向内容图">${defs('g-arrow')}`;
 // Materials are separate nodes, with direct relations to currently visible allocated input.
 for(const [doc,x] of [['paper-b',70],['paper-c',910]]){
  svg+=`<rect x="${x-60}" y="15" width="120" height="42" rx="6" fill="var(--panel)" stroke="var(--edge)"/><text x="${x}" y="42" text-anchor="middle" fill="var(--ink)" font-size="15">${doc=='paper-b'?'源材料 B':'源材料 C'}</text>`;
  for(const a of 'ABC'){let n=edgeMean(gg,'src:'+doc,a+'|'+rr,'source_related',topic);if(n>.001){const [ax,ay]=positions[a];svg+=`<path d="M${x} 58 Q${x} ${ay-65} ${ax} ${ay-62}" stroke="#90959c" stroke-width="${1+n*.4}" opacity=".55" fill="none" marker-end="url(#g-arrow-gray)"><title>${doc} → ${a}：${n.toFixed(1)} 条直接相关输出/场</title></path>`;}}
 }
 for(const a of 'ABC')for(const b of 'ABC')if(a!=b&&rr>1){const n=edgeMean(gg,a+'|'+(rr-1),b+'|'+rr,kind,topic);if(n>.001)svg+=curved(positions[a],positions[b],n,'g-arrow',a+' → '+b);}
 for(const a of 'ABC'){
  const [x,y]=positions[a],v=nodeSummary(gg,a,rr);svg+=ring(x,y,47,v.props,a,`${v.n.toFixed(1)} 条输出 / 场`);
  const nx=a=='A'?70:a=='B'?910:490,ny=a=='C'?640:350;
  let n=avg(gg.map(g=>g.edges.filter(e=>e.source=='new:'+a&&e.target==a+'|'+rr&&(topic=='*'||e.topic==topic)).reduce((s,e)=>s+e.n,0)))||0;
  svg+=`<rect x="${nx-66}" y="${ny-24}" width="132" height="49" rx="5" fill="var(--panel)" stroke="#90959c" stroke-dasharray="5 4"/><text x="${nx}" y="${ny-3}" text-anchor="middle" fill="var(--ink)" font-size="13">${a} 的未匹配产出</text><text x="${nx}" y="${ny+15}" text-anchor="middle" fill="var(--dim)" font-size="13">${n.toFixed(1)} 条 / 场</text><path d="M${nx} ${ny-25} Q${nx} ${y+75} ${x} ${y+60}" stroke="#90959c" stroke-width="${1+n*.18}" stroke-dasharray="5 4" opacity=".6" fill="none" marker-end="url(#g-arrow-gray)"/>`;
  if(rr>1){const self=edgeMean(gg,a+'|'+(rr-1),a+'|'+rr,'self_equivalent',topic);if(self){const sign=a=='B'?1:-1;svg+=`<path d="M${x+sign*40} ${y-35} C${x+sign*115} ${y-120} ${x+sign*115} ${y+65} ${x+sign*54} ${y+12}" stroke="#90959c" stroke-width="${1+self*.3}" fill="none" marker-end="url(#g-arrow-gray)"/><text x="${x+sign*105}" y="${y-5}" text-anchor="middle" fill="var(--dim)" font-size="13">自持 ${self.toFixed(1)}</text>`;}}
 }
 svg+='<text x="490" y="690" text-anchor="middle" fill="var(--dim)" font-size="13">绿箭头：相邻轮同伴关系　灰箭头：材料 / 自持　虚线：未匹配产出　圆环：输出主题</text></svg>';
 $('agent-graph').innerHTML=svg;
 $('graph-caption').innerHTML=`${cLabel[gs.value]} · R${rr} · ${gg.length} 个任务${rr==1?'。首轮尚无同伴投递，因此没有跨 agent 绿色边。':'；绿色边从 R'+(rr-1)+' 的发送者指向 R'+rr+' 的接收者。'}主题筛选作用于边的目标事实；圆环始终显示完整输出分布。<div class="legend">${Object.keys(colors).map(t=>`<span><i class="dot" style="background:${colors[t]}"></i>${names[t]}</span>`).join('')}</div>`;
 drawTime(gg,kind,topic);
 $('graph-text').innerHTML=gg.length==1?'<p>'+esc(JSON.stringify(gg[0].roles))+'</p>'+gg[0].nodes.map(n=>`<details><summary>${n.id} · ${n.n} 条事实</summary><pre>${esc(n.text)}</pre></details>`).join(''):'<p>选择一个具体任务即可查看九个完整原始 turn。</p>';
}
function drawTime(gg,kind,topic){let s=`<svg viewBox="0 0 1040 590" role="img" aria-label="按轮展开的有向事实关系">${defs('t-arrow')}`;const pos=(a,r)=>[160+(r-1)*360,105+'ABC'.indexOf(a)*190];
 for(let r=1;r<=2;r++)for(const a of 'ABC')for(const b of 'ABC'){
  const ty=a==b?'self_equivalent':kind,n=edgeMean(gg,a+'|'+r,b+'|'+(r+1),ty,topic);if(!n)continue;
  const [x1,y1]=pos(a,r),[x2,y2]=pos(b,r),col=a==b?'#90959c':'#1f7a6b';
  s+=`<path d="M${x1+45} ${y1} C${x1+155} ${y1} ${x2-155} ${y2} ${x2-48} ${y2}" fill="none" stroke="${col}" stroke-width="${Math.min(10,1+n*.4)}" opacity=".55" marker-end="url(#t-arrow${a==b?'-gray':''})"><title>${a}|${r} → ${b}|${r+1}：${n.toFixed(1)} 条/场</title></path>`;
 }
 for(let r=1;r<=3;r++){s+=`<text x="${160+(r-1)*360}" y="28" text-anchor="middle" fill="var(--ink)" font-size="20">Round ${r}</text>`;for(const a of 'ABC'){const [x,y]=pos(a,r),v=nodeSummary(gg,a,r);s+=ring(x,y,33,v.props,a,v.n.toFixed(1)+' 条');}}
 $('time-graph').innerHTML=s+'</svg>';
}
for(const id of ['graph-condition','graph-round','graph-case','graph-kind','graph-topic'])$(id).addEventListener('change',drawGraph);drawGraph();
function clinical(id,edges){const ps={A:[260,85],B:[90,310],C:[430,310]};let s=`<svg viewBox="0 0 520 400" role="img" aria-label="临床 source-exclusive 流向">${defs(id+'-a')}`;for(const [a,b,n]of edges)s+=curved(ps[a],ps[b],n,id+'-a',a+' → '+b,'#1f7a6b',37);for(const a of 'ABC'){const[x,y]=ps[a];s+=`<circle cx="${x}" cy="${y}" r="31" fill="var(--panel)" stroke="#1f7a6b" stroke-width="3"/><text x="${x}" y="${y+7}" font-size="23" text-anchor="middle" fill="var(--ink)">${a}</text><text x="${x}" y="${y+55}" text-anchor="middle" font-size="13" fill="var(--dim)">${{A:'病史 / 查体',B:'影像 / 内镜',C:'化验'}[a]}</text>`;}$(id).innerHTML=s+'</svg>'}
clinical('clinical-generic',[['A','B',6],['A','C',5.5],['B','A',1.5],['C','A',1.5]]);
clinical('clinical-specialist',[['B','A',4.5],['C','A',4],['A','B',1],['A','C',2.5]]);
})();
(()=>{
function renderPers(){const top=$('pers-top').value,persona=$('pers-persona').value,rows=MENTOR_DATA.pers_network.edges.filter(r=>r.topology==top&&r.persona==persona),pos={A:[500,75],B:[205,310],C:[795,310]};let s='<svg viewBox="0 0 1000 420" role="img" aria-label="Perspectrum 可见复述网络"><defs><marker id="psArrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L10 5 L0 10z" fill="#1f7a6b"/></marker></defs>';
for(const a of 'ABC')for(const b of 'ABC')if(a!=b){const n=avg(rows.filter(r=>r.sender==a&&r.receiver==b).map(r=>r.n))||0;if(!n)continue;const[x,y]=pos[a],[xx,yy]=pos[b],dx=xx-x,dy=yy-y,len=Math.hypot(dx,dy),ux=dx/len,uy=dy/len,nx=-uy,ny=ux,cx=(x+xx)/2+nx*32,cy=(y+yy)/2+ny*32; s+=`<path d="M${x+ux*40} ${y+uy*40} Q${cx} ${cy} ${xx-ux*45} ${yy-uy*45}" fill="none" stroke="#1f7a6b" stroke-width="${1+n}" marker-end="url(#psArrow)"/><text x="${(x+xx)/2+nx*40}" y="${(y+yy)/2+ny*40}" text-anchor="middle" font-size="17" fill="var(--ink)" stroke="var(--panel)" stroke-width="5" paint-order="stroke">${n.toFixed(1)}</text>`;}
for(const a of 'ABC'){const[x,y]=pos[a];s+=`<circle cx="${x}" cy="${y}" r="33" fill="var(--panel)" stroke="#1f7a6b" stroke-width="3"/><text x="${x}" y="${y+7}" text-anchor="middle" fill="var(--ink)" font-size="24">${a}</text>`;}s+=`<text x="500" y="400" text-anchor="middle" fill="var(--dim)" font-size="15">${top} / ${persona} · 12 个 claim · 每场目标事实出现次数均值</text></svg>`;$('pers-graph').innerHTML=s;}
$('pers-top').addEventListener('change',renderPers);$('pers-persona').addEventListener('change',renderPers);renderPers();
})();

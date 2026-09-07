"""Directed sender redundancy, topic selection and compact time-expanded graphs."""
import json,math,statistics as st
from collections import defaultdict,Counter
from analyze_idrbench_entailment import ROOT,DATA,slot,temporal_edge,CELLS,paired
from analyze_discipline_flow import OUT,mean,div
D=json.loads((OUT/'analysis.json').read_text())
rows=[];graphs=[];samples=[]
for path in sorted(DATA.glob('*.nli.store.json')):
 s=json.loads(path.read_text());d=json.loads(path.with_name(path.name.replace('.nli.store.json','.debate.json')).read_text());lab=json.loads((ROOT/'experiments/labels/discipline_v1'/(d['execution_id']+'.json')).read_text())['labels']
 base={k:d[k] for k in ('execution_id','case_id','condition')};ms,mf=s['mentions'],s['mention_to_fact'];fs=defaultdict(set);mids=defaultdict(set);src=defaultdict(set);eq=defaultdict(set)
 for mid,m in ms.items():
  if slot(m):fs[slot(m)].add(mf[mid]);mids[slot(m)].add(mid)
  elif m['provenance']['channel']=='source':src[m['provenance']['doc_id']].add(mf[mid])
 for r in s['relations']:
  if r['relation']=='EQUIVALENT':eq[r['a']].add(r['b']);eq[r['b']].add(r['a'])
 ownf={k:set().union(*(fs[x] for x in v['visible_self_turns'])) for k,v in d['delivery'].items()};ownm={k:set().union(*(mids[x] for x in v['visible_self_turns'])) for k,v in d['delivery'].items()}
 senders=defaultdict(set);allincoming=defaultdict(set);edge_sets=defaultdict(set);edge_examples=defaultdict(list)
 for idx,r in enumerate(s['relations']):
  if r['relation']=='UNRELATED':continue
  a,b=ms[r['a']],ms[r['b']];kind,e,l=temporal_edge(a,b,r['relation'],d['delivery'])
  if kind.startswith(('peer_','self_')):
   ek,lk=slot(e),slot(l);ef,lf=mf[e['mention_id']],mf[l['mention_id']];allincoming[lk].add(lf)
   if int(lk.split('|')[1])!=int(ek.split('|')[1])+1:continue
   typ=kind.split('_')[1];edge_sets[ek,lk,kind,lab[lf]['topic']].add(lf)
   if len(edge_examples[ek,lk,kind])<4:edge_examples[ek,lk,kind].append(dict(source=e['text'],target=l['text'],source_mid=e['mention_id'],target_mid=l['mention_id'],index=idx))
   prior=ef in ownf[lk] or lf in ownf[lk] or bool(eq[e['mention_id']]&ownm[lk]) or bool(eq[l['mention_id']]&ownm[lk])
   if kind.startswith('peer_'):
    edge_sets[ek,lk,'peer_related',lab[lf]['topic']].add(lf)
    if not prior:
     senders[lk,lf].add(ek.split('|')[0]);edge_sets[ek,lk,'peer_screened',lab[lf]['topic']].add(lf)
  elif kind=='source_involved':
   for source,target in ((a,b),(b,a)):
    if source['provenance']['channel']!='source' or not slot(target):continue
    doc=source['provenance']['doc_id'];k=slot(target)
    if doc in d['delivery'][k]['source_ids']:
     f=mf[target['mention_id']];allincoming[k].add(f);edge_sets['src:'+doc,k,'source_related',lab[f]['topic']].add(f)
 nodes=[]
 for k,ff in sorted(fs.items()):
  a,rr=k.split('|');available=set().union(*(fs[x] for x in d['delivery'][k]['visible_self_turns']+d['delivery'][k]['visible_peer_turns']),*(src[x] for x in d['delivery'][k]['source_ids']))
  unmatched=ff-allincoming[k]-available
  nodes.append(dict(id=k,agent=a,round=int(rr),n=len(ff),unmatched=len(unmatched),topics=dict(Counter(lab[f]['topic'] for f in ff)),text=d['transcript'][k]))
  for t in 'BCXGU*':
   target={f for f in ff if t=='*' or lab[f]['topic']==t};cc=Counter(len(senders[k,f]) for f in target)
   rows.append(dict(base,agent=a,round=int(rr),topic=t,n=len(target),no_peer=div(cc[0],len(target)),one_sender=div(cc[1],len(target)),two_senders=div(cc[2],len(target)),multi_among_related=div(cc[2],cc[1]+cc[2]),exclusive_A=div(sum(senders[k,f]=={'A'} for f in target),len(target)),exclusive_B=div(sum(senders[k,f]=={'B'} for f in target),len(target)),exclusive_C=div(sum(senders[k,f]=={'C'} for f in target),len(target))))
  for t in 'BCXGU':
   edge_sets['new:'+a,k,'unmatched_origin',t]|={f for f in unmatched if lab[f]['topic']==t}
 graph_edges=[dict(source=x,target=y,kind=ty,topic=t,n=len(ff),examples=edge_examples.get((x,y,ty),[])) for (x,y,ty,t),ff in edge_sets.items() if ff]
 graphs.append(dict(base,nodes=nodes,edges=graph_edges,papers=d.get('evidence'),roles=d['roles']))
# Distribution geometry: paired task-level input/output separation & topic selection.
geo=[]
def js(p,q):
 m=[(a+b)/2 for a,b in zip(p,q)]
 return .5*sum(a*math.log2(a/c) for a,c in zip(p,m) if a)+.5*sum(b*math.log2(b/c) for b,c in zip(q,m) if b)
for tr in D['inputs']:
 rs=[r for r in D['turns'] if r['execution_id']==tr['execution_id']]
 for rr in (1,2,3):
  for scope in ('input','output'):
   aa=[r for r in rs if r['round']==rr and r['scope']==scope and r['n']]
   vecs=[[r[t+'_share'] for t in 'BCXGU'] for r in aa];j=[js(p,q) for i,p in enumerate(vecs) for q in vecs[i+1:]]
   geo.append(dict(execution_id=tr['execution_id'],case_id=tr['case_id'],condition=tr['condition'],round=rr,scope=scope,js=mean(j)))
# Paired contrasts, per-task rather than per-fact.
contrasts=[]
for info in ('full','split'):
 cs=[info+'-generic',info+'-specialist'+('-aligned' if info=='split' else '')]
 for topic in '*X':
  for metric in ('one_sender','two_senders','multi_among_related'):
   vals=[]
   for c in cs:
    g=defaultdict(list)
    for r in rows:
     if r['condition']==c and r['round']==3 and r['topic']==topic:g[r['case_id']].append(r[metric])
    vals.append({k:mean(v) for k,v in g.items()})
   cases=[k for k in vals[0] if vals[0][k] is not None and vals[1].get(k) is not None]
   contrasts.append(dict(info=info,topic=topic,metric=metric,generic=mean(vals[0][k] for k in cases),specialist=mean(vals[1][k] for k in cases),**paired([vals[1][k]-vals[0][k] for k in cases])))
geometry_contrasts=[]
for info in ('full','split'):
 for rr in (2,3):
  by={(r['condition'],r['case_id']):r['js'] for r in geo if r['scope']=='output' and r['round']==rr and r['condition'].startswith(info)}
  g=info+'-generic';sp=info+'-specialist'+('-aligned' if info=='split' else '')
  cases=sorted({k[1] for k in by})
  geometry_contrasts.append(dict(info=info,round=rr,metric='output_js',**paired([by[sp,c]-by[g,c] for c in cases])))
result=dict(rows=rows,graphs=graphs,geometry=geo,contrasts=contrasts,geometry_contrasts=geometry_contrasts)
(OUT/'network.json').write_text(json.dumps(result,ensure_ascii=False,separators=(',',':')))
for r in contrasts:print(r['info'],r['topic'],r['metric'],*[round(r[k],3) for k in ('generic','specialist','mean')],r['ci95'])
for c in CELLS:print('JSD',c,[(rr,sc,round(mean(r['js'] for r in geo if r['condition']==c and r['round']==rr and r['scope']==sc),3)) for rr in (1,2,3) for sc in ('input','output')])

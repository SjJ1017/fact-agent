"""Directly visible peer reuse and explicit two-hop hub witnesses, frozen v2 stores."""
import json,re
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
rows=[];relay=[]
for folder in ('perspectrum_pilot_full','perspectrum_pilot_star_chain'):
 for path in sorted((ROOT/'experiments'/folder).glob('*deepseek-v4-flash*.v2.json')):
  m=re.fullmatch(r'perspectrum-(\d+)-deepseek-v4-flash-(full|star|chain)-(neutral|lenses|stance)\.v2.json',path.name)
  if not m:continue
  claim,top,panel=m.groups();d=json.loads(path.with_name(path.name.replace('.v2.json','.debate.json')).read_text());s=json.loads(path.read_text());fs=defaultdict(set)
  for mid,mention in s['mentions'].items():
   p=mention['provenance']
   if p['channel']=='output':fs[f"{p['agent_id']}|{p['round']}"].add(s['mention_to_fact'][mid])
  byedge=defaultdict(set)
  for k,v in d['delivery'].items():
   a,rr=k.split('|');rr=int(rr);prior=set().union(*(fs[f'{a}|{r}'] for r in range(1,rr)))
   for prev in v.get('visible_peer_turns',v.get('peer_turns',[])):
    b,br=prev.split('|')
    for f in (fs[prev]&fs[k])-prior:byedge[b,a].add((k,f))
  for a in 'ABC':
   for b in 'ABC':
    if a!=b:rows.append(dict(claim=claim,topology=top,persona=panel,sender=a,receiver=b,n=len(byedge[a,b])))
  if top=='star':
   for a,b in [('B','C'),('C','B')]:
    initial=fs[a+'|1']-fs[b+'|1']-fs['A|1'];via=initial&fs['A|2']&fs[b+'|3'];endpoint=initial&fs[b+'|3'];received=initial&fs['A|2']
    assert a+'|1' in d['delivery']['A|2'].get('visible_peer_turns',d['delivery']['A|2']['peer_turns'])
    assert 'A|2' in d['delivery'][b+'|3'].get('visible_peer_turns',d['delivery'][b+'|3']['peer_turns'])
    relay.append(dict(claim=claim,persona=panel,sender=a,receiver=b,eligible=len(initial),hub_repeats=len(received),via_hub=len(via),endpoint_without_hub=len(endpoint-via)))
p=ROOT/'findings/data/idrbench-discipline/perspectrum-network.json';p.write_text(json.dumps(dict(edges=rows,relay=relay),indent=2));print('traces',len(rows)//6)
for panel in ['neutral','lenses','stance']:
 rs=[r for r in relay if r['persona']==panel];print(panel,{k:sum(r[k] for r in rs) for k in ['eligible','hub_repeats','via_hub','endpoint_without_hub']})

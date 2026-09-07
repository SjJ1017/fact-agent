"""Functional in/out balance and topic supplier alignment, task-level summaries."""
import json,statistics as st
from collections import defaultdict
from analyze_idrbench_entailment import paired
from analyze_discipline_flow import OUT,mean
N=json.loads((OUT/'network.json').read_text());rows=[]
for g in N['graphs']:
 for rr in (2,3):
  weights=defaultdict(float)
  for e in g['edges']:
   if e['kind']=='peer_screened' and e['target'].endswith('|'+str(rr)):
    weights[e['source'][0],e['target'][0]]+=e['n']
  for a in 'ABC':
   outgoing=sum(weights[a,b] for b in 'ABC' if a!=b);incoming=sum(weights[b,a] for b in 'ABC' if a!=b)
   rows.append(dict(execution_id=g['execution_id'],case_id=g['case_id'],condition=g['condition'],round=rr,agent=a,outgoing=outgoing,incoming=incoming,balance=outgoing-incoming))
summary=[]
for c in sorted({r['condition'] for r in rows}):
 for a in 'ABC':
  rr=[r for r in rows if r['condition']==c and r['round']==3 and r['agent']==a]
  summary.append(dict(condition=c,agent=a,**{k:mean(r[k] for r in rr) for k in ['outgoing','incoming','balance']}))
contrasts=[]
for info in ['full','split']:
 for a in 'ABC':
  rr=[r for r in rows if r['round']==3 and r['agent']==a and r['condition'].startswith(info)]
  vals={(r['condition'],r['case_id']):r['balance'] for r in rr};g=info+'-generic';sp=info+'-specialist'+('-aligned' if info=='split' else '')
  cases=sorted({r['case_id'] for r in rr})
  contrasts.append(dict(info=info,agent=a,**paired([vals[sp,c]-vals[g,c] for c in cases])))
(OUT/'network-balance.json').write_text(json.dumps(dict(rows=rows,summary=summary,contrasts=contrasts),indent=2))
for r in summary:print(r)

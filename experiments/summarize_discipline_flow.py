"""Task-balanced exploratory summaries for the discipline report."""
import json
from collections import defaultdict
from analyze_discipline_flow import OUT,mean
from analyze_idrbench_entailment import CELLS,paired
D=json.loads((OUT/'analysis.json').read_text())
def aggregate(rows,key):
 g=defaultdict(list)
 for r in rows:g[r['case_id']].append(r[key])
 return {k:mean(v) for k,v in g.items()}
results=[]
for info in ('full','split'):
 cs=[info+'-generic',info+'-specialist'+('-aligned' if info=='split' else '')]
 for metric in ('screened','equivalent','related'):
  for topic in ('*','B','C','X','G'):
   vals=[aggregate([r for r in D['interaction'] if r['condition']==c and r['source_round']==2 and r['topic']==topic],metric) for c in cs]
   cases=[k for k in vals[0] if vals[0][k] is not None and vals[1].get(k) is not None]
   results.append(dict(info=info,metric=metric,topic=topic,generic=mean(vals[0][k] for k in cases),specialist=mean(vals[1][k] for k in cases),**paired([vals[1][k]-vals[0][k] for k in cases])))
(OUT/'interaction-contrasts.json').write_text(json.dumps(results,indent=2))
for r in D['contrasts']:
 if r['agent'] in ('AB','C'):print('PROFILE',r['info'],r['agent'],r['metric'],*[round(r[k],4) for k in ('generic','specialist','mean')],r['ci95'])
for c in CELLS:
 print('CONDITION',c)
 for a in 'ABC':
  rs=[r for r in D['turns'] if r['condition']==c and r['agent']==a and r['round']==3 and r['scope']=='output']
  print(a,{t:round(mean(r[t+'_share'] for r in rs),3) for t in 'BCXGU'})
for r in results:
 if r['metric']=='screened':print('INTERACTION',r['info'],r['topic'],*[round(r[k],4) for k in ('generic','specialist','mean')],r['ci95'])

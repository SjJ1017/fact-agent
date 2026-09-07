"""Export the frozen corpus; unvalidated batches remain explicitly unlabelled.
No network calls. User approved proceeding with the remaining small missing fraction.
"""
from label_fact_discipline import *
manifest=json.loads((EVAL/'manifest.json').read_text());summary=json.loads((EVAL/'production-summary.json').read_text())
labels={};missing=[]
for case,task in sorted(manifest['tasks'].items()):
 items=sorted(task['items'].items())
 for j in range(0,len(items),64):
  batch=items[j:j+64];user='<papers>\n'+task['papers']+'\n</papers>\n<facts>\n'+json.dumps([[i,t] for i,(_,t) in enumerate(batch)],ensure_ascii=False)+'\n</facts>'
  spec=dict(model=summary['model'],system=SYSTEM,user=user,max_tokens=2048,temperature=0,extra=settings(summary['model']))
  cache=EVAL/'responses'/summary['scope']/(digest(spec)+'.json')
  if cache.exists():labels.update({k:dict(v,status='labelled') for k,v in json.loads(cache.read_text())['labels'].items()})
  else:
   missing.append(dict(case=case,batch_start=j,n=len(batch)))
   labels.update({uid:dict(topic='U',paper='U',status='unlabelled_invalid_batch') for uid,_ in batch})
unknown=0
for tr in manifest['traces']:
 path=ROOT/tr['store'];assert hashlib.sha256(path.read_bytes()).hexdigest()==tr['sha256']
 s=json.loads(path.read_text());case=tr['case'];papers=manifest['tasks'][case]['papers']
 mapped={fid:dict(labels[digest([case,papers,f['canonical_text']])[:20]],canonical_text=f['canonical_text']) for fid,f in s['facts'].items()}
 unknown+=sum(v['status']!='labelled' for v in mapped.values())
 dump(LABELS/(tr['execution_id']+'.json'),dict(execution_id=tr['execution_id'],layer='discipline_v1',model=summary['model'],system_hash=manifest['system_hash'],source_sha256=tr['sha256'],labels=mapped))
summary.update(exported=len(labels),unlabelled_unique=sum(x['n'] for x in missing),unlabelled_fact_ids=unknown,missing_batches=missing)
records=[json.loads(l) for p in EVAL.glob('*-calls.jsonl') for l in p.read_text().splitlines()]
summary['all_test_and_production_known_cost']=sum(r.get('cost_equivalent_usd',0) for r in records)
summary['all_calls']=len(records);summary['calls_without_usage']=sum('usage' not in r for r in records)
dump(EVAL/'production-summary.json',summary);print(json.dumps(summary,indent=2))

"""Offline Avalon: public evidence, visibility-aware reuse, beliefs and decisions.
No model calls. Frozen traces/stores; game is the resampling unit.
"""
import json,re,hashlib,itertools,statistics,random,csv
from pathlib import Path
from collections import defaultdict,Counter
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'experiments/avalon_5p_deepseek_v4_flash'
OUT=ROOT/'findings/data/avalon-flow'
EVIL={'Assassin','Minion'}
def mean(xs):
 xs=[x for x in xs if x is not None];return statistics.mean(xs) if xs else None
def rate(n,d):return n/d if d else None
def ci(xs):
 xs=[x for x in xs if x is not None]
 if not xs:return dict(n=0,mean=None,lo=None,hi=None)
 rng=random.Random(26);v=sorted(mean(rng.choices(xs,k=len(xs))) for _ in range(10000))
 return dict(n=len(xs),mean=mean(xs),lo=v[250],hi=v[9749])
def norm(m):return (re.sub(r'[^\w]+',' ',m['text'].lower()).strip(),m['polarity'])
def parse_alignment(m):
 # Only whole, unconditional, single-subject alignment clauses. Never regex-search inside an if-clause.
 h=re.fullmatch(r'Player ([0-4]) is (not )?(?:(?:a|an|the) )?(?:loyal )?(Good|Evil|Servant|Minion|Assassin)(?: player)?[. ]*',m['text'],re.I)
 if not h:return None
 subject=int(h[1]);label=h[3].lower();neg=bool(h[2]) or m['polarity']=='negate'
 # Negated specific roles do not determine alignment.
 if neg and label not in ['good','evil']:return None
 side=label in ['evil','minion','assassin'];return subject,(not side if neg else side)
def main():
 games=[];profiles=[];votes=[];claims=[];flow=[];networks=[];relays=[];leaks=[];frontiers=[];turns=[];audit=[];paths=[];sensitivity=defaultdict(Counter)
 for tp in sorted(DATA.glob('*.trace.json')):
  tr=json.loads(tp.read_text());sp=tp.with_name(tp.name.replace('.trace.json','.nli.store.json'));s=json.loads(sp.read_text());gid=tr['execution_id'];roles=tr['roles'];events={e['event_id']:e for e in tr['events']};ms=s['mentions'];byevent=defaultdict(list)
  for mid,m in ms.items():
   eid=m['provenance']['extra'].get('event_id')
   if eid is not None:byevent[eid].append(mid)
  groupmin={}
  for e in events.values():
   k=(e['quest'],e['proposal'],e['phase']);groupmin[k]=min(groupmin.get(k,100000),e['event_id'])
  def cut(e):return groupmin[e['quest'],e['proposal'],e['phase']]
  def known_before(m,e):
   pr=m['provenance'];x=pr['extra'];eid=x.get('event_id',100000)
   return eid<cut(e) and (x['visibility']=='public' or pr['agent_id']==f"P{e['actor']}")
  evils={int(p[1:]) for p,r in roles.items() if r in EVIL};merlin=next(int(p[1:]) for p,r in roles.items() if r=='Merlin')
  assignments=list(map(set,itertools.combinations(range(5),2)));before={};result_list=[]
  for eid,e in events.items():
   before[eid]=[c for c in assignments if all(len(c & set(r['action']['team']))>=r['action']['fail_count'] for r in result_list if r['event_id']<cut(e))]
   if e['phase']=='mission-result':result_list.append(e)
   if e['phase']=='discussion':frontiers.append(dict(game=gid,event=eid,quest=e['quest'],proposal=e['proposal'],player=e['actor'],role=roles[f"P{e['actor']}"],possible_pairs=len(before[eid]),public_fixed=sum(all(p in c for c in before[eid]) or all(p not in c for c in before[eid]) for p in range(5))))
  # Conservative propositional truth and each speaker's information frontier.
  seen=set();gc=[]
  for mid,m in ms.items():
   pr=m['provenance'];x=pr['extra'];h=parse_alignment(m);eid=x.get('event_id');ag=pr['agent_id']
   if not h or eid not in events or not ag or x['modality'] not in ['assertion','belief','speculation']:continue
   sub,side=h;e=events[eid];key=(eid,sub,side)
   if key in seen:continue
   seen.add(key);role=roles[ag];p=int(ag[1:]);poss=before[eid]
   pub_entailed=all((sub in c)==side for c in poss);pub_refuted=all((sub in c)!=side for c in poss)
   ownposs=[c for c in poss if p not in c] if role=='Servant' else [evils]
   own_entailed=all((sub in c)==side for c in ownposs);own_refuted=all((sub in c)!=side for c in ownposs)
   row=dict(game=gid,event=eid,quest=e['quest'],proposal=e['proposal'],speaker=p,role=role,subject=sub,subject_role=roles[f'P{sub}'],evil_claim=side,truth=(sub in evils)==side,public_entailed=pub_entailed,public_refuted=pub_refuted,own_entailed=own_entailed,own_refuted=own_refuted,visibility=x['visibility'],modality=x['modality'],phase=x['phase'],text=m['text'],quote=m['quote'],mention=mid,self_claim=p==sub)
   claims.append(row);gc.append(row)
  # Direct scored edges only, ordered by actual available context (discussion and votes simultaneous).
  eq=defaultdict(set);mf=s['mention_to_fact'];cluster=defaultdict(set)
  for mid,fid in mf.items():cluster[fid].add(mid)
  for r in s['relations']:
   if r['relation']=='EQUIVALENT':eq[r['a']].add(r['b']);eq[r['b']].add(r['a'])
  pred=defaultdict(list);gflow=[];same_time=Counter()
  for r in s['relations']:
   a,b=ms[r['a']],ms[r['b']];xa,xb=a['provenance']['extra'],b['provenance']['extra'];ea,eb=xa.get('event_id'),xb.get('event_id')
   if ea not in events or eb not in events or ea==eb:continue
   if ea>eb:a,b=b,a;xa,xb=xb,xa;ea,eb=eb,ea;swap=True
   else:swap=False
   aa,ba=a['provenance']['agent_id'],b['provenance']['agent_id']
   if aa is None or ba is None:continue
   dest=events[eb]
   if ea>=cut(dest):
    if aa!=ba and xa['visibility']=='public' and xb['visibility']=='public':same_time[r['relation']]+=1
    continue
   if xa['visibility']!='public' and aa!=ba:continue
   if xa['scope']=='state' or xb['scope']=='state':
    if xa['quest']!=xb['quest']:continue
   if xb['phase'] not in ['discussion','proposal','team-vote','assassination']:continue
   ra=r['relation'];typ='equivalent' if ra=='EQUIVALENT' else 'unrelated' if ra=='UNRELATED' else 'weaken' if (ra=='A_ENTAILS_B')!=swap else 'strengthen'
   channel='self_private_to_public' if aa==ba and xa['visibility']!='public' and xb['visibility']=='public' else 'self' if aa==ba else 'public_to_public' if xb['visibility']=='public' else 'public_to_private'
   # Already expressible from a previous own mention = recurrence, not new reception.
   own_prior=any(ms[z]['provenance']['agent_id']==ba and known_before(ms[z],dest)
    and (not (xb['scope']=='state' or ms[z]['provenance']['extra']['scope']=='state') or ms[z]['provenance']['extra']['quest']==xb['quest'])
    for z in eq[b['mention_id']] | cluster[mf[b['mention_id']]])
   props=r.get('properties',{});ma,mb=props.get('margin_ab'),props.get('margin_ba')
   if channel.startswith('public_to') and not own_prior:
    for th in (3.28,5.28,7.28):
     if ma is not None and mb is not None:sensitivity[str(th)]['related_edges']+=ma>=th or mb>=th;sensitivity[str(th)]['equivalent_edges']+=ma>=th and mb>=th
   if typ=='unrelated':continue
   row=dict(game=gid,source=ea,target=eb,sender=int(aa[1:]),receiver=int(ba[1:]),sender_role=roles[aa],receiver_role=roles[ba],channel=channel,kind=typ,prior_self=own_prior,source_mid=a['mention_id'],target_mid=b['mention_id'],source_fact=mf[a['mention_id']],target_fact=mf[b['mention_id']],source_text=a['text'],target_text=b['text'],source_modality=xa['modality'],target_modality=xb['modality'],source_scope=xa['scope'],target_scope=xb['scope'],source_quest=xa['quest'],target_quest=xb['quest'])
   gflow.append(row);pred[b['mention_id']].append(row)
  flow.extend(gflow)
  # Coverage per output event: multiple candidate senders do not multiply a target.
  for eid,mids in byevent.items():
   e=events[eid]
   if e['kind']!='agent' or e['phase'] not in ['discussion','proposal','team-vote','assassination']:continue
   targetset={mf[z] for z in mids};matches=[r for r in gflow if r['target']==eid and r['channel'].startswith('public_to')];novel=[r for r in matches if not r['prior_self']]
   turns.append(dict(game=gid,event=eid,player=e['actor'],role=roles[f"P{e['actor']}"],quest=e['quest'],proposal=e['proposal'],phase=e['phase'],n=len(targetset),related=len({r['target_fact'] for r in matches}),new_related=len({r['target_fact'] for r in novel}),new_equivalent=len({r['target_fact'] for r in novel if r['kind']=='equivalent'}),new_nonrecord=len({r['target_fact'] for r in novel if r['source_modality']!='record' and r['target_modality']!='record'})))
  # Earliest direct predecessor per receiving mention; ties split rather than alphabetical credit.
  graph=defaultdict(float);gpaths=[]
  for mid,rows in pred.items():
   valid=[r for r in rows if r['channel']=='public_to_public' and not r['prior_self']]
   if not valid:continue
   first=min(cut(events[r['source']]) for r in valid);rr=[r for r in valid if cut(events[r['source']])==first];agents=set(r['sender'] for r in rr)
   for a in agents:graph[a,rr[0]['receiver']]+=1/len(agents)
   # Actual intermediary mention must join both edges; source-available direct paths are not proof of mediation.
   for r in valid:
    for q in pred.get(r['source_mid'],[]):
     if q['channel']=='public_to_public' and len({q['sender'],q['receiver'],r['receiver']})==3:
      key=(q['source_mid'],r['source_mid'],mid)
      if key in gpaths:continue
      gpaths.append(key)
      paths.append(dict(game=gid,players=[q['sender'],q['receiver'],r['receiver']],events=[q['source'],r['source'],r['target']],texts=[q['source_text'],r['source_text'],r['target_text']],types=[q['kind'],r['kind']],modalities=[q['source_modality'],r['source_modality'],r['target_modality']]))
  networks.append(dict(game=gid,roles=roles,edges=[dict(sender=a,receiver=b,credit=w) for (a,b),w in graph.items()]))
  # Role-claim first uptake by each recipient, excluding same-batch convergence.
  for c in gc:
   if c['visibility']!='public':continue
   prior=[z for z in gc if z['subject']==c['subject'] and z['evil_claim']==c['evil_claim'] and z['event']<cut(events[c['event']])]
   if any(z['speaker']==c['speaker'] for z in prior):continue
   ps=[z for z in prior if z['visibility']=='public' and z['speaker']!=c['speaker']]
   if not ps:continue
   first=min(cut(events[z['event']]) for z in ps);orig=[z for z in ps if cut(events[z['event']])==first];ors=sorted({z['role'] for z in orig})
   relays.append(dict(c,origin_roles=ors,origin_speakers=sorted({z['speaker'] for z in orig}),origin_event=min(z['event'] for z in orig),known_without_speech=c['own_entailed'] or c['own_refuted']))
  games.append(dict(game=gid,outcome=tr['outcome'],roles=roles,quests=len(tr['quest_outcomes']),successes=tr['score']['good'],failures=tr['score']['evil'],events=len(events),mentions=len(ms),facts=len(s['facts']),relations=len(s['relations']),fourway=s['matching']['counts'],same_batch_related=sum(v for k,v in same_time.items() if k!='UNRELATED'),public_remaining_pairs=len(before[max(events)]),matching=s['matching']))
  audit.append(dict(game=gid,trace=str(tp.relative_to(ROOT)),trace_sha256=hashlib.sha256(tp.read_bytes()).hexdigest(),store=str(sp.relative_to(ROOT)),store_sha256=hashlib.sha256(sp.read_bytes()).hexdigest()))
 out=dict(games=games,profiles=profiles,votes=votes,claims=claims,flow=flow,networks=networks,claim_uptake=relays,assassinations=leaks,frontiers=frontiers,turns=turns,paths=paths,inputs=audit,sensitivity=dict(sensitivity))
 OUT.mkdir(exist_ok=True,parents=True);(OUT/'analysis.json').write_text(json.dumps(out,ensure_ascii=False,separators=(',',':')))
 for name in ['claims','claim_uptake','turns','games']:
  rows=out[name]
  with (OUT/(name+'.csv')).open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('COMPLETE',len(games),'games',len(claims),'claims',len(flow),'related edges',len(paths),'3-agent paths')
 print('CLAIMS')
 for role in ['Merlin','Servant','Minion','Assassin']:
  for vis in ['public','private-until-reveal']:
   cc=[c for c in claims if c['role']==role and c['visibility']==vis];print(role,vis,len(cc),'false',sum(not c['truth'] for c in cc),'ownrefuted',sum(c['own_refuted'] for c in cc),'publicentailed',sum(c['public_entailed'] for c in cc))
if __name__=='__main__':main()

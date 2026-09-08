"""Atom-only Avalon measurements. Trace fields supply timing/role/outcome labels only."""
import json,re,itertools
from collections import Counter,defaultdict
from analyze_information_flow import ROOT,DATA,OUT,mean,ci,rate,EVIL
ROLES=['Merlin','Servant','Minion','Assassin']
def main():
 d=json.loads((OUT/'analysis.json').read_text());games=d['games'];flow=d['flow'];turns=d['turns'];matrix=[];coverage=[];fourway=Counter();sources={};mtags=[];truthfacts=[];rules=[];atomic_cases={}
 for g in games:fourway.update(g['fourway'])
 for role in ROLES:
  for phase in ['discussion','team-vote','assassination']:
   ts=[t for t in turns if t['role']==role and t['phase']==phase];n=sum(t['n'] for t in ts)
   if not n:continue
   rr=dict(role=role,phase=phase,n=n)
   for k in ['new_related','new_equivalent','new_nonrecord']:
    rr[k]=sum(t[k] for t in ts);rr[k+'_ci']=ci([rate(sum(t[k] for t in ts if t['game']==g['game']),sum(t['n'] for t in ts if t['game']==g['game'])) for g in games])
   coverage.append(rr)
 # Direct edge coverage, permitting attribution overlap across sender roles but not within a cell.
 for channel in ['public_to_public','public_to_private']:
  for mode in ['related','equivalent','nonrecord']:
   for sr in ROLES:
    for rr in ROLES:
     n=hit=0;gx=[]
     for g in games:
      ts=[t for t in turns if t['game']==g['game'] and t['role']==rr and (t['phase'] in ['discussion','proposal'] if channel=='public_to_public' else t['phase'] in ['team-vote','assassination'])]
      denom=sum(t['n'] for t in ts);rs=[r for r in flow if r['game']==g['game'] and r['sender_role']==sr and r['receiver_role']==rr and r['channel']==channel and not r['prior_self'] and (mode!='equivalent' or r['kind']=='equivalent') and (mode!='nonrecord' or (r['source_modality']!='record' and r['target_modality']!='record'))];num=len({(r['target'],r['target_fact']) for r in rs});n+=denom;hit+=num;gx.append(rate(num,denom))
     matrix.append(dict(channel=channel,mode=mode,sender=sr,receiver=rr,targets=n,hits=hit,pooled=rate(hit,n),game_mean=ci(gx)))
 # Deduplicated target modality transitions and source modality composition.
 modal=[]
 for sr in ['assertion','belief','speculation','record','directive','strategy','intention']:
  for ta in ['assertion','belief','speculation','record','directive','strategy','intention']:
   rr=[r for r in flow if r['channel']=='public_to_public' and not r['prior_self'] and r['kind']=='equivalent' and r['source_modality']==sr and r['target_modality']==ta]
   modal.append(dict(source=sr,target=ta,n=len({(r['game'],r['target'],r['target_fact']) for r in rr}),games=len({r['game'] for r in rr})))
 # True/false applies to the extracted unconditional alignment proposition, not a universal lie detector.
 truth_by_role=[]
 for role in ROLES:
  for vis in ['public','private-until-reveal','secret']:
   cc=[c for c in d['claims'] if c['role']==role and c['visibility']==vis];truth_by_role.append(dict(role=role,visibility=vis,n=len(cc),false=sum(not c['truth'] for c in cc),self_n=sum(c['self_claim'] for c in cc),false_self=sum(c['self_claim'] and not c['truth'] for c in cc)))
 # Trace timing reconstructed solely from event metadata. No utterance text is used.
 exposure=[];crossprivate=[];profiles=[];publicfirst=[]
 rule_re=re.compile(r'(?:The )?Quest ([1-5])(?: team)? (?:requires|needs) (?:(?:a|full|exactly) )*(2|3|two|three)(?:[- ](?:person|player)(?:s)?(?: team)?)?[.]?',re.I)
 for g in games:
  gid=g['game'];store=json.loads((DATA/(gid+'.nli.store.json')).read_text());ms=store['mentions'];tr=json.loads((DATA/(gid+'.trace.json')).read_text());events={e['event_id']:e for e in tr['events']};groupmin={}
  for e in events.values():
   key=e['quest'],e['proposal'],e['phase'];groupmin[key]=min(groupmin.get(key,100000),e['event_id'])
  def time(mid):
   e=events[ms[mid]['provenance']['extra']['event_id']];return groupmin[e['quest'],e['proposal'],e['phase']]
  pms={mid:m for mid,m in ms.items() if m['provenance']['agent_id'] and m['provenance']['extra']['visibility']=='public'}
  # One node per atomic occurrence; event ids only implement legal exposure.
  atomic_cases[gid]=dict(roles=g['roles'],outcome=g['outcome'],mentions={mid:dict(text=m['text'],polarity=m['polarity'],agent=m['provenance']['agent_id'],batch=time(mid),**m['provenance']['extra']) for mid,m in ms.items() if m['provenance']['agent_id']},edges=[r for r in flow if r['game']==gid])
  gflow=[r for r in flow if r['game']==gid]
  for mid,m in ms.items():
   h=rule_re.fullmatch(m['text'].strip())
   if h and m['provenance']['agent_id'] and m['polarity']=='affirm':
    q=int(h[1]);n={'two':2,'three':3}.get(h[2].lower(),int(h[2]) if h[2].isdigit() else None);x=m['provenance']['extra'];rules.append(dict(game=gid,mention=mid,text=m['text'],quest=q,size=n,truth=n==tr['rules']['quest_team_sizes'][q-1] if 'quest_team_sizes' in tr['rules'] else n==[2,3,2,3,3][q-1],player=int(m['provenance']['agent_id'][1:]),role=g['roles'][m['provenance']['agent_id']],event=x['event_id'],visibility=x['visibility'],phase=x['phase'],modality=x['modality']))
  for p in range(5):
   role=g['roles'][f'P{p}'];ps=[m for m in pms.values() if m['provenance']['agent_id']==f'P{p}'];outgoing=set();destinations=set()
   for r in gflow:
    if r['sender']==p and r['channel'].startswith('public_to') and not r['prior_self']:outgoing.add(r['source_mid']);destinations.add((r['source_mid'],r['receiver']))
   profiles.append(dict(game=gid,player=p,role=role,public_atoms=len(ps),propagated_atoms=len(outgoing),fanout=rate(len(destinations),len(outgoing)),propagation_rate=rate(len(outgoing),len(ps))))
   for channel in ['public_to_public','public_to_private']:
    for mode in ['equivalent','related','nonrecord']:
     ts=[t for t in turns if t['game']==gid and t['player']==p and (t['phase'] in ['discussion','proposal'] if channel=='public_to_public' else t['phase'] in ['team-vote','assassination'])]
     rs=[r for r in gflow if r['receiver']==p and r['channel']==channel and not r['prior_self'] and (mode!='equivalent' or r['kind']=='equivalent') and (mode!='nonrecord' or (r['source_modality']!='record' and r['target_modality']!='record'))]
     bytarget=defaultdict(list)
     for r in rs:bytarget[(r['target'],r['target_fact'])].append(r)
     observed=expected=0
     for target_key,rr in bytarget.items():
      mid=rr[0]['target_mid']
      # Nearest available predecessor, shared attribution if same-batch ties; no A/B alphabetical credit.
      latest=max(time(r['source_mid']) for r in rr);orig={r['sender'] for r in rr if time(r['source_mid'])==latest};observed+=sum(g['roles'][f'P{a}'] in EVIL for a in orig)/len(orig)
      pool=[m for z,m in pms.items() if m['provenance']['agent_id']!=f'P{p}' and time(z)<time(mid) and (mode!='nonrecord' or m['provenance']['extra']['modality']!='record')]
      expected+=rate(sum(g['roles'][m['provenance']['agent_id']] in EVIL for m in pool),len(pool)) or 0
     exposure.append(dict(game=gid,player=p,role=role,channel=channel,mode=mode,n=sum(t['n'] for t in ts),matched=len(bytarget),evil_credit=observed,evil_expected=expected,evil_fraction=rate(observed,len(bytarget)),evil_exposure=rate(expected,len(bytarget))))
  # Canonical semantic keys for narrow truth-labelled classes, independent of NLI blocking recall.
  cc=[dict(c,key=f"alignment:{c['subject']}:{c['evil_claim']}",player=c['speaker']) for c in d['claims'] if c['game']==gid]
  cc += [dict(r,key=f"size:{r['quest']}:{r['size']}",self_claim=False) for r in rules if r['game']==gid]
  kk=defaultdict(list)
  for c in cc:kk[c['key']].append(c)
  for key,cc in kk.items():
   pu=[c for c in cc if c['visibility']=='public'];pv=[c for c in cc if c['visibility']!='public']
   if not pu:continue
   def ct(c):
    e=events[c['event']];return groupmin[e['quest'],e['proposal'],e['phase']]
   first=min(map(ct,pu));orig={c['player'] for c in pu if ct(c)==first};later={c['player'] for c in pu if ct(c)>first}-orig
   privatefirst={c['player'] for c in pv if ct(c)<first};row=dict(game=gid,key=key,text=pu[0]['text'],truth=pu[0]['truth'],public_atoms=len(pu),public_agents=len({c['player'] for c in pu}),private_agents=len({c['player'] for c in pv}),first_event=first,origin_players=sorted(orig),origin_roles=sorted({g['roles'][f'P{p}'] for p in orig}),later_agents=sorted(later),later_without_prior_private=sorted(later-privatefirst),events=sorted({c['event'] for c in cc}),mentions=[c['mention'] for c in cc]);truthfacts.append(row)
   for p in range(5):
    # Same-player private-before-public expression, not access to latent thoughts.
    ownp=[c for c in pv if c['player']==p];ownu=[c for c in pu if c['player']==p]
    if ownp and ownu and min(map(ct,ownp))<min(map(ct,ownu)):crossprivate.append(dict(game=gid,player=p,role=g['roles'][f'P{p}'],key=key,truth=pu[0]['truth'],private_event=min(c['event'] for c in ownp),public_event=min(c['event'] for c in ownu)))
 # Camp comparison controls the 3-vs-2 player-count difference using source-pool proportions.
 camp=[]
 for mode in ['related','equivalent','nonrecord']:
  for channel in ['public_to_public','public_to_private']:
   for campname,rr in [('Good',{'Merlin','Servant'}),('Evil',EVIL)]:
    rows=[r for r in exposure if r['role'] in rr and r['channel']==channel and r['mode']==mode];pergame=[]
    for g in games:
     gs=[r for r in rows if r['game']==g['game']];n=sum(r['matched'] for r in gs)
     if n:pergame.append(dict(game=g['game'],observed=sum(r['evil_credit'] for r in gs)/n,expected=sum(r['evil_expected'] for r in gs)/n))
    camp.append(dict(camp=campname,mode=mode,channel=channel,atoms=sum(r['n'] for r in rows),matched=sum(r['matched'] for r in rows),evil_credit=sum(r['evil_credit'] for r in rows),diff=ci([r['observed']-r['expected'] for r in pergame]),games=pergame))
 # Role contrasts matched at the game level, combining both Servants within each game.
 contrasts=[]
 for phase in ['discussion','team-vote']:
  for metric in ['new_related','new_equivalent','new_nonrecord']:
   vals=[]
   for g in games:
    def rrate(role):
     rs=[t for t in turns if t['game']==g['game'] and t['role']==role and t['phase']==phase];return rate(sum(t[metric] for t in rs),sum(t['n'] for t in rs))
    a,b=rrate('Servant'),rrate('Merlin')
    if a is not None and b is not None:vals.append(a-b)
   contrasts.append(dict(phase=phase,metric=metric,comparison='Servant minus Merlin',**ci(vals)))
 interactions=[]
 for metric in ['new_related','new_equivalent','new_nonrecord']:
  vals=[]
  for g in games:
   def z(role,phase):
    t=[t for t in turns if t['game']==g['game'] and t['role']==role and t['phase']==phase];return sum(x[metric] for x in t)/sum(x['n'] for x in t)
   vals.append((z('Servant','team-vote')-z('Merlin','team-vote'))-(z('Servant','discussion')-z('Merlin','discussion')))
  interactions.append(dict(metric=metric,positive_games=sum(v>0 for v in vals),**ci(vals)))
 out=dict(outcomes=dict(Counter(g['outcome'] for g in games)),fourway=dict(fourway),oneway_share=rate(fourway['A_ENTAILS_B']+fourway['B_ENTAILS_A'],sum(v for k,v in fourway.items() if k!='UNRELATED')),coverage=coverage,matrix=matrix,modality=modal,truth_by_role=truth_by_role,truth_facts=truthfacts,rule_atoms=rules,exposure=exposure,camp=camp,profiles=profiles,contrasts=contrasts,interactions=interactions,private_public=crossprivate,atomic_cases=atomic_cases)
 (OUT/'summary.json').write_text(json.dumps(out,ensure_ascii=False,separators=(',',':')))
 print('CAMP',[(r['camp'],r['channel'],r['mode'],r['matched'],r['diff']) for r in camp]);print('CONTRASTS',contrasts)
 print('TRUTH_FACTS',len(truthfacts),'false',sum(not r['truth'] for r in truthfacts),'RULE',len(rules),sum(not r['truth'] for r in rules));print('false diffuse',[(r['game'],r['key'],r['public_agents'],r['origin_roles'],r['later_agents']) for r in truthfacts if not r['truth']]);print('PRIVATE_PUBLIC',crossprivate)
if __name__=='__main__':main()

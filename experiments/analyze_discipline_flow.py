"""Offline topic-conditioned profiles on a completely labelled frozen corpus."""
import json, hashlib, statistics as st
from collections import defaultdict, Counter
from itertools import permutations, combinations
from analyze_idrbench_entailment import ROOT, DATA, CELLS, slot, temporal_edge, write_csv, paired
OUT=ROOT/'findings/data/idrbench-discipline'
TOPICS='BCXGU'
def mean(xs):
    xs=[x for x in xs if x is not None]
    return st.mean(xs) if xs else None
def div(a,b):return a/b if b else None
def counts(fs,lab):return {t:sum(lab[f]['topic']==t for f in fs) for t in TOPICS}
def profile(fs,lab):
    c=counts(fs,lab);return dict(n=len(fs),**c,**{t+'_share':div(v,len(fs)) for t,v in c.items()})
def main():
    turns=[];edges=[];overlaps=[];uptake=[];transitions=[];examples=[];facts=[];inputs=[]
    for path in sorted(DATA.glob('*.nli.store.json')):
        s=json.loads(path.read_text());d=json.loads(path.with_name(path.name.replace('.nli.store.json','.debate.json')).read_text())
        side=json.loads((ROOT/'experiments/labels/discipline_v1'/ (d['execution_id']+'.json')).read_text());lab=side['labels']
        assert set(lab)==set(s['facts']) and side['source_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
        base={k:d[k] for k in ('execution_id','case_id','condition')};inputs.append(dict(base,store=str(path.relative_to(ROOT)),system_hash=side['system_hash'],matching=s['matching']))
        ms,mf=s['mentions'],s['mention_to_fact'];fs=defaultdict(set);mids=defaultdict(set);src=defaultdict(set);eq=defaultdict(set)
        for mid,m in ms.items():
            k=slot(m)
            if k:fs[k].add(mf[mid]);mids[k].add(mid)
            elif m['provenance']['channel']=='source':src[m['provenance']['doc_id']].add(mf[mid])
        for r in s['relations']:
            if r['relation']=='EQUIVALENT':eq[r['a']].add(r['b']);eq[r['b']].add(r['a'])
        ownf={k:set().union(*(fs[x] for x in v['visible_self_turns'])) for k,v in d['delivery'].items()}
        ownm={k:set().union(*(mids[x] for x in v['visible_self_turns'])) for k,v in d['delivery'].items()}
        for k,v in d['delivery'].items():
            a,rr=k.split('|'); sources=set().union(*(src[x] for x in v['source_ids']));peer=set().union(*(fs[x] for x in v['visible_peer_turns']));fresh=set().union(*(fs[x] for x in v['peer_turns']))
            groups=dict(output=fs[k],input=sources|peer|ownf[k],source=sources,peer=peer,self=ownf[k],delivered=fresh)
            for scope,ff in groups.items():
                row=dict(base,agent=a,round=int(rr),scope=scope,**profile(ff,lab))
                row['own_share']=div(sum(lab[f]['topic']=={'A':'B','B':'C'}.get(a) for f in ff)+.5*sum(lab[f]['topic']=='X' for f in ff),len(ff)) if a in 'AB' else None
                row['proposal_share']=div(sum(lab[f]['paper']=='P' for f in ff),len(ff));turns.append(row)
        cov=defaultdict(set);scr=defaultdict(set);ov=defaultdict(set);used=defaultdict(set);tested=defaultdict(set);tx=defaultdict(set)
        for idx,r in enumerate(s['relations']):
            a,b=ms[r['a']],ms[r['b']];ak,bk=slot(a),slot(b)
            if not ak or not bk:continue
            aa,ar=ak.split('|');ba,br=bk.split('|')
            if ar==br and aa!=ba and r['relation']!='UNRELATED':
                for x,y,mid in ((ak,bk,r['a']),(bk,ak,r['b'])):
                    ov[x,y,'related'].add(mf[mid])
                    if r['relation']=='EQUIVALENT':ov[x,y,'equivalent'].add(mf[mid])
            kind,early,late=temporal_edge(a,b,r['relation'],d['delivery'])
            if not kind.startswith('peer_'):continue
            ek,lk=slot(early),slot(late)
            if int(lk.split('|')[1])!=int(ek.split('|')[1])+1:continue
            em,lm=early['mention_id'],late['mention_id'];ef,lf=mf[em],mf[lm];typ=kind[5:]
            tested[lk].add(ef);cov[ek,lk,'scored'].add(lf)
            if typ=='unrelated':continue
            cov[ek,lk,typ].add(lf);cov[ek,lk,'related'].add(lf);used[lk].add(ef)
            prior=ef in ownf[lk] or lf in ownf[lk] or bool(eq[em]&ownm[lk]) or bool(eq[lm]&ownm[lk])
            if not prior:
                scr[ek,lk,typ].add(lf);scr[ek,lk,'related'].add(lf)
            tx[lab[ef]['topic'],lab[lf]['topic'],typ].add((ef,lf))
            if len([e for e in examples if e['execution_id']==d['execution_id']])<6 and not prior:
                examples.append(dict(base,index=idx,source_slot=ek,target_slot=lk,kind=typ,source_topic=lab[ef]['topic'],target_topic=lab[lf]['topic'],source_text=early['text'],target_text=late['text'],source_turn=d['transcript'][ek],target_turn=d['transcript'][lk]))
        for rr in (1,2,3):
            for i,j in permutations('ABC',2):
                ik,jk=f'{i}|{rr}',f'{j}|{rr}'
                for t in TOPICS+'*':
                    target={f for f in fs[jk] if t=='*' or lab[f]['topic']==t}
                    overlaps.append(dict(base,round=rr,sender=i,receiver=j,topic=t,n=len(target),**{ty:div(len(ov[jk,ik,ty]&target),len(target)) for ty in ('equivalent','related')}))
        for rr in (1,2):
            for i,j in permutations('ABC',2):
                ik,jk=f'{i}|{rr}',f'{j}|{rr+1}'
                if ik not in d['delivery'][jk]['visible_peer_turns']:continue
                for t in TOPICS+'*':
                    target={f for f in fs[jk] if t=='*' or lab[f]['topic']==t}
                    row=dict(base,source_round=rr,sender=i,receiver=j,topic=t,n=len(target))
                    for ty in ('equivalent','weaken','strengthen','related','scored'):
                        row[ty]=div(len(cov[ik,jk,ty]&target),len(target))
                    row['screened']=div(len(scr[ik,jk,'related']&target),len(target));edges.append(row)
            for j in 'ABC':
                jk=f'{j}|{rr+1}';ff=set().union(*(fs[x] for x in d['delivery'][jk]['peer_turns'] if int(x.split('|')[1])==rr))
                for t in TOPICS:
                    target={f for f in ff if lab[f]['topic']==t}
                    uptake.append(dict(base,round=rr+1,agent=j,topic=t,n=len(target),used_n=len(target&used[jk]),scored_n=len(target&tested[jk]),rate=div(len(target&used[jk]),len(target)),scored_rate=div(len(target&tested[jk]),len(target))))
        for (a,b,ty),pairs in tx.items():transitions.append(dict(base,source_topic=a,target_topic=b,kind=ty,n=len(pairs)))
        for fid,f in s['facts'].items():
            positions=sorted({slot(ms[mid]) or ms[mid]['provenance'].get('doc_id','source') for mid in f['mention_ids']})
            facts.append(dict(base,fact_id=fid,**lab[fid],positions=positions))
    assert len(inputs)==40 and len({i['system_hash'] for i in inputs})==1
    OUT.mkdir(parents=True,exist_ok=True)
    for name,rows in [('turns',turns),('interaction',edges),('overlap',overlaps),('uptake',uptake),('transitions',transitions)]:write_csv(OUT/(name+'.csv'),rows)
    # Paired task-level effects: no pooling atomic facts as independent observations.
    contrasts=[]
    for info in ('full','split'):
        generic=info+'-generic';special=info+'-specialist'+('-aligned' if info=='split' else '')
        for agent in ('A','B','AB','C'):
            for metric in ('output_own','enrichment','output_X'):
                if agent=='C' and metric!='output_X':continue
                vals={}
                for c in (generic,special):
                    for case in sorted({r['case_id'] for r in turns}):
                        rs=[r for r in turns if r['condition']==c and r['case_id']==case and r['agent'] in agent and r['round']>=2]
                        field='X_share' if metric=='output_X' else 'own_share'
                        out=mean(r[field] for r in rs if r['scope']=='output');inp=mean(r[field] for r in rs if r['scope']=='input')
                        vals[c,case]=(out-inp if out is not None and inp is not None else None) if metric=='enrichment' else out
                cases=[case for case in sorted({r['case_id'] for r in turns}) if vals[generic,case] is not None and vals[special,case] is not None]
                contrasts.append(dict(info=info,agent=agent,metric=metric,generic=mean(vals[generic,k] for k in cases),specialist=mean(vals[special,k] for k in cases),**paired([vals[special,k]-vals[generic,k] for k in cases])))
    result=dict(inputs=inputs,turns=turns,interaction=edges,overlap=overlaps,uptake=uptake,transitions=transitions,contrasts=contrasts,examples=examples,facts=facts)
    (OUT/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n')
    print(json.dumps(contrasts,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

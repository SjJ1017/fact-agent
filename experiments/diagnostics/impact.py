import json, glob, re, statistics as st
from collections import defaultdict
STOP=set('a an the is are was were of to in on for that this it its and or not no by with as be been'.split())
def toks(t): return {w for w in re.findall(r'[a-z0-9]+', t.lower()) if w not in STOP}

sib_tot=fact_tot=0; per=[]
novel_hit=novel_tot=0
examples=[]
for f in glob.glob('experiments/perspectrum_pilot_*/*deepseek*.v2.json'):
    s=json.load(open(f)); dbg=f.replace('.v2.json','.debate.json')
    d=json.load(open(dbg))
    out={m for m,v in s['mentions'].items() if v['provenance']['channel']=='output'}
    ids=[fid for fid,fa in s['facts'].items() if set(fa['mention_ids'])&out]
    T={i:toks(s['facts'][i]['canonical_text']) for i in ids}
    sib=set()
    for i in ids:
        for j in ids:
            if i>=j: continue
            a,b=T[i],T[j]
            if not a or not b: continue
            # 严格子集且共享主语词 -> 粒度兄弟
            if (a<b or b<a) and len(a&b)>=2:
                sib.add(i); sib.add(j)
                if len(examples)<8 and abs(len(a)-len(b))<=3:
                    examples.append((s['facts'][i]['canonical_text'], s['facts'][j]['canonical_text']))
    sib_tot+=len(sib); fact_tot+=len(ids); per.append(len(sib)/len(ids) if ids else 0)

    # 这些粒度兄弟里，有多少被算成了"新事实"
    said=defaultdict(set)
    for mid,m in s['mentions'].items():
        p=m['provenance']
        if p['channel']=='output' and p.get('agent_id') and p.get('round'):
            fid=s['mention_to_fact'].get(mid)
            if fid: said[(p['agent_id'],p['round'])].add(fid)
    rounds=sorted({r for _,r in said}); ags=sorted({a for a,_ in said})
    recv=defaultdict(set)
    for slot,info in d.get('delivery',{}).items():
        b,r=slot.split('|')
        for peer in info.get('peer_turns',[]):
            a,pr=peer.split('|'); recv[(b,int(r))]|=said[(a,int(pr))]
    for a in ags:
        for r in rounds:
            mine=set().union(*[said[(a,x)] for x in rounds if x<r]) if r>rounds[0] else set()
            fresh=said[(a,r)]-mine
            nov=fresh-recv[(a,r)]
            novel_tot+=len(nov); novel_hit+=len(nov & sib)
print(f'output 事实 {fact_tot}，其中与同场另一条构成严格子集关系的 {sib_tot} = {sib_tot/fact_tot:.1%}（每场中位 {st.median(per):.1%}）')
print(f'被记为「新事实」的共 {novel_tot}，其中 {novel_hit} 条是粒度兄弟 = {novel_hit/novel_tot:.1%}')
print()
print('样例（同场、一条是另一条的严格子集）：')
for a,b in examples:
    print(f'   · {a}')
    print(f'     {b}')

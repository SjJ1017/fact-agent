import json, glob, re, statistics as st
from collections import defaultdict, Counter
EV=re.compile(r'\bE[1-4]\b')
DOSSIER=re.compile(r'\b(dossier|evidence|source|panelist|claim)\b', re.I)

tot=defaultdict(lambda: Counter())
for f in glob.glob('experiments/perspectrum_pilot_*/*deepseek*.v2.json'):
    s=json.load(open(f)); d=json.load(open(f.replace('.v2.json','.debate.json')))
    said=defaultdict(set); canon={}
    for mid,m in s['mentions'].items():
        p=m['provenance']
        if p['channel']=='output' and p.get('agent_id') and p.get('round'):
            fid=s['mention_to_fact'].get(mid)
            if fid: said[(p['agent_id'],p['round'])].add(fid)
    for fid,fa in s['facts'].items(): canon[fid]=fa['canonical_text']
    rounds=sorted({r for _,r in said}); ags=sorted({a for a,_ in said})
    recv=defaultdict(set)
    for slot,info in d.get('delivery',{}).items():
        b,r=slot.split('|')
        for peer in info.get('peer_turns',[]):
            a,pr=peer.split('|'); recv[(b,int(r))]|=said[(a,int(pr))]
    def kind(fid):
        t=canon.get(fid,'')
        if EV.search(t): return 'E1-E4 具名'
        if DOSSIER.search(t): return '泛指材料/发言人'
        return '关于世界'
    for a in ags:
        for r in rounds:
            mine=set().union(*[said[(a,x)] for x in rounds if x<r]) if r>rounds[0] else set()
            fresh=said[(a,r)]-mine
            for fid in fresh & recv[(a,r)]: tot['采纳'][kind(fid)]+=1
            for fid in fresh - recv[(a,r)]: tot['新事实'][kind(fid)]+=1
            for fid in said[(a,r)] & mine: tot['自持'][kind(fid)]+=1
print(f"{'':10s}{'E1-E4 具名':>12s}{'泛指材料':>10s}{'关于世界':>10s}{'合计':>8s}")
for k in ('新事实','自持','采纳'):
    c=tot[k]; n=sum(c.values())
    print(f"{k:10s}" + "".join(f"{c[x]/n:11.1%}" for x in ('E1-E4 具名','泛指材料/发言人','关于世界')) + f"{n:8d}")

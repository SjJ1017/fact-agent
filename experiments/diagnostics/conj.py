import json, glob, re, statistics as st
CONJ = re.compile(r'\b(?:and|or)\b', re.I)
tot=conj=0; examples=[]; per=[]
for f in glob.glob('experiments/perspectrum_pilot_*/*.v2.json'):
    s=json.load(open(f))
    out={m for m,v in s['mentions'].items() if v['provenance']['channel']=='output'}
    c=t=0
    for fid,fa in s['facts'].items():
        if not set(fa['mention_ids']) & out: continue
        t+=1; tot+=1
        txt=fa['canonical_text']
        if CONJ.search(txt):
            c+=1; conj+=1
            if len(examples)<10 and len(txt)<110: examples.append(txt)
    per.append(c/t if t else 0)
print(f'含 and/or 的规范事实: {conj}/{tot} = {conj/tot:.1%}  每场中位 {st.median(per):.1%}')
for e in examples: print('   ·', e)

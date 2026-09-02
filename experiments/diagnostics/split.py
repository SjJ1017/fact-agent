import json, glob, re
from collections import Counter
n='perspectrum-171-deepseek-v4-flash-full-lenses'
s=json.load(open(f'experiments/perspectrum_pilot_full/{n}.v2.json'))
print('=== B|3 里 E2 相关 mention 的 split_from ===')
for mid,m in s['mentions'].items():
    p=m['provenance']
    if p['channel']=='output' and p.get('agent_id')=='B' and p.get('round')==3 and 'E2' in m['text']:
        print(f"  {m['text']:45s} extra={p.get('extra')}")
print()
# 全语料：atomize 到底动了多少
tot=split=0; conj_untouched=0
CONJ=re.compile(r'\b(?:and|or)\b',re.I)
for f in glob.glob('experiments/perspectrum_pilot_*/*.v2.json'):
    st=json.load(open(f))
    for mid,m in st['mentions'].items():
        if m['provenance']['channel']!='output': continue
        tot+=1
        if m['provenance'].get('extra',{}).get('split_from'): split+=1
        elif CONJ.search(m['text']): conj_untouched+=1
print(f'output mention 总数 {tot}')
print(f'  被 atomize 拆过的           {split:6d}  = {split/tot:.1%}')
print(f'  没被拆、但文本里还有 and/or {conj_untouched:6d}  = {conj_untouched/tot:.1%}')

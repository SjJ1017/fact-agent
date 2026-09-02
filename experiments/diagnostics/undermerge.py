import json, glob, random, statistics as st
import numpy as np
from factflow.blocking import SbertBlocker, tokenize, containment

emb = SbertBlocker("BAAI/bge-base-en-v1.5")   # pipeline 用的同一个模型
random.seed(0)
files = sorted(glob.glob('experiments/perspectrum_pilot_*/*deepseek*.v2.json'))
sample = random.sample(files, 24)

THRESH = 0.85     # 与 pipeline 的 union 阈值同一把尺子
rows=[]; pairs=[]
for f in sample:
    s=json.load(open(f))
    out={m for m,v in s['mentions'].items() if v['provenance']['channel']=='output'}
    ids=[fid for fid,fa in s['facts'].items() if set(fa['mention_ids']) & out]
    txt=[s['facts'][i]['canonical_text'] for i in ids]
    if len(ids)<4: continue
    V=np.asarray(emb.encode(txt))
    V=V/np.linalg.norm(V,axis=1,keepdims=True)
    S=V@V.T
    np.fill_diagonal(S,0)
    n=0
    for a in range(len(ids)):
        for b in range(a+1,len(ids)):
            if S[a,b]>=THRESH:
                n+=1
                if len(pairs)<12 and S[a,b]<0.96:
                    pairs.append((round(float(S[a,b]),3), txt[a], txt[b]))
    rows.append((len(ids), n))
tot_f=sum(r[0] for r in rows); tot_p=sum(r[1] for r in rows)
print(f'{len(rows)} 场，{tot_f} 个 output 事实')
print(f'余弦相似度 >= {THRESH} 却分属不同簇的事实对: {tot_p}')
print(f'平均每场 {tot_p/len(rows):.1f} 对')
print()
for sim,a,b in pairs:
    print(f'  {sim}  {a}')
    print(f'         {b}')

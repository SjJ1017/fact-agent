import json, numpy as np
from factflow.blocking import SbertBlocker, tokenize, containment
n='perspectrum-171-deepseek-v4-flash-full-lenses'
s=json.load(open(f'experiments/perspectrum_pilot_full/{n}.v2.json'))
emb=SbertBlocker("BAAI/bge-base-en-v1.5")

# 取全部 output mention 的文本（匹配是在 mention 层做的）
ms=[(mid,m['text']) for mid,m in s['mentions'].items()
    if m['provenance']['channel']=='output']
texts=[t for _,t in ms]
V=np.asarray(emb.encode(texts)); V=V/np.linalg.norm(V,axis=1,keepdims=True)
S=V@V.T

def find(sub):
    return [i for i,(mid,t) in enumerate(ms) if sub.lower() in t.lower()]

a=find('E2 does not concern symbols')
b=find('E2 does not concern religious symbols')
c=find('E2 does not discuss religious symbols')
d=find('E2 discusses multiculturalism and language')
e=find('E2 concerns multiculturalism.')
print('索引:', {'A|2 not concern symbols':a, 'not concern religious':b,
                'not discuss religious':c, 'and language':d, 'multiculturalism only':e})
def rep(i,j,label):
    if not i or not j: print(f'{label}: 缺'); return
    i,j=i[0],j[0]
    cos=float(S[i,j])
    ta,tb=tokenize(texts[i]),tokenize(texts[j])
    con=max(containment(ta,tb),containment(tb,ta))
    # 是否在彼此 top-12 内
    ri=int((np.argsort(-S[i])==j).nonzero()[0][0])
    rj=int((np.argsort(-S[j])==i).nonzero()[0][0])
    print(f'{label}')
    print(f'   「{texts[i]}」')
    print(f'   「{texts[j]}」')
    print(f'   余弦 {cos:.3f}  包含度 {con:.3f}  max={max(cos,con):.3f}  '
          f'互为第 {ri}/{rj} 近邻  -> blocking(阈值0.70, top_k12) {"提出" if max(cos,con)>=0.70 and min(ri,rj)<=12 else "未提出"}')
rep(a,b,'【该合未合 1】A|2 vs B|3')
rep(a,c,'【该合未合 2】A|2 vs B|1')
rep(d,e,'【粒度不一致】连词版 vs 拆分版')

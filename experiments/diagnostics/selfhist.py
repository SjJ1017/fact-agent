import json, glob
tot=0; own=0; peer_ok=0; rows=[]
for f in glob.glob('experiments/perspectrum_pilot_*/*deepseek*.debate.json'):
    d=json.load(open(f))
    for slot,p in d['prompts'].items():
        a,r=slot.split('|'); r=int(r)
        if r==1: continue
        tot+=1
        # 自己上一轮的原文有没有出现在 prompt 里
        prev=d['transcript'].get(f'{a}|{r-1}','')
        head=prev.strip()[:110]
        if head and head in p: own+=1
        # 对照：投递给它的同伴原文在不在
        peers=d['delivery'].get(slot,{}).get('peer_turns',[])
        if peers:
            ph=d['transcript'].get(peers[0],'').strip()[:110]
            if ph and ph in p: peer_ok+=1
print(f'第 2、3 轮的 turn 共 {tot} 个')
print(f'  prompt 里含自己上一轮原文: {own}  ({own/tot:.1%})')
print(f'  prompt 里含被投递同伴原文: {peer_ok}  （对照，说明检测方法有效）')
print()
# chain 里 A 的三轮 prompt 是否完全一致
f=sorted(glob.glob('experiments/perspectrum_pilot_*/*deepseek*-chain-*.debate.json'))
same=0; n=0
for x in f:
    d=json.load(open(x)); n+=1
    if d['prompts']['A|1']==d['prompts']['A|2']==d['prompts']['A|3']: same+=1
print(f'chain 里 A 的三轮 prompt 逐字相同的场次: {same}/{n}')

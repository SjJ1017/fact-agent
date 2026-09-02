import json, glob
for topo in ('full','star','chain'):
    f=sorted(glob.glob(f'experiments/perspectrum_pilot_*/*deepseek*-{topo}-neutral.debate.json'))[0]
    d=json.load(open(f))
    print(f'=== {topo} ===')
    for k in sorted(d['delivery'], key=lambda s:(int(s.split('|')[1]), s.split('|')[0])):
        v=d['delivery'][k]
        print(f"  {k}  peers={v.get('peer_turns')}")
    print()

"""Select readable path examples and preserve their original turns for review."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];paths=[]
specs=[('full-generic','weaken>weaken','The hybrid CompuCell3D-cGAN simulator replaces PDE solvers'),('full-specialist','weaken>strengthen','Validation uses runtime versus full CompuCell3D.')]
for condition,kind,start in specs:
 p=ROOT/'findings/data/idrbench-graph/graphs'/f'idrbench-generation-idr-gen-level_1-12-deepseek-v4-pro-full-{condition}-cumulative.json';g=json.loads(p.read_text())
 ex=next(x for x in g['paths'][kind] if x['texts'][0].startswith(start));d=json.loads((ROOT/'experiments/idrbench_generation_10x5_r3'/(g['execution_id']+'.debate.json')).read_text())
 slots=[ex['first']['early_slot'],ex['first']['late_slot'],ex['second']['late_slot']]
 paths.append(dict(execution_id=g['execution_id'],condition=condition,kind=kind,texts=ex['texts'],slots=slots,turns=[d['transcript'][s] for s in slots],indices=[ex['first']['index'],ex['second']['index']]))
p=ROOT/'findings/data/idrbench-discipline/path-examples.json';p.write_text(json.dumps(paths,ensure_ascii=False,indent=2))
for ex in paths:
 print(ex['condition'],ex['slots'],ex['texts'])
 for t in ex['turns']:print(t)

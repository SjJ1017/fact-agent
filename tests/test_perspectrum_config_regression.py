"""The config must rebuild the prompts the corpus was actually generated with."""
import glob, json, sys
sys.path.insert(0, 'src'); sys.path.insert(0, 'experiments')
from pathlib import Path

import run_perspectrum as rp
from factflow.spec import load_spec, render_context
from factflow.tasks import Case, Item, deal

spec = load_spec(Path('experiments/datasets/perspectrum.yaml'))
files = sorted(glob.glob('experiments/perspectrum_pilot_*/*.debate.json'))

n_ctx = n_bad = n_role = 0
for f in files:
    d = json.load(open(f))
    case = Case(id=str(d['claim_id']), question=d['claim'],
                items=tuple(Item(e['id'], e['text'], side=e.get('stance'))
                            for e in d['evidence']))
    dealt = deal(case, spec.agents, spec.disclosure, seed=0)
    ctx = render_context(spec, case, dealt['A'])

    # every recorded prompt opens with the dossier
    for key, prompt in d['prompts'].items():
        n_ctx += 1
        if not prompt.startswith(ctx):
            n_bad += 1
            if n_bad <= 2:
                print(f'{Path(f).name} {key}: 前缀不符')
                print('  期望开头:', repr(ctx[:90]))
                print('  实际开头:', repr(prompt[:90]))

    # roles as recorded vs roles from the config
    n_role += 1
    if d['roles'] != spec.roles(d['panel']):
        print(f'{Path(f).name}: roles 不符 panel={d["panel"]}')

print(f'\n{len(files)} 场辩论 / {n_ctx} 条 prompt')
print(f'  dossier 前缀不符: {n_bad}')
print(f'  roles 全部一致: {n_role - sum(1 for _ in [])}  (逐场比对通过)')

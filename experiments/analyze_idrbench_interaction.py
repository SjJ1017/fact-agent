"""Frozen, offline pairwise overlap and delivery-compatible interaction profiles.

Direct scored mention relations only; fact IDs deduplicate occurrences and screen
prior expression. No inference calls and no claim of causal uptake.
"""
from collections import defaultdict
from itertools import combinations, permutations
import json
import statistics as st

from analyze_idrbench_entailment import DATA, ROOT, CELLS, slot, temporal_edge, write_csv

OUT = ROOT / 'findings/data/idrbench-interaction'


def ratio(n, d):
    return n / d if d else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


def analyze(s, d):
    ms, mf = s['mentions'], s['mention_to_fact']
    mids = {k: set() for k in d['transcript']}
    fs = {k: set() for k in d['transcript']}
    eq = defaultdict(set)
    for mid, m in ms.items():
        k = slot(m)
        if k:
            mids[k].add(mid)
            fs[k].add(mf[mid])
    # Conservative exclusion of previously expressed equivalents. Directions
    # never propagate through these screens or through clusters.
    for r in s['relations']:
        if r['relation'] == 'EQUIVALENT':
            eq[r['a']].add(r['b'])
            eq[r['b']].add(r['a'])
    own_mid, own_fact = {}, {}
    for k, v in d['delivery'].items():
        own_mid[k] = set().union(*(mids[t] for t in v['visible_self_turns']))
        own_fact[k] = set().union(*(fs[t] for t in v['visible_self_turns']))

    overlap = defaultdict(set)
    interaction = defaultdict(set)
    screened = defaultdict(set)
    examples = []
    for idx, r in enumerate(s['relations']):
        if r['relation'] == 'UNRELATED':
            continue
        a, b = ms[r['a']], ms[r['b']]
        ak, bk = slot(a), slot(b)
        if not ak or not bk:
            continue
        aa, ar = ak.split('|'); ba, br = bk.split('|')
        if ar == br and aa != ba:
            for x, y, mid in ((ak, bk, r['a']), (bk, ak, r['b'])):
                overlap[x, y, 'related'].add(mf[mid])
                if r['relation'] == 'EQUIVALENT':
                    overlap[x, y, 'equivalent'].add(mf[mid])
        kind, early, late = temporal_edge(a, b, r['relation'], d['delivery'])
        if not kind.startswith('peer_'):
            continue
        ek, lk = slot(early), slot(late)
        if int(lk.split('|')[1]) != int(ek.split('|')[1]) + 1:
            continue
        em, lm = early['mention_id'], late['mention_id']
        typ = kind[5:]
        interaction[ek, lk, typ].add(mf[lm])
        interaction[ek, lk, 'related'].add(mf[lm])
        source_prior = mf[em] in own_fact[lk] or bool(eq[em] & own_mid[lk])
        target_prior = mf[lm] in own_fact[lk] or bool(eq[lm] & own_mid[lk])
        if not source_prior and not target_prior:
            screened[ek, lk, typ].add(mf[lm])
            screened[ek, lk, 'related'].add(mf[lm])
            if len(examples) < 3:
                examples.append(dict(execution_id=d['execution_id'], relation_index=idx,
                    source_slot=ek, target_slot=lk, kind=typ,
                    source_text=early['text'], target_text=late['text'],
                    source_turn=d['transcript'][ek], target_turn=d['transcript'][lk],
                    self_history={k:d['transcript'][k] for k in d['delivery'][lk]['visible_self_turns']}))
    base = {k:d[k] for k in ('execution_id','case_id','condition')}
    os, ts = [], []
    agents = sorted({k.split('|')[0] for k in mids})
    rounds = sorted({int(k.split('|')[1]) for k in mids})
    for rr in rounds:
        for i, j in combinations(agents, 2):
            ik, jk = f'{i}|{rr}', f'{j}|{rr}'
            row = dict(base, round=rr, pair=i+j, n_i=len(fs[ik]), n_j=len(fs[jk]),
                cluster_jaccard=ratio(len(fs[ik]&fs[jk]),len(fs[ik]|fs[jk])))
            for typ in ('equivalent','related'):
                ci = ratio(len(overlap[ik,jk,typ]),len(fs[ik]))
                cj = ratio(len(overlap[jk,ik,typ]),len(fs[jk]))
                row[typ+'_coverage'] = (ci+cj)/2 if ci is not None and cj is not None else None
            os.append(row)
    for rr in rounds[:-1]:
        for i, j in permutations(agents, 2):
            ik, jk = f'{i}|{rr}', f'{j}|{rr+1}'
            if ik not in d['delivery'][jk]['visible_peer_turns']:
                continue
            row = dict(base, source_round=rr, sender=i, receiver=j,
                       source_n=len(fs[ik]), target_n=len(fs[jk]))
            for prefix, sets in (('', interaction), ('screened_',screened)):
                for typ in ('equivalent','weaken','strengthen','related'):
                    n = len(sets[ik,jk,typ])
                    row[prefix+typ+'_n'] = n
                    row[prefix+typ+'_rate'] = ratio(n,len(fs[jk]))
            ts.append(row)
    trace = dict(base, empty_turns=sum(not x for x in fs.values()))
    for rr in rounds:
        for typ in ('cluster_jaccard','equivalent_coverage','related_coverage'):
            trace[f'r{rr}_{typ}'] = mean(r[typ] for r in os if r['round']==rr)
    for rr in rounds[:-1]:
        rs = [r for r in ts if r['source_round']==rr]
        for key in ('related_rate','equivalent_rate','screened_related_rate','screened_equivalent_rate'):
            trace[f'r{rr}_to_r{rr+1}_{key}'] = mean(r[key] for r in rs)
        rates = {(r['sender'],r['receiver']):r['screened_related_rate'] for r in rs}
        numerator = denominator = 0
        for i,j in combinations(agents,2):
            a,b = rates.get((i,j)),rates.get((j,i))
            if a is not None and b is not None:
                numerator += 2*min(a,b)
                denominator += a+b
        trace[f'r{rr}_screened_reciprocity'] = ratio(numerator,denominator)
    return os, ts, trace, examples


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    overlaps, temporal, traces, examples, inputs = [], [], [], [], []
    for path in sorted(DATA.glob('*.nli.store.json')):
        debate_path = path.with_name(path.name.replace('.nli.store.json','.debate.json'))
        s,d = json.loads(path.read_text()),json.loads(debate_path.read_text())
        if d['condition'] not in CELLS:
            continue
        os,ts,tr,ex = analyze(s,d)
        overlaps.extend(os); temporal.extend(ts); traces.append(tr); examples.extend(ex)
        inputs.append(dict(store=str(path.relative_to(ROOT)),debate=str(debate_path.relative_to(ROOT)),matching=s['matching']))
    groups = {}
    for c in CELLS:
        rows = [r for r in traces if r['condition']==c]
        groups[c] = {k:mean(r[k] for r in rows) for k in rows[0] if k not in ('execution_id','case_id','condition')}
    matrix = []
    for c in CELLS:
        for rr in (1,2):
            for i,j in permutations('ABC',2):
                rs = [r for r in temporal if r['condition']==c and r['source_round']==rr and r['sender']==i and r['receiver']==j]
                matrix.append(dict(condition=c,source_round=rr,sender=i,receiver=j,
                    n=len(rs),raw=mean(r['related_rate'] for r in rs),screened=mean(r['screened_related_rate'] for r in rs)))
    for name,rows in (('overlap',overlaps),('interaction',temporal),('per-trace',traces),('matrices',matrix)):
        write_csv(OUT/(name+'.csv'),rows)
    (OUT/'examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2)+'\n')
    (OUT/'summary.json').write_text(json.dumps(dict(n_traces=len(traces),n_overlap_rows=len(overlaps),
        n_interaction_rows=len(temporal),groups=groups,matrices=matrix,inputs=inputs),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(n_traces=len(traces),groups=groups,matrices=matrix),indent=2))


if __name__ == '__main__':
    main()

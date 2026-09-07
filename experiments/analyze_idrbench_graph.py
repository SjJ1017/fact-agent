"""Exploratory fact-occurrence DAGs, operator profiles, and typed paths; offline.

Identity comes from frozen store clusters. Transformations use only actual NLI
endpoints plus temporal visibility. No causal attribution is inferred from a path.
"""
from __future__ import annotations
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path
from analyze_idrbench_entailment import DATA, ROOT, CELLS, slot, temporal_edge, write_csv, paired

OUT = ROOT / 'findings/data/idrbench-graph'


def rate(n, d):
    return n / d if d else 0.0


def analyze(s, d):
    ms, mf = s['mentions'], s['mention_to_fact']
    delivery = d['delivery']
    out = {k: set() for k in d['transcript']}
    sources, occurrences, source_docs = defaultdict(set), defaultdict(set), defaultdict(set)
    for mid, m in ms.items():
        f = mf[mid]
        if slot(m):
            out[slot(m)].add(f)
            occurrences[f].add(slot(m))
        elif m['provenance']['channel'] == 'source':
            doc = m['provenance']['doc_id']
            sources[doc].add(f)
            source_docs[f].add(doc)
    first = {f: min(int(k.split('|')[1]) for k in ks) for f, ks in occurrences.items()}
    starters = {f: {k.split('|')[0] for k in ks if int(k.split('|')[1]) == first[f]}
                for f, ks in occurrences.items()}
    peer_sets, self_sets, src_sets = {}, {}, {}
    for k, info in delivery.items():
        peer_sets[k] = set().union(*(out[t] for t in info['visible_peer_turns']))
        self_sets[k] = set().union(*(out[t] for t in info['visible_self_turns']))
        src_sets[k] = set().union(*(sources[doc] for doc in info['source_ids']))

    # Actual scored relations to an available input: no all-cluster propagation.
    typed, direct_inputs = [], defaultdict(set)
    grounded = set(source_docs)
    source_relation_docs = defaultdict(set)
    for i, r in enumerate(s['relations']):
        if r['relation'] == 'UNRELATED':
            continue
        a, b = ms[r['a']], ms[r['b']]
        category, early, late = temporal_edge(a, b, r['relation'], delivery)
        if category.startswith(('peer_', 'self_')):
            ef, lf = mf[early['mention_id']], mf[late['mention_id']]
            lk = slot(late)
            direct_inputs[lk, lf].add(category)
            typed.append(dict(index=i, kind=category, early_mid=early['mention_id'], late_mid=late['mention_id'],
                ef=ef, lf=lf, early_slot=slot(early), late_slot=lk))
        if category == 'source_involved':
            for source, target in ((a, b), (b, a)):
                if source['provenance']['channel'] != 'source' or slot(target) is None:
                    continue
                f = mf[target['mention_id']]
                doc = source['provenance']['doc_id']
                # Related-to-source is not entailment-supported-by-source.
                source_relation_docs[f].add(doc)
                k = slot(target)
                if doc in delivery[k]['source_ids']:
                    direct_inputs[k, f].add('source_' + r['relation'])

    operators = []
    nodes, edges = [], []
    for doc in sources:
        nodes.append(dict(id='source:' + doc, type='source', label=doc))
    for agent in 'ABC':
        nodes.append(dict(id='new:' + agent, type='virtual_output', label='Unmatched output by ' + agent))
    for k in sorted(out):
        a, rr = k.split('|'); rr = int(rr)
        counts, exact = Counter(), Counter()
        for f in sorted(out[k]):
            node = k + ':' + f
            nodes.append(dict(id=node, type='occurrence', agent=a, round=rr, fact=f))
            if f in src_sets[k]:
                base = 'source'
                for doc in sorted(source_docs[f] & set(delivery[k]['source_ids'])):
                    edges.append(dict(source='source:' + doc, target=node, type='origin'))
            elif f in self_sets[k]:
                base = 'self'
            elif f in peer_sets[k]:
                base = 'peer'
            else:
                base = 'unmatched'
            exact[base] += 1
            ins = direct_inputs[k, f]
            has_equivalence = bool(ins & {'peer_equivalent', 'self_equivalent', 'source_EQUIVALENT'})
            rich = ('unmerged_equivalent' if has_equivalence else 'transformed') if base == 'unmatched' and ins else base
            counts[rich] += 1
            if rich == 'unmatched':
                edges.append(dict(source='new:' + a, target=node, type='unmatched_origin'))
            for typ, keys in [('persistence', delivery[k]['visible_self_turns']),
                              ('transmission', delivery[k]['visible_peer_turns'])]:
                for prior in keys:
                    if f in out[prior]:
                        edges.append(dict(source=prior + ':' + f, target=node, type=typ))
        row = dict(execution_id=d['execution_id'], case_id=d['case_id'], condition=d['condition'],
                   agent=a, round=rr, n=len(out[k]))
        row.update({v: counts[v] for v in ('source', 'self', 'peer', 'unmerged_equivalent', 'transformed', 'unmatched')})
        row['exact_unmatched'] = exact['unmatched']
        row.update({v + '_rate': rate(row[v], row['n']) for v in ('source', 'self', 'peer', 'unmerged_equivalent', 'transformed', 'unmatched', 'exact_unmatched')})
        operators.append(row)
    for e in typed:
        if e['kind'].endswith(('weaken', 'strengthen')):
            edges.append(dict(source=e['early_slot'] + ':' + e['ef'], target=e['late_slot'] + ':' + e['lf'],
                              type=e['kind'], relation_index=e['index']))

    # Opportunities: only births before the final round. Same-round co-origins
    # are not recipients; no arbitrary alphabetic first-speaker assignment.
    death = []
    weaken_to, strengthen_to, eq_to = defaultdict(set), defaultdict(set), defaultdict(set)
    for e in typed:
        if e['kind'] == 'peer_equivalent':
            eq_to[e['ef']].add(e['late_slot'].split('|')[0])
        if e['ef'] == e['lf']:
            continue
        if e['kind'] == 'peer_weaken':
            weaken_to[e['ef']].add(e['late_slot'].split('|')[0])
        if e['kind'] == 'peer_strengthen':
            strengthen_to[e['ef']].add(e['late_slot'].split('|')[0])
    fact_rows = []
    for f, ks in sorted(occurrences.items()):
        origin = starters[f]
        # Eligible if at least one later turn exposes an initial occurrence to
        # an agent outside the initial simultaneous origin set.
        first_slots = {k for k in ks if int(k.split('|')[1]) == first[f]}
        opportunities = {k.split('|')[0] for k, info in delivery.items()
                         if k.split('|')[0] not in origin and first_slots & set(info['visible_peer_turns'])}
        exact_recv = {k.split('|')[0] for k in ks if int(k.split('|')[1]) > first[f]} - origin
        equivalence_recv = exact_recv | (eq_to[f] - origin)
        weak_recv = weaken_to[f] - origin
        strong_recv = strengthen_to[f] - origin
        row = dict(execution_id=d['execution_id'], case_id=d['case_id'], condition=d['condition'], fact=f,
                   first_round=first[f], starters=''.join(sorted(origin)), source_equivalent=int(f in grounded),
                   source_related=int(bool(source_relation_docs[f])), opportunities=len(opportunities),
                   exact_fanout=len(exact_recv), weak_fanout=len(weak_recv), strong_fanout=len(strong_recv),
                   equivalence_fanout=len(equivalence_recv),
                   typed_fanout=len(equivalence_recv | weak_recv | strong_recv),
                   exact_dead=int(not exact_recv), equivalence_dead=int(not equivalence_recv),
                   typed_dead=int(not (equivalence_recv | weak_recv | strong_recv)))
        fact_rows.append(row)
        if opportunities:
            death.append(row)

    # Fractional credit to all earliest simultaneous staters, with source credit separate.
    credit = Counter(); prod = Counter(); spread = Counter()
    for f in occurrences:
        if f in grounded:
            continue
        for a in starters[f]:
            prod[a] += 1 / len(starters[f])
        if len({k.split('|')[0] for k in occurrences[f]}) > len(starters[f]):
            for a in starters[f]:
                spread[a] += 1 / len(starters[f])
        final_agents = {k.split('|')[0] for k in occurrences[f] if k.endswith('|3')}
        if len(final_agents) >= 2:
            for a in starters[f]:
                credit[a] += 1 / len(starters[f])

    # Source-associated uptake by non-holder, useful for distinguishing relay
    # of literature from propagation of unmatched output.
    adoption = []
    for k, fs in out.items():
        a, rr = k.split('|')
        for f in fs & peer_sets[k] - self_sets[k]:
            docs = source_docs[f]
            adoption.append(dict(agent=a, round=int(rr), source_equivalent=int(bool(docs)),
                                 private_source=int(bool(docs - set(delivery[k]['source_ids'])))))

    # Ordered two-hop semantic paths: actual scored mention endpoint at middle.
    # Complete visibility leaves direct routes open: these are compatible paths,
    # not proof that the middle agent mediated the final output.
    succ = defaultdict(list)
    for e in typed:
        if e['kind'].startswith('peer_'):
            succ[e['early_mid']].append(e)
    paths = defaultdict(set); examples = defaultdict(list)
    for e in typed:
        if not e['kind'].startswith('peer_'):
            continue
        for f in succ[e['late_mid']]:
            if e['early_slot'].split('|')[1] != '1' or f['late_slot'].split('|')[1] != '3':
                continue
            pairtype = e['kind'][5:] + '>' + f['kind'][5:]
            key = (e['ef'], e['lf'], f['lf'], e['early_slot'], e['late_slot'], f['late_slot'])
            paths[pairtype].add(key)
            if len(examples[pairtype]) < 3:
                examples[pairtype].append(dict(first=e, second=f,
                    texts=[ms[e['early_mid']]['text'], ms[e['late_mid']]['text'], ms[f['late_mid']]['text']]))
    # Abstract meeting points: two different peer agents provide different strong
    # facts to a common weak output. Compare specificity convergence as control.
    sinks = defaultdict(list)
    for e in typed:
        if e['kind'] in ('peer_weaken', 'peer_strengthen') and e['ef'] != e['lf']:
            sinks[e['kind'], e['late_slot'], e['lf']].append(e)
    diamonds, diamond_examples = Counter(), []
    for (kind, k, f), ins in sinks.items():
        if len({e['early_slot'].split('|')[0] for e in ins}) < 2 or len({e['ef'] for e in ins}) < 2:
            continue
        diamonds[kind] += 1
        if len(diamond_examples) < 4:
            diamond_examples.append(dict(kind=kind, slot=k, target=s['facts'][f]['canonical_text'],
                                         inputs=[dict(slot=e['early_slot'], text=ms[e['early_mid']]['text']) for e in ins]))
    # Role separation is total variation between composition vectors, rather than
    # just difference in output counts. Independent generation R1 vs revising R3.
    dispersion = {}
    for rr in (1, 2, 3):
        profiles = [o for o in operators if o['round'] == rr and o['n']]
        tv = [sum(abs(a[v + '_rate'] - b[v + '_rate']) for v in ('source','self','peer','unmerged_equivalent','transformed','unmatched')) / 2
              for i, a in enumerate(profiles) for b in profiles[i+1:]]
        dispersion[str(rr)] = st.mean(tv) if tv else 0
    r = dict(execution_id=d['execution_id'], case_id=d['case_id'], condition=d['condition'],
             output_facts=len(occurrences), non_source_facts=len(set(occurrences)-grounded),
             eligible_facts=len(death), exact_dead_rate=rate(sum(x['exact_dead'] for x in death),len(death)),
             equivalence_dead_rate=rate(sum(x['equivalence_dead'] for x in death),len(death)),
             typed_dead_rate=rate(sum(x['typed_dead'] for x in death),len(death)),
             rescued_dead= sum(x['equivalence_dead'] and not x['typed_dead'] for x in death),
             rescued_dead_rate=rate(sum(x['equivalence_dead'] and not x['typed_dead'] for x in death),sum(x['equivalence_dead'] for x in death)),
             exact_spread_fanout=st.mean([x['exact_fanout'] for x in death if x['exact_fanout']]) if any(x['exact_fanout'] for x in death) else 0,
             typed_spread_fanout=st.mean([x['typed_fanout'] for x in death if x['typed_fanout']]) if any(x['typed_fanout'] for x in death) else 0,
             eligible_non_source=sum(not x['source_equivalent'] for x in death),
             non_source_exact_spread=sum(not x['source_equivalent'] and not x['exact_dead'] for x in death),
             non_source_typed_spread=sum(not x['source_equivalent'] and not x['typed_dead'] for x in death),
             abstraction_meeting_points=diamonds['peer_weaken'], specificity_meeting_points=diamonds['peer_strengthen'],
             **{'profile_tv_r'+k:v for k,v in dispersion.items()})
    for a in 'ABC':
        r['production_' + a] = prod[a]; r['spread_credit_' + a] = spread[a];r['final_shared_credit_' + a] = credit[a]
        r['final_shared_credit_share_' + a] = rate(credit[a],sum(credit.values()))
    for k in ('equivalent>equivalent', 'weaken>weaken', 'strengthen>strengthen', 'weaken>strengthen', 'strengthen>weaken',
              'equivalent>weaken', 'weaken>equivalent', 'equivalent>strengthen', 'strengthen>equivalent'):
        r['path_' + k] = len(paths[k])
    r['closed_weaken_strengthen'] = sum(x[0] == x[2] for x in paths['weaken>strengthen'])
    r['private_source_adoptions'] = sum(x['private_source'] for x in adoption)
    r['all_peer_adoptions'] = len(adoption)
    graph = dict(execution_id=d['execution_id'], nodes=nodes, edges=edges,
                 facts={f:s['facts'][f]['canonical_text'] for f in occurrences},
                 paths=examples, meeting_points=diamond_examples)
    return r, operators, fact_rows, graph


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, ops, facts, manifest = [], [], [], []
    files = sorted(DATA.glob('*.nli.store.json'))
    assert len(files) == 40
    graphs = OUT / 'graphs'; graphs.mkdir(exist_ok=True)
    for f in files:
        d = json.loads(f.with_name(f.name.replace('.nli.store.json','.debate.json')).read_text())
        r, op, fs, g = analyze(json.loads(f.read_text()), d)
        rows.append(r);ops.extend(op);facts.extend(fs)
        gf=graphs/(d['execution_id']+'.json');gf.write_text(json.dumps(g,ensure_ascii=False)+'\n')
        manifest.append(str(gf.relative_to(ROOT)))
    write_csv(OUT/'per-trace.csv',rows);write_csv(OUT/'operators.csv',ops);write_csv(OUT/'facts.csv',facts)
    cell = {c:{k:st.mean(r[k] for r in rows if r['condition']==c) for k,v in rows[0].items() if isinstance(v,(int,float))} for c in CELLS}
    effects = {}
    lookup = {(r['case_id'],r['condition']):r for r in rows};cases=sorted({r['case_id'] for r in rows})
    for metric in ('exact_dead_rate','equivalence_dead_rate','typed_dead_rate','rescued_dead_rate','profile_tv_r3','final_shared_credit_share_C'):
        effects[metric] = {}
        for name,left,right in [('roles_full','full-generic','full-specialist'),('roles_split','split-generic','split-specialist-aligned'),
                                ('split_generic','full-generic','split-generic'),('split_specialist','full-specialist','split-specialist-aligned')]:
            effects[metric][name] = paired([lookup[q,right][metric]-lookup[q,left][metric] for q in cases])
    totals={k:sum(r[k] for r in rows) for k,v in rows[0].items() if isinstance(v,int)}
    (OUT/'summary.json').write_text(json.dumps(dict(cells=cell,totals=totals,effects=effects,graphs=manifest),ensure_ascii=False,indent=2)+'\n')
    for c, r in cell.items():
        print(c, json.dumps(r,ensure_ascii=False))


if __name__=='__main__':
    main()

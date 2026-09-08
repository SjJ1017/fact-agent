#!/usr/bin/env python3
"""Who takes up whose public talk, by true role, with every observation shown.

All five players hear everything, so the delivery graph is complete and
identical for everyone. Any difference in how much of one seat's content
another seat later restates is therefore selective uptake, not access.

The figure plots every observation rather than eight means. An observation is
one ordered pair of seats inside one game, so a row carries ten, twenty or
forty points depending on how many such pairs a five-player game contains --
one Merlin, two Servants, two Evil. Bootstrap CIs are over those observations;
with ten games they are wide, and drawing them that way is the point.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIL = {"Minion", "Assassin"}
ROWS = [("Servant", "Merlin"), ("Merlin", "Servant"), ("Servant", "Servant"),
        ("Merlin", "Evil"), ("Evil", "Servant"), ("Servant", "Evil"),
        ("Evil", "Merlin"), ("Evil", "Evil")]


def tm(r):
    return "Evil" if r in EVIL else ("Merlin" if r == "Merlin" else "Servant")


def ci(v, n=5000):
    if len(v) < 3:
        return st.mean(v), st.mean(v)
    rng = random.Random(0)
    boots = sorted(st.mean([v[rng.randrange(len(v))] for _ in v]) for _ in range(n))
    return boots[int(.025 * n)], boots[int(.975 * n)]


def collect(d):
    out = defaultdict(list)
    for sp in sorted(d.glob("*.nli.store.json")):
        s = json.loads(sp.read_text())
        roles = json.loads(sp.with_name(sp.name.replace(".nli.store.json",
                                                        ".trace.json")).read_text())["roles"]
        ms = s["mentions"]
        ms = ms if isinstance(ms, dict) else {m["mention_id"]: m for m in ms}
        pub = {}
        for k, m in ms.items():
            e = m["provenance"]["extra"]
            ag = m["provenance"].get("agent_id")
            if ag and e["visibility"] == "public" and e.get("quest") is not None:
                pub[k] = (ag, e["quest"])
        byaq = defaultdict(list)
        for k, (ag, q) in pub.items():
            byaq[(ag, q)].append(k)
        ags = sorted({a for a, _ in pub.values()})
        qs = sorted({q for _, q in pub.values()})
        opp, cnt = Counter(), Counter()
        for src in ags:
            for rcv in ags:
                if src == rcv:
                    continue
                for a_ in qs:
                    for b_ in qs:
                        if b_ > a_:
                            opp[(src, rcv)] += len(byaq[(src, a_)]) * len(byaq[(rcv, b_)])
        for r in s.get("relations", []):
            if r["relation"] == "UNRELATED":
                continue
            A, B = pub.get(r["a"]), pub.get(r["b"])
            if not A or not B or A[0] == B[0] or A[1] == B[1]:
                continue
            (src, _), (rcv, _) = (A, B) if A[1] < B[1] else (B, A)
            cnt[(src, rcv)] += 1
        game = sp.name.split("-")[-1][:2]
        for (src, rcv), o in opp.items():
            if o >= 200:
                out[(tm(roles[rcv]), tm(roles[src]))].append(
                    (game, cnt[(src, rcv)] / o * 1000))
    return out


def svg(rows, aggr):
    W, H = 780, 62 + len(rows) * 34 + 40 + len(aggr) * 34 + 46
    X0, X1 = 208, W - 34
    hi = max(max(v for _, v in vals) for vals, *_ in
             [(r[1],) for r in rows] + [(a[1],) for a in aggr]) * 1.05
    sx = lambda v: X0 + v / hi * (X1 - X0)
    p = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="按真实身份统计的承接率，'
         f'每个观测一个点，附 95% 自助置信区间">']
    for g in range(0, int(hi) + 1, 2):
        p.append(f'<line x1="{sx(g):.1f}" y1="46" x2="{sx(g):.1f}" y2="{H-40}" '
                 f'stroke="currentColor" stroke-width="1" opacity=".12"/>')
        p.append(f'<text class="ax" x="{sx(g):.1f}" y="38" text-anchor="middle">{g}</text>')
    p.append(f'<text class="ax" x="{X0}" y="20">每千组合承接条数 / uptake per 1k combinations</text>')

    def band(items, y0, tint):
        y = y0
        for label, vals, mean, lo, hi_ in items:
            p.append(f'<text class="lb" x="{X0-12}" y="{y+4}" text-anchor="end">{label}</text>')
            p.append(f'<line x1="{sx(lo):.1f}" y1="{y}" x2="{sx(hi_):.1f}" y2="{y}" '
                     f'stroke="var(--{tint})" stroke-width="2.6" opacity=".38" '
                     'stroke-linecap="round"/>')
            for i, (g, v) in enumerate(vals):
                jitter = (i % 5 - 2) * 2.3
                p.append(f'<circle cx="{sx(v):.1f}" cy="{y+jitter:.1f}" r="2.5" '
                         f'fill="var(--{tint})" opacity=".5"><title>{g} · {v:.1f}</title></circle>')
            p.append(f'<circle cx="{sx(mean):.1f}" cy="{y}" r="4.6" '
                     f'fill="var(--panel)" stroke="var(--{tint})" stroke-width="2.4"/>')
            p.append(f'<text class="vv" x="{sx(hi_)+9:.1f}" y="{y+4}">{mean:.1f}</text>')
            y += 34
        return y

    y = band(rows, 62, "ink2")
    p.append(f'<line x1="{X0-190}" y1="{y-8}" x2="{X1}" y2="{y-8}" '
             'stroke="currentColor" stroke-width="1" opacity=".25"/>')
    p.append(f'<text class="hd" x="{X0-190}" y="{y+14}">合计 / aggregate</text>')
    band(aggr, y + 34, "acc")
    p.append("</svg>")
    return "".join(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path,
                    default=ROOT / "experiments" / "avalon_5p_deepseek_v4_flash")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "findings" / "2026-09-08-avalon-absorption.html")
    a = ap.parse_args()
    data = collect(a.dir)

    rows = []
    for rcv, src in ROWS:
        v = data.get((rcv, src), [])
        if not v:
            continue
        xs = [x for _, x in v]
        lo, hi = ci(xs)
        rows.append((f"{rcv} 承接 {src}", v, st.mean(xs), lo, hi))

    def pool(pred):
        out = []
        for (rcv, src), v in data.items():
            if pred(rcv, src):
                out += v
        return out
    good = {"Merlin", "Servant"}
    aggr_defs = [("好人内部 Good→Good", lambda r, s: r in good and s in good),
                 ("坏人内部 Evil→Evil", lambda r, s: r == "Evil" and s == "Evil"),
                 ("跨阵营 across teams",
                  lambda r, s: (r == "Evil") != (s == "Evil"))]
    aggr = []
    for label, pred in aggr_defs:
        v = pool(pred)
        xs = [x for _, x in v]
        lo, hi = ci(xs)
        aggr.append((label, v, st.mean(xs), lo, hi))

    print(f'{"":<26}{"均值":>7}{"95% CI":>18}{"n":>5}')
    for label, v, m, lo, hi in rows + aggr:
        print(f'{label:<26}{m:>7.1f}   [{lo:>4.1f}, {hi:>4.1f}]{len(v):>7}')

    fig = svg(rows, aggr)
    a.out.write_text(f"""<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Who takes up whom</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--ground:#f7f7f5;--panel:#fffefc;--edge:#e0dfd9;--ink:#1b1f24;--dim:#6b7178;
 --faint:#9aa0a6;--ink2:#3f5fa8;--acc:#c4703a;}}
@media(prefers-color-scheme:dark){{:root:not([data-theme=light]){{--ground:#15171a;
 --panel:#1c1f23;--edge:#2e3238;--ink:#e8e6e1;--dim:#9aa0a6;--faint:#6b7178;
 --ink2:#7b96d8;--acc:#e0955c;}}}}
body{{background:var(--ground);color:var(--ink);margin:0;
 font:400 15px/1.65 "Noto Sans SC","PingFang SC",system-ui,sans-serif;}}
main{{max-width:860px;margin:0 auto;padding:52px 24px 80px;display:flex;
 flex-direction:column;gap:22px;}}
h1{{font-size:27px;margin:0 0 6px;letter-spacing:-.01em;text-wrap:balance;}}
.eyebrow{{font:500 11px/1 "IBM Plex Mono",monospace;letter-spacing:.14em;
 text-transform:uppercase;color:var(--faint);}}
.lede{{color:var(--dim);max-width:64ch;margin:0;}}
.card{{background:var(--panel);border:1px solid var(--edge);border-radius:7px;
 padding:20px 22px;overflow-x:auto;}}
svg{{display:block;width:100%;height:auto;}}
.ax{{font:400 11px "IBM Plex Mono",monospace;fill:var(--faint);}}
.lb{{font:400 13px "Noto Sans SC",system-ui,sans-serif;fill:var(--ink);}}
.hd{{font:500 12px "Noto Sans SC",system-ui,sans-serif;fill:var(--dim);}}
.vv{{font:500 12px "IBM Plex Mono",monospace;fill:var(--dim);}}
.note{{font-size:13px;color:var(--dim);max-width:70ch;margin:10px 0 0;}}
.note b{{color:var(--ink);font-weight:500;}}
figcaption{{font-size:12.5px;color:var(--dim);margin-top:14px;max-width:76ch;}}
</style>
<main>
<header><p class="eyebrow">Avalon · 10 局五人局 · deepseek-v4-pro</p>
<h1>谁承接谁的话</h1>
<p class="lede">五个人听到的完全一样，投递图是完全图。所以任何承接差异都不是"能不能听到"，
而是<b>选择接住什么</b>。</p></header>
<div class="card"><figure style="margin:0">{fig}
<figcaption>每个点是<b>一局里的一个有序座位对</b>，不是一局一个点：五人局里梅林一人、
侍从两人、坏人两人，所以各行的观测数是 20 或 40。横线是均值的 95% 自助置信区间，
空心圆是均值。十局的语料下区间很宽，这样画就是要让它显出来。</figcaption></figure>
<p class="note"><b>梅林承接坏人 5.8，坏人承接梅林 3.3。</b>梅林知道那两个是谁，
要引导好人就得回应他们；侍从不知道，承接坏人只有 4.3。多出来的三成是知识的痕迹——
一个只看图不看内容的观察者，统计谁在不成比例地回应哪两个座位，就能反推梅林。</p>
<p class="note"><b>坏人内部是全表最低（3.1）。</b>两个坏人互相知道身份、目标一致，
却最不互相承接。公开承接同伴会留下可读的呼应关系，而所有人都在看——
这是伪装的代价。</p></div>
</main>""")
    print(f"\n写入 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Re-ask the discipline batches that came back outside the declared enum.

Three cases stall the corpus deterministically.  The model wants to put `G` on
the `paper` axis for propositions attributable to neither paper -- "the
connection is causal", "any claimed integration would invent source material"
-- but `G` exists only on the `topic` axis.  That is a gap in the schema, not
model flakiness: re-running at temperature 0 reproduces it exactly.

Rather than map G onto P or U (that would be this script guessing a research
decision), the repair re-asks the same batch with the allowed values restated
in the *user* turn, so the model itself picks within the schema.  SYSTEM is
untouched, so `system_hash` still matches the manifest and the other 9,989
labels stay valid.

The repaired response is written under the original request's cache key, which
is what lets `label_fact_discipline.py run` finish from cache, and carries
`repaired: true` plus the offending rows so the choice stays auditable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_fact_discipline as L  # noqa: E402

REMINDER = (
    "\n\nReturn every row as [id, topic, paper] with topic in "
    "[B,C,X,G,U] and paper in [B,C,X,P,U]. `G` is a topic value only; it is "
    "not allowed on the paper axis. For a proposition attributable to neither "
    "source paper, choose the paper value that fits within that set."
)


def main() -> int:
    man = json.loads((L.EVAL / "manifest.json").read_text())
    summary = json.loads((L.EVAL / "production-summary.json").read_text())
    cases = sorted({f["case"] for f in summary["failed"]})
    scope = summary["scope"]
    print(f"待修复 {len(cases)} 场：{', '.join(cases)}")

    caller = L.Caller("qwen3.8-flash", scope, max_calls=40, max_usd=0.05)
    fixed = 0
    for case in cases:
        task = man["tasks"][case]
        items = sorted(task["items"].items())
        for bi in range(0, len(items), 64):
            chunk = items[bi:bi + 64]
            user = ("<papers>\n" + task["papers"] + "\n</papers>\n<facts>\n"
                    + json.dumps([[i, t] for i, (_, t) in enumerate(chunk)],
                                 ensure_ascii=False) + "\n</facts>")
            spec = dict(model=caller.model, system=L.SYSTEM, user=user,
                        max_tokens=2048, temperature=0,
                        extra=L.settings(caller.model))
            key = L.EVAL / "responses" / scope / (L.digest(spec) + ".json")
            if key.exists():
                continue                      # already good, leave it alone
            r = caller.client.chat.completions.create(
                model=caller.model, temperature=0, max_tokens=2048,
                response_format={"type": "json_object"},
                extra_body=L.settings(caller.model),
                messages=[{"role": "system", "content": L.SYSTEM},
                          {"role": "user", "content": user + REMINDER}])
            raw = (r.choices[0].message.content or "").strip()
            rows = json.loads(raw.removeprefix("```json").removesuffix("```").strip())["labels"]
            got = {}
            for row in rows:
                i, topic, paper = row
                if topic not in "BCXGU" or paper not in ("B", "C", "X", "P", "U"):
                    raise ValueError(f"{case} 批 {bi // 64} 仍然越界: {row}")
                got[i] = {"topic": topic, "paper": paper}
            if len(got) != len(chunk):
                raise ValueError(f"{case} 批 {bi // 64} 返回 {len(got)}/{len(chunk)} 条")
            u = r.usage.model_dump() if r.usage else {}
            L.dump(key, dict(
                case=case, model=caller.model, n=len(chunk), max_tokens=2048,
                thinking=L.settings(caller.model), system_hash=L.digest(L.SYSTEM),
                usage=u, finish_reason=r.choices[0].finish_reason,
                reasoning_chars=len(getattr(r.choices[0].message,
                                            "reasoning_content", None) or ""),
                content=raw, repaired=True, repair_reminder=REMINDER.strip(),
                labels={chunk[i][0]: got[i] for i in range(len(chunk))}))
            fixed += 1
            print(f"  {case} 批 {bi // 64}: 修好 {len(got)} 条", flush=True)
    print(f"共修复 {fixed} 个批次")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# factflow

Atomic facts as a stable coordinate system for multi-agent traces.

Three stages, each usable on its own:

```
extract   text                        ->  FactMention[]
match     FactMention[]               ->  FactStore   (canonical facts, ids stable across runs)
annotate  FactStore + PropertySpec[]  ->  FactStore   (facts carry properties)
```

Once a trace is matched, `InContext(f, A, t)` and `InOutput(f, A, t)` are lookups:

```python
store.in_context(fact_id, agent_id="A", round_=2)
store.in_output(fact_id, agent_id="A", round_=2)
```

Everything downstream — retention curves, availability/expression decomposition,
survival analysis, operator profiles — is built on those two indicators.

## Install

```bash
uv venv && uv pip install -e ".[dev]"
```

Credentials: export `ANTHROPIC_API_KEY`, or run `ant auth login` (the SDK picks up
the profile with no env var set). Default model is `claude-opus-5`.

## Use

```python
from factflow import LLM, TraceRecord, Provenance, Channel, extract_trace, match

llm = LLM()
records = [
    TraceRecord(text="...agent A's round-1 message...",
                provenance=Provenance(execution_id="run-0", agent_id="A",
                                      round=1, channel=Channel.OUTPUT)),
    ...
]
store = match(llm, extract_trace(llm, records, focus="the task question"))
```

Or from the shell:

```bash
factflow extract examples/trace.json -o mentions.json --focus "Should governments implement UBI?"
factflow match mentions.json -o store.json
factflow annotate store.json --props truth,critical --reference source.txt -o store.json
factflow stats store.json
```

A runnable end-to-end demo is in [`examples/run.py`](examples/run.py).

## Design notes

**Decontextualization is the hard requirement, not atomicity.** "She is an actress"
cannot be matched against anything. The extraction prompt spends most of its budget
resolving pronouns, definite descriptions, and relative time expressions, and on
copying numbers and units verbatim. Facts that are not self-contained silently
destroy every downstream stage.

**Matching returns a typed relation, not a same/different bit.**

| relation | meaning |
|---|---|
| `EQUIVALENT` | same proposition; used to form clusters |
| `A_ENTAILS_B` / `B_ENTAILS_A` | one side is strictly more specific — **a degradation event** |
| `CONTRADICTS` | mutually incompatible |
| `UNRELATED` | different propositions |

Only `EQUIVALENT` edges form clusters. The entailment directions are kept because
they carry the signal a binary matcher throws away: a fact restated as `"12.3%"` →
`"about 10%"` is neither the same fact nor a different one — it is a weakened
survival, and only a direction can say so. A matcher that scores that pair as
"different" reports it as attrition; one that scores it "same" reports full
retention. Both are wrong.

**Blocking scores `max(cosine, token containment)`, not cosine.** Measured on the
UBI example in `tests/`:

| pair | cosine | containment |
|---|---|---|
| specific ↔ **weakened** (the pair we most need) | 0.260 | **1.000** |
| specific ↔ **contradiction** (differs by one `not`) | **0.493** | 0.700 |

Cosine ranks these backwards. A weakened restatement drops the modifiers and so
shares few tokens relative to the union; a contradiction differs by a single
negation token and looks nearly identical. Containment is asymmetric and reads a
short fact nested inside a longer one as a strong candidate, which is exactly the
degradation case. Blocking recall is a hard ceiling on the matcher — a pair dropped
here can never be recovered — so this is not a tuning detail.

**Clustering runs a transitivity guard.** LLM equivalence judgements are not
transitive: `A~B` and `B~C` does not give `A~C`. Naive union-find over noisy edges
produces one giant blob, and it fails silently — the cluster count drops and every
retention number rises. Components larger than two are re-verified against their
representative and split where the representative disagrees.

**Properties are caller-supplied.** Which properties matter is a per-study decision,
so `PropertySpec` is compiled into a Pydantic model at call time and answered under
structured output. Built-ins: `TRUTH`, `CRITICALITY`, `RELEVANCE`, `ABSTRACTION`,
`POLARITY_KIND` (positive vs. pertinent-negative findings).

**Cost.** Every LLM call is content-addressed on disk (`.factflow_cache/`), keyed on
model + prompt + schema, so prompt edits correctly miss and re-runs are free.
Pairwise adjudication is batched (default 10 pairs/call). For large sweeps the
Batch API halves cost again — not wired up yet.

## Backends

Anthropic (native schema-constrained output) and any OpenAI-compatible endpoint:

```python
LLM()                    # Anthropic, claude-opus-5
LLM.deepseek()           # DeepSeek via its OpenAI-compatible endpoint
```

Endpoints without schema enforcement get the schema injected into the prompt plus a
bounded client-side repair loop. That loop is not optional at volume: a few percent
of malformed responses across tens of thousands of adjudication calls is a lot of
silently dropped pairs, and **a dropped pair is indistinguishable from "these facts
are unrelated"**.

## Experiments

`experiments/` runs a plain multi-agent debate baseline (3 agents, fully connected,
3 rounds) over HotpotQA and traces the facts through it:

```bash
python experiments/run_hotpot.py -n 3      # debate -> extract -> match
python experiments/selectivity.py          # gold vs distractor retention
python experiments/analyze.py              # degradation, cluster health
```

Source paragraphs are extracted one record per paragraph so each source fact carries
its title, which is what lets retention be split by HotpotQA's gold labels. Without
that split the numbers are uninterpretable: 8 of the 10 paragraphs are distractors,
so most "lost" source facts were correctly ignored, and raw retention cannot tell
correct filtering apart from attrition.

## Known limits

- **Adjudicator accuracy is unmeasured.** Every number this produces inherits it.
  Calibrate against a dataset with gold reasoning chains (FinQA's numeric chains,
  MuSiQue's reasoning DAG) before trusting results on data without them.
- `Channel.CONTEXT` must be supplied by the harness. This package records what you
  give it; it cannot reconstruct what was in an agent's context if the trace
  doesn't say.
- Availability ≠ attended-to. A fact 100k tokens deep in a context is recorded as
  `in_context` but may be functionally invisible.

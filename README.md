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

**A quantifier over a set is an atomicity defect, not a reference one.** `"Both were
American"` looks like an unresolved referent, and the first fix treated it that way —
resolve `"Both"` into the named subjects. That is wrong: the sentence asserts *one thing
per member*, and those per-member facts are normally already extracted from the same
text. Resolving it keeps a duplicate that can never be matched or checked on its own;
distributing it removes one. `"both"`, `"all"`, `"each"`, `"neither"`, `"they"` are now
handled under atomicity — emit one fact per member, and never the quantified sentence
alongside them.

The same pass added a rule for claims about the *source* rather than the world
(`"Both entries explicitly state their nationality as American"`, `"The documents
describe Animorphs as…"`): emit the underlying world fact if there is one, nothing
otherwise.

**Atomicity needs worked examples, not a rule.** The prompt said "split conjunctions
into separate facts" from the start and the model ignored it: `"Scott Derrickson is an
American director, screenwriter and producer"` came back as one fact. That does not stay
contained — a later restatement of one conjunct (`"Scott Derrickson is American"`) then
looks like a *weakened* version of the whole list, and the matcher records a degradation
event that never happened. Adding four worked splits plus the cases that must stay whole
(`"black and white film"`, `"Bosnia and Herzegovina"`) took facts carrying an unsplit
conjunction from 8.8% to 3.6%, and entailment edges touching one from 14.3% to 0.0%.

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

**Choosing a model matters less than it looks.** `factflow bench` scores a model on a
gold probe set built from defects that have actually cost us numbers (15 extraction
probes, 21 relation probes). Across 17 models from OpenAI and the OpenCode Go gateway,
on identical prompts, at published models.dev rates:

| model | provider | extract | relations | false-EQ | reliab. | $/bench |
|---|---|---|---|---|---|---|
| hy3 | Go | 96.6% | **100%** | 0 | 100% | **$0.0026** |
| glm-5.3-flash | Go | 96.6% | **100%** | 0 | 100% | $0.0042 |
| gpt-5.4-mini | OpenAI | **100%** | **100%** | 0 | 100% | $0.0277 |
| deepseek-v4-pro | Go | **100%** | **100%** | 0 | 95% | $0.0461 |
| gpt-4.1 | OpenAI | **100%** | **100%** | 0 | 100% | $0.0765 |
| gpt-5.5 | OpenAI | 100% | 100% | 0 | **45%** | $0.1364 |
| gpt-4.1-nano | OpenAI | 94.8% | **76.2%** | **2** | 100% | $0.0039 |

Price does not predict accuracy. `hy3` matches `gpt-4.1` on relations at 1/29th the cost;
the two most expensive runs in the full set beat nothing. Only `gpt-4.1-nano` is
disqualified on quality — it is the sole model emitting false `EQUIVALENT` edges, the one
error that compounds, since a single false equivalence merges two whole clusters.

**Reliability now separates models more than accuracy does.** `minimax-m3` scores 100/100
— on the 30% of calls that returned. `gpt-5.5` failed 55%. A model that is perfect on the
calls it answers and drops half of them is unusable in a pipeline making hundreds of calls
per question, so bench scores quality only over calls that returned and reports
reliability separately. An earlier version conflated the two and made three healthy models
look weak.

**The probe set has saturated.** Eleven of 17 score 100% on extraction, eight on
relations. That answers the original question — for current models this is not where the
risk is — and means the next version needs harder cases.

**Costs are fetched, not typed.** Hand-entered rates were 2–4× low on the whole gpt-5
family, which flattered exactly the models that turned out to be poor value. Token counts
were always measured; only the rate table was guesswork, and it now comes from models.dev.

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

**Threshold trades cost against relation type, not against recall flatly.** Measured
over four real stores:

| setting | EQUIV | ENTAIL | CONTRA | pairs |
|---|---|---|---|---|
| .45 / 20 | 99% | 97% | 100% | 3164 |
| **.50 / 12** (default) | **99%** | **90%** | **100%** | **2214** |
| .55 / 12 | 98% | 84% | 100% | 1793 |
| .65 / 10 | 97% | 64% | 100% | 1273 |
| .75 / 10 | 95% | 64% | 100% | 1103 |

Contradictions survive any threshold — a negation differs by one token. Equivalence
degrades gently. **Entailment collapses**, for exactly the reason containment exists: a
weakened restatement shares few tokens with its source. If you only need clustering
(retention, selectivity), `.65` costs 40% as much. If you need degradation analysis,
don't go above `.50`.

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

`experiments/` runs multi-agent frameworks over a QA benchmark and traces the facts
through them.

```bash
python experiments/run_qa.py --dataset medqa    -n 40          # topology x role sweep
python experiments/run_qa.py --dataset mmlu-pro --screen 160 -n 40   # hard subset
python experiments/run_hotpot.py -n 6                          # HotpotQA + fact tracing
python experiments/selectivity.py                              # gold vs distractor retention
python experiments/bench_models.py --provider opencode         # score a model set
```

`frameworks.py` implements three topologies — `full` (everyone reads everyone, the
Du et al. debate default), `chain` (A→B→C, each reads only its predecessor), and `star`
(spokes read the hub, the hub reads all) — crossed with role assignment (generalists, or
named specialists in the MedAgents style).

### Benchmarks saturate, which is the main experimental finding

| dataset | single agent | full debate | verdict |
|---|---|---|---|
| HotpotQA (distractor) | — | 100%, unanimous | retrieval, not collaboration |
| MedQA-USMLE 4-opt | **92%** (37/40) | 92% full · 95% chain · 92% star · 92% specialists | saturated |
| MMLU-Pro (phys/chem/eng) | **87%** (26/30) | 90%, 29/30 unanimous | saturated |

A committee cannot add much to a single agent already at 87–92%, and the agents mostly
just agree — unanimity runs 29/30 to 39/40. Topology and role assignment change nothing
measurable at that ceiling.

**Ten questions is not a sample.** MedQA at n=10 showed 80%→90% and looked like clear
evidence that collaboration helps. At n=40 it was 92%→92%. The entire apparent effect was
one question of noise, and it would have been reported as a finding.

### Screening for a hard subset

`--screen N` runs the single agent over N questions first, keeps the ones it fails, and
adds a matched control sample of ones it passes. This isolates the questions where
collaboration could matter at all, and the control catches collaboration *breaking*
answers a single agent already had right — which a hard-cases-only sample would hide.

On MMLU-Pro the single agent failed 27 of 160. Accuracy on that subset is low by
construction; what matters is how many failures get recovered and how many controls get
broken.

## Known limits

- **Adjudicator accuracy is unmeasured.** Every number this produces inherits it.
  Calibrate against a dataset with gold reasoning chains (FinQA's numeric chains,
  MuSiQue's reasoning DAG) before trusting results on data without them.
- `Channel.CONTEXT` must be supplied by the harness. This package records what you
  give it; it cannot reconstruct what was in an agent's context if the trace
  doesn't say.
- Availability ≠ attended-to. A fact 100k tokens deep in a context is recorded as
  `in_context` but may be functionally invisible.

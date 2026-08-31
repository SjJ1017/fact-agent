# Token-Aligned Fact Lifecycle: Persona Changes Churn More Than Uptake

## Conclusion

On the current 12 Perspectrum claims with `deepseek-v4-flash` and a shared,
all-to-all dossier, epistemic lenses create substantially more matched facts per
visible output token than neutral prompts, but those facts turn over faster and
are less often observed as uptake from a delivered peer message. Explicit
stance prompts mainly spend more output tokens; they do not reliably increase
the final fact portfolio per token and also reduce observed uptake. Thus the
available evidence supports **persona changes fact demography**, not yet that
it improves evidence coverage or information integration.

## Data

- Raw debates: `experiments/perspectrum_pilot_full/*deepseek*.debate.json`.
- Matched stores: the corresponding user-maintained `.v2.json` files.
- 36 runs: 12 claims paired across `neutral`, `lenses`, and `stance`; all use
  the `full` topology and three rounds.
- The local analysis copy and JSON were written under `/tmp`:
  `/tmp/factflow-token-clock-deepseek/` and
  `/tmp/factflow-token-lifecycle-deepseek.json`. They are not git artefacts.

Each condition has 11 runs with 9/9 factful output slots and one with 8/9.
Nine claims are complete in all three conditions. Quote-based location succeeds
for 1,242/1,486 neutral mentions (83.6%), 1,466/1,685 lens mentions (87.0%),
and 1,337/1,600 stance mentions (83.6%). Failed quote locations are retained at
the end of their turn and flagged rather than dropped.

## How To Reproduce

No model/API calls are needed. First construct independent annotated copies:

```bash
cd /Users/shenjiajun/Desktop/EPFL/AXA/project/factflow
PYTHONPATH=src .venv/bin/python experiments/token_clock.py \
  experiments/perspectrum_pilot_full \
  --glob '*deepseek*.v2.json' \
  --output-dir /tmp/factflow-token-clock-deepseek

PYTHONPATH=src .venv/bin/python experiments/analyze_token_lifecycle.py \
  /tmp/factflow-token-clock-deepseek \
  --debate-dir experiments/perspectrum_pilot_full \
  --out-json findings/data/perspectrum-deepseek-token-lifecycle.json
```

The clock uses the locally cached `BAAI/bge-base-en-v1.5` tokenizer over visible
text. `cumulative_visible_output_tokens` excludes prompt repetition; the
secondary total includes visible prompts. Neither is provider billing and both
exclude hidden reasoning. Turns are ordered `round -> agent` for accounting
only; agents within a round were generated concurrently.

## Results

Condition means over 12 claims:

| Persona | Output tokens | Unique output facts | Facts / 1k output tokens | Final-round facts | Final facts / 1k | Observed uptake rate | R1 facts surviving to R3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Neutral | 1,348 | 72.4 | 53.7 | 32.3 | 24.0 | 24.1% | 22.5% |
| Lenses | 1,415 | 97.7 | 69.7 | 39.9 | 28.7 | 18.0% | 17.8% |
| Stance | 1,569 | 93.4 | 59.6 | 35.6 | 22.6 | 17.2% | 16.5% |

At a matched visible-output budget, lenses surface facts fastest: by 500 tokens
they have 39.8 distinct facts on average, versus 32.8 for neutral and 34.8 for
stance. At 1,000 tokens the corresponding values are 74.3, 56.0, and 66.7.
All 12 runs per condition reach these two budgets; the 1,500-token comparison
is too incomplete to interpret.

Paired differences against the neutral run on the same claim, with 20,000
claim-bootstrap 95% intervals:

| Metric | Lenses minus neutral | Stance minus neutral |
|---|---:|---:|
| Unique output facts | +25.3 [12.5, 37.4] | +21.0 [11.3, 30.3] |
| Facts / 1k output tokens | +16.0 [7.0, 24.1] | +5.9 [-0.7, 11.9] |
| Final-round facts | +7.7 [2.8, 12.4] | +3.3 [-2.9, 9.2] |
| Final facts / 1k output tokens | +4.7 [1.2, 8.5] | -1.4 [-5.9, 2.8] |
| Round-2 births | +12.4 [5.7, 19.9] | +11.4 [7.8, 15.5] |
| Round-3 deaths | +10.6 [4.8, 17.2] | +10.3 [5.8, 14.5] |
| R1 to R3 survival | -4.7 pp [-12.9, 4.4] | -6.0 pp [-11.5, -0.6] |
| Observed uptake rate | -6.1 pp [-11.5, -0.2] | -6.9 pp [-13.0, -1.4] |
| Mean fact span in visible output tokens | -45.3 [-77.7, -12.4] | -27.8 [-68.2, 16.6] |

`Observed uptake` means a fact newly stated by agent B after it occurred in a
peer turn that `delivery` shows was supplied to B. It is a trace fact, not a
claim that B learned the fact from A: all agents also received the same dossier.

The complete-slot sensitivity analysis (nine paired claims) preserves the
direction: lenses add 20.5 facts/1k output tokens and lower uptake by 6.3 pp;
stance changes final facts/1k by -0.9 and lowers uptake by 7.8 pp.

## Interpretation

`Lenses` is not simply verbose: it adds 16.0 unique canonical facts per 1k
visible output tokens and 4.7 final facts per 1k. But it also has substantially
more late births and deaths, shorter fact spans, and lower uptake. The current
best reading is **productive exploration plus aggressive pruning**, not better
integration.

`Stance` creates more facts only by also using 220 additional visible output
tokens per debate on average. Its facts-per-token and final-facts-per-token
changes are not reliable, while its round-one survival and uptake decline. In
this setup, explicit advocacy appears to prolong/reframe discussion more than
it creates an efficient shared fact pool.

## What This Cannot Establish

- There is no post-hoc `SUPPORT/UNDERMINE/NEUTRAL` label for output facts yet,
  so this does **not** report stance entropy, polarized balance, or Gini.
- Gold perspectives have not been added and independently matched, so final
  fact count is not grounded coverage or factual quality.
- Every run is all-to-all with a common dossier. Uptake is observational; the
  experiment does not test private-information transfer or topology effects.
- n=12 claims is a pilot. Bootstrap intervals quantify variation across these
  claims, not generalization to all debates.

## Next Step

Run the additive gold-perspective pass on copies of the v2 stores, then measure
support/undermine coverage, balance, and survival by stance on the same token
clock. In parallel, run the already implemented ring/star conditions with a
strictly recorded delivery graph. Only the sparse-context study can answer
whether persona/topology improves genuine information transport.

# Persona, Topology, and Fact Flow: Design Memo

## Core question

How do persona configuration and communication topology change the creation,
distribution, and selection of atomic facts in multi-agent debate; when do
those changes improve the grounded comprehensiveness of the final answer rather
than merely consume more discussion budget?

The three research questions are deliberately separated:

1. **Explore.** Do persona conditions change the early fact portfolio and
   support/undermine coverage?
2. **Integrate.** Do facts become visible to, and subsequently appear in, other
   agents' outputs under each topology?
3. **Select.** Does an early rich or balanced fact portfolio predict better
   grounded final coverage at comparable token cost?

## Experimental factors

### Persona family

| Condition | Prompt intervention | Interpretation |
|---|---|---|
| General | All agents receive a matched general instruction to consider the case comprehensively. | Baseline. |
| Critical protocol | A fixed share of agents must identify counterexamples, unsupported steps, and omitted assumptions. | Tests critique as an interaction rule, without asserting different expertise. |
| Stance-diverse | Agents are assigned support, undermine, and adjudication stances. | Tests explicit disagreement. |
| Epistemic-lens-diverse | Agents focus on causal evidence, implementation/trade-offs, or scope/uncertainty. | Tests complementary analytic frames. |
| Identity/background | Agents are assigned social, educational, or cultural backgrounds. | Exploratory only: it can alter style or stereotypes without defining a verifiable information difference. |

The first study should use `General`, `Critical protocol`, and one of
`Stance-diverse` or `Epistemic-lens-diverse`, with prompts matched in length and
format. Professional titles should not be the main manipulation: an
epistemic-lens label gives a clearer prediction about the information type that
should be surfaced.

### Communication topology

Start with all-to-all, star, and ring/chain while holding the number of agents,
rounds, model, source dossier, and team token budget fixed. Two-party debate is
a separate interaction regime because it changes agent count and adversarial
structure as well as the graph. Selective graphs are a second-stage treatment;
their routing criterion must be pre-registered (role, novelty, confidence, or
past contribution).

## Fact-flow measurements

All measures operate on post-hoc extracted and matched canonical facts. Debate
prompts contain neither fact ids nor extraction instructions.

### Portfolio and stance

- `N_t`: unique canonical facts expressed in round `t`.
- `novel_t`, `live_t`, `born_t`, `died_t`, and `churn_t = born_t + died_t`.
- Stance entropy, over unique facts relative to the claim:
  \(H=-\sum_s p_s\log p_s\), reported normalized or as \(\exp(H)\).
- Polarized balance, excluding neutral facts:
  \(B_{\\pm}=1-|n_+-n_-|/(n_++n_-)\).
- Dual-sided coverage: whether both support and undermine facts occur.

Gini is an optional robustness measure of concentration. It is not the primary
balance statistic: with only support, undermine, and neutral buckets,
`B_{\\pm}` answers the substantive question more directly.

### Transport

An adoption event \(f:A_t\\to B_{t+1}\) requires: (i) `A` expressed fact `f`,
(ii) that turn was actually delivered to `B`, (iii) `B` had not previously
expressed `f`, and (iv) `B` then expresses it in the next pre-specified window.
Report novel adoption rate, fact reach, origin diversity among final facts, and
grounded adoption rate. In a shared-dossier all-to-all setting this is observed
uptake, not causal proof that `A` taught `B`; causal transfer requires sparse
visibility or independently retrieved evidence.

### Lifecycle and efficiency

For every output fact mention, store its estimated position on a trace clock:

- `cumulative_visible_output_tokens`: generated discussion tokens before the
  supporting quote plus its position in the current turn.
- `cumulative_visible_total_tokens`: the same clock including visible prompt
  tokens.

This is not provider billing or hidden-reasoning usage. Existing traces lack
per-call API usage, so they are annotated with one fixed local tokenizer and a
deterministic `round -> agent` order. Same-round turns were generated in
parallel; their within-round order is accounting only.

Use token-aligned survival curves by stance, grounding, source agent, and
adoption status. Report a half-life and reintroduction rate alongside the
number of live facts: a stable count can conceal complete turnover.

## Outcome and interpretation

Evaluate final answers using available human key points/perspectives/evidence:

- total and dual-sided gold coverage;
- final grounded precision;
- balanced grounded coverage; and
- marginal grounded coverage per 1,000 generated tokens.

Plot the Pareto frontier of final grounded balanced coverage against team token
cost, with early stance diversity and adoption as diagnostic dimensions. A
condition that raises fact count or entropy without improving grounded coverage
or efficiency is rich but unproductive. A condition that improves coverage at
matched cost is useful diversity.

Randomizing persona and topology supports causal claims about their effect on
fact-flow profiles. The relationship between a profile and final quality is
initially associational; causal claims about the mechanism require a later
controlled message-ablation or routing intervention.

## Immediate validation

Use the existing Perspectrum traces for stance-aware exploratory analysis,
then validate the same measures on a newer long-context human-debate resource
such as ArgCMV. The first topology study should avoid a full factorial sweep:
three persona families by three graphs, repeated across claims, is sufficient
to establish whether the effect sizes are large enough to justify expansion.

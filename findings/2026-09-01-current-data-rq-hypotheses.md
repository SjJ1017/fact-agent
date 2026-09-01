# What the Current Data Already Suggests

*2026-09-01; canonical artifacts and paths are listed in
`2026-09-01-shared-experiment-status.md`.*

## Main Reading

Persona diversity is not one treatment. The current full-topology data separate
three effects that should be evaluated independently:

1. **Exploration:** how many distinct canonical facts enter the discussion.
2. **Composition:** how balanced the support/undermine/neutral portfolio is.
3. **Integration:** how often a novel agent output reuses a fact from an
   actually delivered peer turn.

The three existing panels occupy different regions of this space: epistemic
lenses increase exploration and churn; explicit stances increase stance
composition diversity; neutral panels show the highest observed peer uptake.

## Strong, RQ-Relevant Patterns

### 1. Epistemic lenses produce fact richness, not stance diversity

Relative to neutral, lenses add about 25 output facts in both models:

- DeepSeek: +25.3 [12.5, 37.4], and +16.0 facts per 1k output tokens
  [7.0, 24.1].
- GLM: +25.3 [11.5, 38.2], and +9.1 facts per 1k output tokens
  [4.8, 13.0].

Yet their stance entropy at 1,000 tokens is indistinguishable from neutral in
both models. Therefore `#facts` should be called **richness**, not stance or
semantic diversity. Persona prompts determine which dimension of diversity is
changed.

### 2. Explicit stances create balanced but more siloed portfolios

Stance personas increase stance entropy/balance but reduce observed peer uptake:

- DeepSeek final balance: +0.124 [0.029, 0.243]; final entropy: +0.069
  [0.005, 0.140]. Uptake rate: -6.9 percentage points
  [-13.0, -1.4].
- GLM entropy at 1,000 tokens: +0.072 [0.023, 0.127]. Final balance has the
  same positive direction but is not reliable at n=11. Uptake rate: -6.7
  points [-11.5, -1.9].

Thus balanced evidence can arise without cross-agent integration. A useful
working term is **segregated diversity**: different roles preserve different
sides while rarely adopting one another's propositions.

### 3. Lenses show the clearest candidate for wasteful diversity

Lenses increase production but reduce integration in both models:

- DeepSeek uptake rate: -6.1 points [-11.5, -0.2].
- GLM uptake rate: -12.2 points [-16.5, -7.6].

The downstream conversion is model-dependent. DeepSeek lenses retain +7.7
final facts [2.8, 12.4] and +4.7 final facts per 1k tokens [1.2, 8.5]. GLM
lenses retain only +2.5 final facts [-6.6, 12.5] and approximately zero extra
final facts per 1k tokens. The same persona can therefore produce productive
exploration in one model and mostly churn in another.

GLM lenses are the strongest concrete garbage-diversity profile: +25 facts,
lower uptake, 10/11 claims with lower round-one-to-final survival, and a mean
fact span 125 tokens shorter than neutral. Their final stance entropy is also
lower by 0.116 [-0.237, -0.007] despite no entropy difference at 1,000 tokens.

### 4. Early stance composition predicts final stance composition

The claim-level Spearman correlation between balance at 1,000 tokens and final
balance is 0.82 for DeepSeek and 0.73 for GLM. It remains positive within most
persona panels (DeepSeek: 0.74--0.84; GLM lenses: 0.89, stance: 0.79).

This suggests an early portfolio probe may predict whether the final argument
will remain dual-sided. It is not causal evidence: the final set is selected
from the earlier set, so mechanical dependence is expected. Still, it supports
an early-stopping or routing signal worth validating out of sample.

### 5. Stance balance and information uptake are empirically orthogonal

Across runs, entropy at 1,000 tokens has essentially no rank correlation with
uptake (DeepSeek rho=-0.08; GLM rho=-0.07). High stance diversity does not imply
that agents absorbed information from peers. This justifies keeping Explore
and Integrate as separate RQs rather than combining them into one diversity
score.

### 6. Most full-topology output is additive, not observably transmissive

Even though every agent receives every peer's previous turn, aggregate uptake
rates are at most 24% (DeepSeek neutral) and 17% (GLM neutral), and are lower
under both diverse-persona treatments. Under the present matching definition,
most novel output facts are independently surfaced or derived from the common
dossier, rather than copied from a delivered peer turn.

This is observed discourse reuse, not causal learning. Matching may also miss
qualified paraphrases, so the safe claim is that exact canonical-fact uptake is
a minority behavior.

## Useful New Hypotheses

1. **Integration-capacity interaction:** persona-induced diversity improves
   final quality only when topology provides a synthesizer or other integration
   mechanism. Without it, fact births rise faster than uptake and retention.
2. **Two kinds of waste:** lenses cause churn waste (many births and deaths),
   while stance personas cause silo waste (balanced portfolios with low uptake).
3. **Model x persona interaction:** framework value depends on the model's
   ability to convert exploration into final retained facts, not just on the
   prompt or graph.
4. **Balance retention, not initial balance:** an effective integrator may be
   identified by how little polarized balance collapses between an early budget
   and the final round.
5. **Communication-aware efficiency:** GLM spends substantially more output
   tokens before peer uptake becomes possible. Cross-model comparisons should
   use tokens since first peer exposure or exposure opportunities, not only
   absolute cumulative tokens.
6. **Adopted-and-retained yield:** distinguish peer facts that are merely
   repeated from peer facts that survive into the final portfolio.
7. **Stance-conditioned transport:** test whether agents preferentially adopt
   same-side facts and reject or transform opposite-side facts. This can expose
   echo-chamber dynamics hidden by aggregate uptake.
8. **Origin concentration:** measure whether final facts come mainly from one
   agent/hub even when their stance distribution is balanced.

## Analyses Possible Without New Debate Runs

- Decompose final facts into self-originated, peer-adopted, and independently
  convergent facts.
- Plot uptake and final survival separately for support, undermine, and neutral
  facts.
- Compute sender-receiver matrices and final-origin entropy by persona.
- Measure adopted-and-retained yield and time-to-uptake.
- Compare balance collapse and mortality claim by claim, with paired intervals.
- Match final facts to existing Perspectrum gold perspectives/evidence to test
  whether richness and balance improve grounded coverage rather than only
  internal composition.

## Important Design Warning for Star and Chain

The completed star/chain traces are raw debates and cannot yet support fact-flow
conclusions. In the current implementation A is both a graph position and a
specific role: advocate in the stance panel and causal analyst in lenses. In
chain, roles are also ordered A -> B -> C. Any topology result can therefore be
confounded by role-position alignment. Rotate roles across graph positions or
model role position explicitly before claiming a pure topology effect.

## Metrics That Currently Fail

- Binary dual-sided coverage saturates near 100% and has little discriminative
  value; use entropy and polarized balance.
- Marginal percentile bands across claims are not uncertainty on paired effects.
- Fact count alone cannot distinguish productive exploration from churn.
- Full-context uptake cannot establish that one agent taught another a fact it
  otherwise could not know.

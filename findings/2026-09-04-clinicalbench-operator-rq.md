# From Fact Survival to Agent Operators

## DelibTrace Boundary

The current questions, stated as whether roles preserve diversity and whether
preserved diversity improves outcomes, overlap too strongly with DelibTrace.
DelibTrace already asks whether factual and stance diversity survives
multi-round discussion, varies persona prompts and communication topology, and
tests the downstream judgment impact of factual attrition.

The distinct object should not be another survival curve. It should be the
realized, fact-conditioned computation performed by each agent:

> Given the source facts, self-history, and peer facts visible to an agent,
> which facts does it activate, retain, drop, relay, qualify, combine, or
> originate?

DelibTrace's graph describes which agents can communicate. The proposed graph
describes which facts actually travel through those permitted channels.

## Revised Questions

**RQ1: Role-induced operator specialization.** Holding the model and available
information fixed, do role prompts induce stable and distinguishable
fact-transformation profiles across rounds and cases?

**RQ2: Flow-mediated utility.** When do complementary operator profiles create
evidence-supported integration paths that improve the final result, rather
than merely producing more or more-diverse facts?

Information allocation is a factorial intervention, not a competing topic:
full versus split context tests whether roles specialize attention or merely
reflect different inputs. A split-role mismatch condition, created by
permuting the role prompts while keeping source partitions fixed, can separate
role-driven behavior from source-driven behavior.

## Evidence in the Current Pilot

These are descriptive means over two task prompts on one patient.

Using distributions over source-document type plus an agent-new bucket, the
mean pairwise Jensen-Shannon divergence between agent profiles changes as
follows:

| Condition | Round 1 | Round 2 |
|---|---:|---:|
| full-generic | 0.022 | 0.012 |
| full-specialist | 0.111 | 0.134 |
| split-generic | 0.429 | 0.078 |
| split-specialist | 0.440 | 0.085 |

With shared input, specialist roles create and maintain a distinguishable
selection profile while generic agents become more similar. With split input,
both role settings begin highly different and converge sharply after exchange;
roles do not preserve endpoint content diversity there.

The realized source-exclusive flow nevertheless changes direction. Mean
fact-edge counts per run are:

- Split-generic: A->B 6.0, A->C 5.5, B->A 1.5, C->A 1.5.
- Split-specialist: B->A 4.5, C->A 4.0, A->B 1.0, A->C 2.5.

Here A holds history/examination and acts as the bedside/differential
clinician; B holds imaging/endoscopy; C holds laboratory evidence. Generic
agents mainly broadcast A's history facts outward. Aligned specialist roles
instead route B/C specialist evidence into A. Thus output content can
homogenize while functional heterogeneity remains visible in who acts as
source, filter, relay, or integrator.

This directional reversal is a promising graph hypothesis, not a result:
there are only two repeated prompts over one patient and edge counts inherit
matching error.

## Operator Profile

For each agent-turn, condition on the facts it could see and estimate:

- Activation: available source facts selected into output, by clinical domain.
- Persistence/drop: own prior facts retained or removed.
- Uptake: previously unseen peer facts selected, with source-exclusive edges
  treated as the strongest evidence.
- Origination: unmatched output facts from an agent-specific virtual source.
- Transformation: qualify, contradict, or synthesize relations between input
  and output facts.

The first four can be approximated with the current atomic matching. The last
requires relation labeling. Treating every unmatched output as a genuinely new
fact would conflate valid inference, hallucination, granularity changes, and
matcher false negatives.

## Minimal Next Experiment

Run independent ClinicalBench cases under five paired conditions:

1. Full context, generic agents.
2. Full context, specialist roles.
3. Split context, generic agents.
4. Split context, aligned specialist roles.
5. Split context, role prompts permuted across source partitions.

Optionally manipulate role persistence directly: retain roles in round 2,
remove them in round 2, or swap them in round 2. This tests whether operator
profiles follow the role instruction and whether role-induced differences
survive information exchange.

Primary mechanism tests should be cross-case role-profile separability,
within-role stability, cross-role transfer matrices, integration-path yield,
and supported-final-fact yield per token. Outcome analysis should test:

role/information intervention -> operator profile -> realized fact-flow graph
-> supported final facts -> official task score

Raw fact count, entropy, and retention remain diagnostics, not quality
objectives.

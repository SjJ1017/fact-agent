# Shared Experiment Status

**Read this file before launching an experiment, re-labelling facts, or
reporting a result.** Multiple agents are operating this workspace. Do not infer
the state from an earlier chat message: inspect the paths and counts below
first.

## Canonical Artifacts (checked 2026-09-02)

| Artifact | Path | Current state |
|---|---|---|
| DeepSeek full debates | `experiments/perspectrum_pilot_full/*deepseek-v4-flash-full-*.debate.json` | 36 = 12 claims x 3 personas |
| GLM full debates | `experiments/perspectrum_pilot_full/*glm-5.3-flash-full-*.debate.json` | 33 = 11 claims x 3 personas; claim 233 is absent in all three panels |
| DeepSeek star/chain debates | `experiments/perspectrum_pilot_star_chain/*.debate.json` | 72 = 12 claims x 3 personas x 2 graphs |
| DeepSeek star/chain matched stores | `experiments/perspectrum_pilot_star_chain/*.v2.json` | 72/72 complete |
| Canonical stance labels | `experiments/labels/stance/*.json` | 141 = 108 DeepSeek + 33 GLM |
| Canonical token clocks | `experiments/labels/token_clock/*.json` | 129 = 96 DeepSeek + 33 GLM; chain is 24/36 |

The repository sidecars under `experiments/labels/` supersede temporary copies.
Do **not** launch `label_fact_stance.py` merely because a `/tmp` directory is
absent. The 12 missing token clocks are all chain runs; the current flow-profile,
effective-structure, and RQ-extension analyses do not depend on those token clocks.

## Current Derived Results

All of these are post-hoc. Facts, labels, token clocks, and plots never enter a
debate agent prompt.

| Model | Lifecycle JSON | Stance JSON | Dashboard PNG | Paired sample |
|---|---|---|---|---:|
| DeepSeek-v4-flash | `/tmp/factflow-token-lifecycle-deepseek.json` | `/tmp/factflow-stance-diversity-deepseek-all.json` | `/tmp/factflow-dashboard-deepseek.png` | 12 claims |
| GLM-5.3-flash | `/tmp/factflow-token-lifecycle-glm.json` | `/tmp/factflow-stance-diversity-glm.json` | `/tmp/factflow-dashboard-glm.png` | 11 claims |

Observed-peer-uptake trajectories are at `/tmp/factflow-uptake-deepseek.png`
and `/tmp/factflow-uptake-glm.png`. Their value at budget `t` is the per-run
mean share of novel output facts expressed by `t` that had appeared in a peer
turn actually delivered to that agent. It is not the probability that a
delivered fact was adopted, and in the shared-dossier full topology it is not
causal proof of learning.

The earlier DeepSeek stance result in
`/tmp/factflow-stance-labeled-deepseek-judge/` is a different label snapshot.
Never mix its reported values with plots made from `factflow-stance-all`.
For the canonical snapshot, DeepSeek final polarized balance is 0.554 (stance)
versus 0.429 (neutral), paired difference +0.124, bootstrap 95% interval
[0.029, 0.243].

## Topology Semantics

The 72 sparse-topology traces have been extracted, matched, and stance-labelled.
Their canonical matched stores are the `.v2.json` files above.

- `star`: A is the communication hub; A receives B and C, while B and C receive
  A. It is **not** an instructor/synthesizer intervention.
- `chain`: A receives no peer output, B receives A, C receives B.

Do not rerun `run_perspectrum.py`, `retrace.py`, or stance labelling for
`perspectrum_pilot_star_chain`: all 72 debate/store/stance artifacts exist.
Only token clocks are incomplete (24/36 chain), and any completion must first
check which 12 execution ids are missing rather than rerunning all 36.

## 2026-09-02 Analysis Update

- `experiments/analyze_effective_structure.py` now reads all 108 DeepSeek runs;
  `experiments/report/effective-structure.html` includes chain in its table,
  figures, statistical summaries, and limitations.
- `adopt_r1uniq` now uses all round-1-unique facts as its denominator. The old
  0.82/0.86 values were final-survivor-conditioned and must not be cited.
- `experiments/analyze_rq_extensions.py` writes
  `findings/data/rq-extensions.json` fully offline. It adds persona×topology
  difference-in-differences, a round-2 exposure→uptake→retention funnel, strict
  chain path counts, and condition-adjusted verdict coupling.
- New-only report: `findings/2026-09-02-rq-review-and-new-analyses.md`.
- No model calls were made. Existing debate conditions and label artifacts were
  not modified.

## Required Pre-Run Check

Before any model call or write-heavy run:

1. Inspect the target artifact directory with `find ... | wc -l` and check the
   relevant manifest or result JSON.
2. Check active processes with `ps ... | rg '<script name>'`.
3. Reuse canonical paths above. Write new speculative outputs to a clearly
   named, separate directory.
4. State in the progress update: input paths, output path, expected count, and
   whether the call sends content to a model.
5. Update this file when a stage completes, fails, is superseded, or becomes
   canonical.

## Code Added This Session

- `experiments/plot_factflow_dashboard.py`: five token-aligned panels
  (entropy, polarized balance, expressed, dead, survived), intentionally with
  no uncertainty shading.
- `experiments/plot_stance_diversity.py`: now uses mean-based y limits and no
  percentile shading.
- `experiments/plot_uptake_trajectory.py`: token-aligned observed-peer-uptake
  trajectory, using the trace delivery graph and no uncertainty shading.
- `experiments/run_perspectrum.py`: loads only `OPENCODE_API_KEY` from the
  project-local `.env` for non-interactive runs when it is otherwise absent.

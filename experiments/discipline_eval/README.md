# IDRBench topic / attribution labelling (2026-09-07)

Frozen cohort: 40 NLI stores, 11,201 fact IDs, 10,181 unique (task, full paired abstracts, exact canonical text) inputs. No extraction or matching reruns. Existing free-text `labels/provenance` labels are not mixed into this layer.

The two axes are **topic B/C/X/G/U** (relative paper contribution, shared/both, generic, unclear) and **paper B/C/X/P/U** (describes a paper, both/comparison, proposed design/panel judgement, unclear). These are neither a universal academic taxonomy nor verified source entailment. Same-discipline paper pairs are classified by contribution/method. X does not certify successful integration; P does not certify novelty.

## Cost-controlled configuration

- OpenCode **Go** endpoint, only `.env` `OPENCODE_API_KEY_2`; no fallback.
- `qwen3.8-flash`, `enable_thinking=false`, temperature 0, batch 64, max output 2048, concurrency 4.
- System hash `0a5669c0d0e186284db471db7486be524f34eb7e73e49f5cbbde0160bb38bbd8`.
- Four 64-fact development calls: zero reasoning output, approximately 5–8 seconds each. Topic 30/32, attribution 32/32 on deliberately selected **agent-labelled development probes**, not an independent human test set.
- GLM smoke ignored the requested thinking-disable setting, produced 8,366 reasoning characters, and exhausted its output cap without labels. GPT luna smoke returned HTTP 500. Neither used for production.
- All locally recorded tests and production: **$0.132341098 catalogue-equivalent known usage**; one failed call has no usage. This is a quota comparison based on fetched models.dev rates, not a Go cash invoice.
- Ledger `*-calls.jsonl` includes failures; `responses/` contains successful exact-request caches. Successful-call cost alone excludes failed billed usage and is not the full cost.

## Completion policy

Production initially requested 163 batches. Invalid paper enum G was rejected, never guessed as another class. Same frozen requests were retried; 160 batches passed, 9,989 inputs labelled. At the user's instruction, the remaining 3 batches / 192 inputs were left unlabelled. They map to 228 fact IDs (2.04% of all IDs), in cases 49, 55 and 95. All 40 traces stay in every analysis.

`export_discipline_partial.py` is an offline export of all frozen inputs. Missing inputs receive topic U, paper U and `status=unlabelled_invalid_batch`; model-produced U retains `status=labelled`. Nothing is silently dropped. Figures combine both under “未判定 / 未标注”, and sidecars preserve the distinction. Do not keep retrying these batches unless requested.

## Artifacts

- `experiments/labels/discipline_v1/<execution_id>.json`: lightweight fact-keyed sidecars with canonical text, frozen store SHA256, model and prompt hash.
- `manifest.json`: full source context, exact deduplication IDs and 40 source checksums.
- `production-summary.json`: valid coverage, missing batches, tokens and cost ledger totals.
- `findings/data/idrbench-discipline/`: offline distribution, interaction, graph and review data.
- `findings/2026-09-07-mentor-report.html`: standalone mentor report; all plots/data embedded.

Offline rebuild:

```sh
.venv/bin/python experiments/analyze_discipline_flow.py
.venv/bin/python experiments/summarize_discipline_flow.py
.venv/bin/python experiments/analyze_discipline_network.py
.venv/bin/python experiments/analyze_network_balance.py
.venv/bin/python experiments/analyze_perspectrum_network_report.py
.venv/bin/python experiments/build_fact_path_examples.py
.venv/bin/python experiments/build_mentor_report.py
```

Interactions use direct scored mention relations with actual visibility; cluster IDs only deduplicate and screen prior expression. Main temporal window is adjacent rounds. Topic effects are task-paired, not fact-level independent tests. Existing NLI threshold is a shared 5.28 in both directions; oracle-fitted evaluation remains an optimistic upper bound. Reported comparisons are exploratory, without multiple-comparison correction.

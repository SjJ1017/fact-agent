# Avalon five-player self-play: execution record

## Status

- **Current stage:** implementation and offline validation.
- **Model calls made:** none at the time this entry was created.
- **Planned corpus:** 10 complete five-player games using
  `deepseek-v4-flash` through OpenCode Go at temperature 0.95.
- **Pipeline boundary:** generate channel-aware traces, extract mentions, and
  atomize mentions. Do **not** run matching.

## Inputs and outputs

- Configuration: `experiments/datasets/avalon_5p.yaml`
- Generator: `experiments/avalon/run_avalon.py`
- Offline audit: `experiments/avalon/audit_traces.py`
- Extraction: `experiments/avalon/extract_atomize.py`
- Trace/output directory: `experiments/avalon_5p_deepseek_v4_flash/`
- Generation log: `experiments/avalon_5p_deepseek_v4_flash/generation.log`
- Extraction log: `experiments/avalon_5p_deepseek_v4_flash/extraction.log`

Raw traces distinguish public discussion/proposals, votes that are private
until the simultaneous reveal, secret mission actions, and public environment
outcomes. Every model call receives the full public history and that player's
own private action history. Role assignments are balanced across ten games:
each player is Merlin twice, Minion twice, Assassin twice, and Servant four
times.

## Prompt basis

The phase structure and prompts follow the official AvalonBench implementation:

- <https://github.com/jonathanmli/Avalon-LLM>
- <https://github.com/jonathanmli/Avalon-LLM/blob/main/src/server/tasks/avalon/prompts.py>
- <https://github.com/jonathanmli/Avalon-LLM/blob/main/src/server/tasks/avalon/agents/llm_with_discussion.py>

The local prompt fixes the upstream introduction's five-player Evil count typo
and adds explicit channel semantics required for fact-flow analysis. It requests
concise rationales rather than hidden chain-of-thought.

## Validation gate

Run two games first. Continue to ten only if both traces:

1. reach a legal terminal state in three to five quests;
2. contain no empty or unparsable actions after repair;
3. prevent Good players from submitting FAIL mission cards;
4. expose public speech/proposals to all five players while keeping private
   votes and mission actions private;
5. show non-duplicated, bounded public discussion rather than cache replay.

## Commands

```bash
set -a && . ./.env && set +a
.venv/bin/python experiments/avalon/run_avalon.py --games 2 --parallel-games 1 \
  2>&1 | tee experiments/avalon_5p_deepseek_v4_flash/generation.log
.venv/bin/python experiments/avalon/audit_traces.py \
  experiments/avalon_5p_deepseek_v4_flash --limit 2
```

After the gate passes, the resumable generator skips the first two completed
files and fills the corpus to ten. Extraction writes `.atomized.json` files
whose `facts`, `mention_to_fact`, and `relations` fields remain empty.

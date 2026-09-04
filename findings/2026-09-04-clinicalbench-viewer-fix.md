# ClinicalBench Trace Viewer Repair

The dedicated viewer now contains all 9 valid saved stores: the balanced
8-cell analysis set plus the valid department-routing store. It renders 1,199
located mentions and reports 53 unplaced mentions (95.8% placement).

Flow edges are precomputed from the saved delivery record instead of inferred
from DOM order:

- Source edges respect each agent's assigned `source_ids`.
- Same-agent recurrence is labeled persistence.
- Peer transmission requires that the fact is new to the receiving agent and
  was actually visible from a prior peer turn.
- Lexical evidence-label references are not treated as fact provenance.

Markdown bold and code delimiters are parsed before fact highlighting, so raw
`**` markers no longer leak when a matched fact splits a styled phrase.

Validation completed:

- All 9 stores rebuild into viewer records.
- The payload contains 849 typed edges.
- A synthetic split-information regression covers source ownership,
  persistence, valid uptake, and the prior false-positive peer edge.
- Python syntax and `git diff --check` pass.
- The generated viewer was inspected in-browser with all 9 options, styled
  text, fact selection, and a transmission edge rendering correctly.

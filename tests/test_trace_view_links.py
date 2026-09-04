import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from build_trace_view import build_links  # noqa: E402


def _span(fact_id: str) -> list[dict]:
    return [{"a": 0, "b": 4, "f": fact_id, "t": "fact",
             "fs": [{"f": fact_id, "t": "fact"}]}]


def test_links_respect_source_ownership_and_new_to_receiver_uptake():
    fact_id = "f1"
    spans = {slot: _span(fact_id) for slot in
             ("A|1", "B|1", "A|2", "B|2", "C|2")}
    debate = {
        "roles": {"A": "", "B": "", "C": ""},
        "transcript": {slot: "fact" for slot in
                       ("A|1", "B|1", "C|1", "A|2", "B|2", "C|2")},
        "evidence": [{"id": "doc-b"}],
        "delivery": {
            "A|1": {"source_ids": ["doc-a"]},
            "B|1": {"source_ids": ["doc-b"]},
            "C|1": {"source_ids": ["doc-c"]},
            "A|2": {"source_ids": ["doc-a"], "visible_peer_turns": ["B|1", "C|1"]},
            "B|2": {"source_ids": ["doc-b"], "visible_peer_turns": ["A|1", "C|1"]},
            "C|2": {"source_ids": ["doc-c"], "visible_peer_turns": ["A|1", "B|1"]},
        },
    }

    edges = build_links(debate, spans, {fact_id: ["doc-b"]})[fact_id]
    triples = {(edge["kind"], edge["from"], edge["to"]) for edge in edges}

    assert ("origin", "doc-b", "B|1") in triples
    assert ("origin", "doc-b", "A|1") not in triples
    assert ("persistence", "A|1", "A|2") in triples
    assert ("persistence", "B|1", "B|2") in triples
    assert ("transmission", "B|1", "A|2") not in triples
    assert ("transmission", "A|1", "C|2") in triples
    assert ("transmission", "B|1", "C|2") in triples

"""Synthetic drug pairs, because the real ones are memorised.

The real-drug case set failed for its purpose, and failed informatively. Given
only one dossier, the model scored 69% and named the *partner's* enzyme 75% of
the time — it had never been shown that dossier. On fluconazole + warfarin it
said so outright:

    "warfarin's CYP2C9 dependence is from standard pharmacologic knowledge,
     not the supplied text"

Fluconazole + warfarin is one of the most documented interactions in medicine.
The join is unnecessary when the conclusion is memorised, so a correct answer
says nothing about whether the evidence was combined — which is the one thing
the experiment exists to measure.

Invented drugs and invented enzymes fix that by construction. Prior knowledge is
zero, so a correct answer can only come from joining the two mechanism facts,
and `solo-half` becomes a true floor at chance rather than at 69%.

The generator holds everything else constant with the real set: one mechanism
fact per drug, plausible clinical facts, distractors to create filtering load,
and negatives that pair an inhibitor with a substrate of a *different* enzyme so
the surface pattern is identical.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from cases import DDICase, DrugFacts

# Invented enzyme family. Deliberately not CYP: a model that has memorised CYP
# substrate tables can still pattern-match on "CYP3A4 is common, so probably
# yes", and the point is to remove every route to the answer except the join.
ENZYMES = ["HTX-1A", "HTX-2C", "HTX-3F", "HTX-4B", "HTX-5D", "HTX-6K", "HTX-7M", "HTX-8P"]

PREFIX = ["Zal", "Mor", "Ven", "Kess", "Dral", "Tunn", "Orb", "Pel", "Rix", "Cadd",
          "Sev", "Yur", "Braxt", "Immo", "Nyl", "Quor"]
SUFFIX = ["otinib", "apine", "adol", "ustat", "izumab", "afil", "oxetine", "acaine",
          "ipramol", "ozide", "erol", "atide"]

CLINICAL = [
    "{d} is given once daily by mouth.",
    "{d} reaches peak plasma concentration about two hours after a dose.",
    "{d} is approved for use in adults only.",
    "{d} is supplied as 10 mg and 25 mg tablets.",
    "{d} is taken with food to reduce nausea.",
    "{d} has a plasma half-life of roughly fourteen hours.",
    "{d} is contraindicated in severe hepatic impairment.",
    "{d} is dosed by body weight in patients under 50 kg.",
]
DISTRACTORS = [
    "{d} is 92% bound to plasma proteins.",
    "{d} causes mild headache in about 8% of patients.",
    "{d} is stored below 25 degrees Celsius.",
    "{d} was first registered in 2019.",
    "{d} is excreted mainly in the faeces.",
    "{d} has no known effect on the QT interval.",
    "{d} is available as a generic formulation.",
    "{d} does not require renal dose adjustment.",
]

INHIBITOR = "{d} is a strong inhibitor of {e}."
INDUCER = "{d} is a potent inducer of {e}."
SUBSTRATE = "{d} is cleared almost entirely by {e}."
PRODRUG = "{d} requires {e} to form its active metabolite."


def _name(rng: random.Random, used: set[str]) -> str:
    while True:
        n = rng.choice(PREFIX) + rng.choice(SUFFIX)
        if n not in used:
            used.add(n)
            return n


def _drug(rng: random.Random, name: str, mechanism: str) -> DrugFacts:
    clinical = [t.format(d=name) for t in rng.sample(CLINICAL, 2)]
    distractors = [t.format(d=name) for t in rng.sample(DISTRACTORS, 2)]
    return DrugFacts(name, mechanism, clinical, distractors)


def generate(n_positive: int = 12, n_negative: int = 12, seed: int = 0) -> list[DDICase]:
    """Balanced synthetic cases. Positives share an enzyme; negatives do not."""
    rng = random.Random(seed)
    used: set[str] = set()
    cases: list[DDICase] = []

    for i in range(n_positive):
        enz = rng.choice(ENZYMES)
        perp, obj = _name(rng, used), _name(rng, used)
        modifier, direction = rng.choice([(INHIBITOR, "raised"), (INDUCER, "reduced")])
        target = rng.choice([SUBSTRATE, PRODRUG])
        a = _drug(rng, perp, modifier.format(d=perp, e=enz))
        b = _drug(rng, obj, target.format(d=obj, e=enz))
        consequence = (f"{direction} exposure to {obj}" if target is SUBSTRATE
                       else f"{'reduced' if modifier is INHIBITOR else 'raised'} activation of {obj}")
        cases.append(DDICase(f"syn-pos-{i:02d}", a, b, True, enz, consequence,
                             "both dossiers name the same enzyme"))

    for i in range(n_negative):
        e1, e2 = rng.sample(ENZYMES, 2)
        perp, obj = _name(rng, used), _name(rng, used)
        modifier = rng.choice([INHIBITOR, INDUCER])
        target = rng.choice([SUBSTRATE, PRODRUG])
        a = _drug(rng, perp, modifier.format(d=perp, e=e1))
        b = _drug(rng, obj, target.format(d=obj, e=e2))
        cases.append(DDICase(f"syn-neg-{i:02d}", a, b, False, "none shared",
                             "no shared metabolic route",
                             f"{e1} modifier paired with an {e2} substrate"))

    rng.shuffle(cases)
    return cases


def with_decoys(case: DDICase, n_decoys: int, seed: int = 0) -> tuple[str, str]:
    """Bury each agent's real dossier among decoy drugs.

    At one dossier per agent the join is trivial: there is exactly one candidate
    fact on each side and 100% is the expected score. The task only becomes a
    real search when the relevant pair has to be found among many, so this
    returns each agent's view as several dossiers, exactly one of which is the
    drug under question.

    Decoy enzymes are drawn to include the case's own enzyme, so a decoy can
    share an enzyme with the *other* agent's real drug. An agent that answers by
    scanning for any enzyme match will then join the wrong pair.
    """
    rng = random.Random(f"decoy:{case.case_id}:{seed}")
    used = {case.drug_a.name, case.drug_b.name}
    views = []
    for real, partner in ((case.drug_a, case.drug_b), (case.drug_b, case.drug_a)):
        partner_enz = ENZYMES[0]
        for e in ENZYMES:
            if e in partner.mechanism:
                partner_enz = e
                break
        decoys = []
        for i in range(n_decoys):
            nm = _name(rng, used)
            # Half the decoys use the partner's enzyme: a lure for enzyme-scanning.
            enz = partner_enz if i % 2 == 0 else rng.choice(ENZYMES)
            tmpl = rng.choice([INHIBITOR, INDUCER, SUBSTRATE, PRODRUG])
            decoys.append(_drug(rng, nm, tmpl.format(d=nm, e=enz)))
        block = [real] + decoys
        rng.shuffle(block)
        views.append("\n\n".join(d.dossier(shuffle_seed=seed) for d in block))
    return views[0], views[1]


def load_synthetic(only: str | None = None, **kw) -> list[DDICase]:
    cases = generate(**kw)
    if not only:
        return cases
    if only == "positive":
        return [c for c in cases if c.interacts]
    if only == "negative":
        return [c for c in cases if not c.interacts]
    return [c for c in cases if c.case_id == only]

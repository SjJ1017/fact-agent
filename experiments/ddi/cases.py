"""Drug-interaction cases with an explicit two-fact mechanism chain.

Why this task and not another QA set: every benchmark tried so far was saturated
(single agent 87-92%), so a committee had nothing to add and the agents simply
agreed. A CYP-mediated interaction is different in a way that matters here --
the answer follows from joining exactly TWO facts held in different places:

    A: "fluconazole inhibits CYP2C9"
    B: "warfarin is metabolised by CYP2C9"
    -> fluconazole raises warfarin exposure -> bleeding risk

Split those across agents and the join either happens or it does not. That is a
binary, observable event, which is what makes it possible to separate "answered
correctly by combining evidence" from "answered correctly from prior knowledge"
-- a distinction accuracy cannot make and the whole reason for tracing facts.

NEGATIVE CONTROLS are the load-bearing part. Each one pairs an inhibitor with a
substrate of a *different* enzyme, so the surface pattern ("one inhibits
something, one is metabolised by something") is identical to the positives.
Only an agent that actually checks enzyme identity can separate them; pattern
matching scores 50%.

These cases encode well-documented pharmacology and are written for benchmark
use. Verify against DrugBank or the current labels before drawing any clinical
conclusion from them, and treat the fact wording as fixtures rather than as a
reference source.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DrugFacts:
    """One drug's dossier as it would be handed to an agent.

    `mechanism` is the single join-critical fact. `clinical` and `distractor`
    facts are true and plausible but irrelevant to this pair - they create the
    filtering load that makes "which facts get discarded" measurable.
    """

    name: str
    mechanism: str
    clinical: list[str] = field(default_factory=list)
    distractors: list[str] = field(default_factory=list)

    def dossier(self, shuffle_seed: int | None = None) -> str:
        """The drug's facts as one block.

        The mechanism fact is not placed first: if it always led, an agent could
        do well by reading only the first bullet, and the filtering load this
        task is meant to create would disappear.
        """
        body = [self.mechanism] + self.clinical + self.distractors
        if shuffle_seed is not None:
            import random

            # Seed per drug, not per run: every dossier has the same number of
            # facts, so one shared seed puts the mechanism at the same index in
            # all of them and an agent could learn the position instead of the
            # content.
            random.Random(f"{self.name}:{shuffle_seed}").shuffle(body)
        return f"{self.name}\n" + "\n".join(f"- {b}" for b in body)

    def all_facts(self) -> list[str]:
        return [self.mechanism] + self.clinical + self.distractors


@dataclass
class DDICase:
    case_id: str
    drug_a: DrugFacts
    drug_b: DrugFacts
    interacts: bool
    enzyme: str
    consequence: str
    why: str

    @property
    def critical_pair(self) -> tuple[str, str]:
        """The two facts whose co-location constitutes the join."""
        return (self.drug_a.mechanism, self.drug_b.mechanism)


def _d(name, mech, clinical=(), distractors=()) -> DrugFacts:
    return DrugFacts(name, mech, list(clinical), list(distractors))


FLUCONAZOLE = _d(
    "Fluconazole", "Fluconazole is a moderate inhibitor of CYP2C9.",
    ["Fluconazole is used to treat candidiasis and cryptococcal meningitis.",
     "Fluconazole has high oral bioavailability, close to 90%."],
    ["Fluconazole is eliminated predominantly unchanged in the urine.",
     "Fluconazole can prolong the QT interval at high doses."])
WARFARIN = _d(
    "Warfarin", "Warfarin's more potent S-enantiomer is metabolised by CYP2C9.",
    ["Warfarin is a vitamin K antagonist used for anticoagulation.",
     "Warfarin therapy is monitored by INR."],
    ["Warfarin has a narrow therapeutic index.",
     "Warfarin is highly bound to plasma albumin."])
CLARITHROMYCIN = _d(
    "Clarithromycin", "Clarithromycin is a strong inhibitor of CYP3A4.",
    ["Clarithromycin is a macrolide antibiotic.",
     "Clarithromycin is used in Helicobacter pylori eradication regimens."],
    ["Clarithromycin is associated with taste disturbance.",
     "Clarithromycin achieves high intracellular concentrations."])
SIMVASTATIN = _d(
    "Simvastatin", "Simvastatin is extensively metabolised by CYP3A4.",
    ["Simvastatin is an HMG-CoA reductase inhibitor used to lower LDL cholesterol.",
     "Simvastatin is usually taken in the evening."],
    ["Simvastatin is a lactone prodrug hydrolysed to its active acid.",
     "Simvastatin carries a dose-dependent risk of myopathy."])
RIFAMPIN = _d(
    "Rifampin", "Rifampin is a potent inducer of CYP3A4.",
    ["Rifampin is a first-line antimycobacterial agent.",
     "Rifampin colours urine and tears orange."],
    ["Rifampin is also a P-glycoprotein inducer.",
     "Rifampin is given as part of combination therapy to limit resistance."])
ETHINYLESTRADIOL = _d(
    "Ethinylestradiol", "Ethinylestradiol is metabolised by CYP3A4.",
    ["Ethinylestradiol is the oestrogen component of combined oral contraceptives.",
     "Ethinylestradiol undergoes enterohepatic recirculation."],
    ["Ethinylestradiol increases hepatic synthesis of clotting factors.",
     "Ethinylestradiol is administered orally once daily in most regimens."])
FLUOXETINE = _d(
    "Fluoxetine", "Fluoxetine is a strong inhibitor of CYP2D6.",
    ["Fluoxetine is a selective serotonin reuptake inhibitor.",
     "Fluoxetine has an unusually long half-life among SSRIs."],
    ["Fluoxetine's active metabolite is norfluoxetine.",
     "Fluoxetine is approved for major depressive disorder and bulimia nervosa."])
TAMOXIFEN = _d(
    "Tamoxifen", "Tamoxifen requires CYP2D6 to form its active metabolite endoxifen.",
    ["Tamoxifen is a selective oestrogen receptor modulator used in breast cancer.",
     "Tamoxifen is typically given for five to ten years adjuvantly."],
    ["Tamoxifen increases the risk of endometrial carcinoma.",
     "Tamoxifen has agonist activity on bone."])
CIPROFLOXACIN = _d(
    "Ciprofloxacin", "Ciprofloxacin is a potent inhibitor of CYP1A2.",
    ["Ciprofloxacin is a fluoroquinolone antibiotic.",
     "Ciprofloxacin is active against many Gram-negative organisms."],
    ["Ciprofloxacin absorption is reduced by divalent cations.",
     "Ciprofloxacin carries a warning for tendon rupture."])
THEOPHYLLINE = _d(
    "Theophylline", "Theophylline is cleared principally by CYP1A2.",
    ["Theophylline is a methylxanthine bronchodilator.",
     "Theophylline serum concentrations are monitored in practice."],
    ["Theophylline has a narrow therapeutic range.",
     "Theophylline clearance is increased in smokers."])
OMEPRAZOLE = _d(
    "Omeprazole", "Omeprazole inhibits CYP2C19.",
    ["Omeprazole is a proton pump inhibitor.",
     "Omeprazole is used for reflux disease and peptic ulcer."],
    ["Omeprazole is acid-labile and given as an enteric-coated formulation.",
     "Omeprazole reduces gastric acid secretion for longer than its plasma half-life."])
CLOPIDOGREL = _d(
    "Clopidogrel", "Clopidogrel is a prodrug requiring CYP2C19 for activation.",
    ["Clopidogrel is an antiplatelet agent used after coronary stenting.",
     "Clopidogrel irreversibly blocks the P2Y12 receptor."],
    ["Clopidogrel's antiplatelet effect persists for the platelet lifespan.",
     "Clopidogrel is often combined with aspirin."])
PAROXETINE = _d(
    "Paroxetine", "Paroxetine is a strong inhibitor of CYP2D6.",
    ["Paroxetine is a selective serotonin reuptake inhibitor.",
     "Paroxetine is associated with discontinuation symptoms if stopped abruptly."],
    ["Paroxetine has anticholinergic activity relative to other SSRIs.",
     "Paroxetine is highly protein bound."])
METOPROLOL = _d(
    "Metoprolol", "Metoprolol is metabolised principally by CYP2D6.",
    ["Metoprolol is a beta-1 selective adrenergic blocker.",
     "Metoprolol is used in heart failure and after myocardial infarction."],
    ["Metoprolol is available as tartrate and succinate salts.",
     "Metoprolol undergoes extensive first-pass metabolism."])
KETOCONAZOLE = _d(
    "Ketoconazole", "Ketoconazole is a strong inhibitor of CYP3A4.",
    ["Ketoconazole is an imidazole antifungal.",
     "Ketoconazole absorption requires an acidic gastric environment."],
    ["Ketoconazole has been associated with hepatotoxicity.",
     "Ketoconazole inhibits adrenal steroidogenesis at high doses."])
MIDAZOLAM = _d(
    "Midazolam", "Midazolam is metabolised almost entirely by CYP3A4.",
    ["Midazolam is a short-acting benzodiazepine used for procedural sedation.",
     "Midazolam is water-soluble at low pH."],
    ["Midazolam is used as a probe substrate in interaction studies.",
     "Midazolam has an active 1-hydroxy metabolite."])

CASES: list[DDICase] = [
    DDICase("ddi-pos-01", FLUCONAZOLE, WARFARIN, True, "CYP2C9",
            "reduced warfarin clearance, raised INR and bleeding risk",
            "an inhibitor of the enzyme that clears the object drug"),
    DDICase("ddi-pos-02", CLARITHROMYCIN, SIMVASTATIN, True, "CYP3A4",
            "raised simvastatin exposure and risk of myopathy or rhabdomyolysis",
            "an inhibitor of the enzyme that clears the object drug"),
    DDICase("ddi-pos-03", RIFAMPIN, ETHINYLESTRADIOL, True, "CYP3A4",
            "accelerated clearance and possible contraceptive failure",
            "an inducer of the enzyme that clears the object drug"),
    DDICase("ddi-pos-04", FLUOXETINE, TAMOXIFEN, True, "CYP2D6",
            "reduced formation of endoxifen and possible loss of efficacy",
            "an inhibitor blocking activation of a prodrug"),
    DDICase("ddi-pos-05", CIPROFLOXACIN, THEOPHYLLINE, True, "CYP1A2",
            "reduced theophylline clearance and risk of toxicity",
            "an inhibitor of the enzyme that clears the object drug"),
    DDICase("ddi-pos-06", OMEPRAZOLE, CLOPIDOGREL, True, "CYP2C19",
            "reduced activation of clopidogrel and weaker antiplatelet effect",
            "an inhibitor blocking activation of a prodrug"),
    DDICase("ddi-pos-07", PAROXETINE, METOPROLOL, True, "CYP2D6",
            "raised metoprolol exposure with bradycardia or hypotension",
            "an inhibitor of the enzyme that clears the object drug"),
    DDICase("ddi-pos-08", KETOCONAZOLE, MIDAZOLAM, True, "CYP3A4",
            "markedly raised midazolam exposure and prolonged sedation",
            "an inhibitor of the enzyme that clears the object drug"),

    # Negative controls: same surface pattern, different enzymes. Only an agent
    # that checks enzyme identity can tell these from the positives.
    DDICase("ddi-neg-01", FLUCONAZOLE, METOPROLOL, False, "none shared",
            "no CYP-mediated interaction by this mechanism",
            "inhibitor of CYP2C9 paired with a CYP2D6 substrate"),
    DDICase("ddi-neg-02", CIPROFLOXACIN, SIMVASTATIN, False, "none shared",
            "no CYP-mediated interaction by this mechanism",
            "inhibitor of CYP1A2 paired with a CYP3A4 substrate"),
    DDICase("ddi-neg-03", OMEPRAZOLE, THEOPHYLLINE, False, "none shared",
            "no CYP-mediated interaction by this mechanism",
            "inhibitor of CYP2C19 paired with a CYP1A2 substrate"),
    DDICase("ddi-neg-04", PAROXETINE, ETHINYLESTRADIOL, False, "none shared",
            "no CYP-mediated interaction by this mechanism",
            "inhibitor of CYP2D6 paired with a CYP3A4 substrate"),
    DDICase("ddi-neg-05", CLARITHROMYCIN, WARFARIN, False, "none shared",
            "no interaction by the CYP2C9 route these dossiers describe",
            "inhibitor of CYP3A4 paired with a CYP2C9 substrate"),
    DDICase("ddi-neg-06", KETOCONAZOLE, CLOPIDOGREL, False, "none shared",
            "no interaction by the CYP2C19 route these dossiers describe",
            "inhibitor of CYP3A4 paired with a CYP2C19 substrate"),
    DDICase("ddi-neg-07", FLUOXETINE, MIDAZOLAM, False, "none shared",
            "no interaction by the CYP3A4 route these dossiers describe",
            "inhibitor of CYP2D6 paired with a CYP3A4 substrate"),
    DDICase("ddi-neg-08", RIFAMPIN, TAMOXIFEN, False, "none shared",
            "no interaction by the CYP2D6 activation route these dossiers describe",
            "inducer of CYP3A4 paired with a CYP2D6-activated prodrug"),
]


def load_cases(only: str | None = None) -> list[DDICase]:
    if not only:
        return list(CASES)
    if only == "positive":
        return [c for c in CASES if c.interacts]
    if only == "negative":
        return [c for c in CASES if not c.interacts]
    return [c for c in CASES if c.case_id == only]

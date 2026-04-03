from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    lowered = lowered.replace("/", " ")
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return f" {lowered} " if lowered else " "


def _pretty_list(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


@dataclass(frozen=True)
class EvidenceSource:
    title: str
    url: str


@dataclass(frozen=True)
class ConditionGuidance:
    name: str
    summary: str
    escalation: str
    groups: tuple[tuple[str, ...], ...]
    group_labels: tuple[str, ...]
    minimum_groups: int
    required_group_indices: tuple[int, ...] = ()
    related_labels: tuple[str, ...] = ()
    recall_floor: float = 1.0
    sources: tuple[EvidenceSource, ...] = ()


@dataclass(frozen=True)
class ConditionMatch:
    guidance: ConditionGuidance
    matched_group_indices: tuple[int, ...]
    matched_terms: tuple[str, ...]
    score: float

    @property
    def name(self) -> str:
        return self.guidance.name

    @property
    def rationale(self) -> str:
        matched_labels = [self.guidance.group_labels[idx] for idx in self.matched_group_indices]
        if not matched_labels:
            return f"Matched clinician safety pattern for {self.guidance.name}."
        return f"Matched clinician safety pattern for {self.guidance.name}: {_pretty_list(matched_labels)}."


@dataclass(frozen=True)
class CriticalBenchmarkExample:
    symptoms: str
    condition: str


def default_critical_benchmark_path(data_dir: Path) -> Path:
    return data_dir / "critical_conditions_benchmark.jsonl"


def load_critical_benchmark(path: Path) -> list[CriticalBenchmarkExample]:
    if not path.exists():
        return []

    examples: list[CriticalBenchmarkExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            examples.append(
                CriticalBenchmarkExample(
                    symptoms=row["symptoms"],
                    condition=row["condition"],
                )
            )
    return examples


def _matches_group(normalized_text: str, group: tuple[str, ...]) -> str | None:
    for term in group:
        normalized_term = _normalize_text(term)
        if normalized_term in normalized_text:
            return term
    return None


def detect_critical_conditions(query: str, condition_set: tuple[ConditionGuidance, ...] | None = None) -> list[ConditionMatch]:
    normalized_query = _normalize_text(query)
    guidances = condition_set or MUST_NOT_MISS_CONDITIONS
    matches: list[ConditionMatch] = []

    for guidance in guidances:
        matched_group_indices: list[int] = []
        matched_terms: list[str] = []
        for idx, group in enumerate(guidance.groups):
            matched_term = _matches_group(normalized_query, group)
            if matched_term is None:
                continue
            matched_group_indices.append(idx)
            matched_terms.append(matched_term)

        if len(matched_group_indices) < guidance.minimum_groups:
            continue
        if any(required_idx not in matched_group_indices for required_idx in guidance.required_group_indices):
            continue

        score = len(matched_group_indices) / max(len(guidance.groups), 1)
        matches.append(
            ConditionMatch(
                guidance=guidance,
                matched_group_indices=tuple(matched_group_indices),
                matched_terms=tuple(matched_terms),
                score=score,
            )
        )

    matches.sort(key=lambda item: (item.score, len(item.matched_group_indices)), reverse=True)
    return matches


def guidance_for_label(label: str) -> ConditionGuidance | None:
    normalized_label = label.strip().lower()
    for guidance in MUST_NOT_MISS_CONDITIONS:
        if normalized_label == guidance.name:
            return guidance
        if normalized_label in guidance.related_labels:
            return guidance
    return None


def evidence_cards_for_conditions(safety_matches: list[ConditionMatch], labels: list[str]) -> list[dict[str, object]]:
    cards: list[dict[str, object]] = []
    seen: set[str] = set()

    for match in safety_matches:
        if match.name in seen:
            continue
        seen.add(match.name)
        cards.append(
            {
                "condition": match.name,
                "summary": match.guidance.summary,
                "rationale": match.rationale,
                "escalation": match.guidance.escalation,
                "sources": [{"title": source.title, "url": source.url} for source in match.guidance.sources],
            }
        )

    for label in labels:
        guidance = guidance_for_label(label)
        if guidance is None or guidance.name in seen:
            continue
        seen.add(guidance.name)
        cards.append(
            {
                "condition": guidance.name,
                "summary": guidance.summary,
                "rationale": f"Curated clinician reference for {guidance.name}.",
                "escalation": guidance.escalation,
                "sources": [{"title": source.title, "url": source.url} for source in guidance.sources],
            }
        )
    return cards


def build_assessment_sentence(
    *,
    primary_label: str,
    alternative_labels: list[str],
    unknown: bool,
    low_confidence: bool,
    safety_matches: list[ConditionMatch],
) -> str:
    urgent_conditions = [match.name for match in safety_matches[:3]]
    alternatives = [label for label in alternative_labels if label and label != primary_label][:2]

    if unknown and urgent_conditions:
        return (
            "This symptom pattern does not map cleanly to a known diagnosis in the model, "
            f"so urgent clinician review is recommended and {_pretty_list(urgent_conditions)} should be ruled out."
        )
    if unknown:
        return (
            "This symptom pattern does not map cleanly to a known diagnosis in the model, "
            "so it should be treated as an unknown presentation that needs clinician review."
        )

    if alternatives and safety_matches:
        return (
            f"This symptom pattern is most consistent with {primary_label}, but {_pretty_list(alternatives)} "
            "should also be considered, and the red-flag features here justify urgent in-person evaluation."
        )
    if alternatives and low_confidence:
        return (
            f"This symptom pattern is most consistent with {primary_label}, but {_pretty_list(alternatives)} "
            "are also plausible and this remains a low-confidence differential."
        )
    if alternatives:
        return f"This symptom pattern is most consistent with {primary_label}, but {_pretty_list(alternatives)} are also plausible."
    if safety_matches:
        return (
            f"This symptom pattern is most consistent with {primary_label}, "
            "and the red-flag features here justify urgent in-person evaluation."
        )
    return f"This symptom pattern is most consistent with {primary_label}."


MUST_NOT_MISS_CONDITIONS: tuple[ConditionGuidance, ...] = (
    ConditionGuidance(
        name="rabies",
        summary=(
            "CDC notes that rabies after an exposure can begin with fever or headache plus discomfort, prickling, "
            "or itching at the bite site, and once neurologic symptoms start it is nearly always fatal."
        ),
        escalation="Urgent same-day medical evaluation is needed after a possible exposure, especially if neurologic symptoms are present.",
        groups=(
            ("animal bite", "animal scratch", "bite site", "bit me", "scratched me", "bat bite", "dog bite", "rabid animal"),
            ("tingling", "prickling", "itching", "burning", "numbness"),
            ("hydrophobia", "fear of water", "cannot swallow water", "trouble swallowing", "dysphagia", "throat spasms"),
            ("confusion", "delirium", "hallucinations", "abnormal behavior", "agitation", "insomnia"),
            ("paralysis", "weakness", "drooling", "excessive saliva"),
        ),
        group_labels=(
            "possible bite or scratch exposure",
            "bite-site tingling or abnormal sensation",
            "hydrophobia or swallowing difficulty",
            "neurologic or behavioral symptoms",
            "progressive weakness, paralysis, or excess saliva",
        ),
        minimum_groups=3,
        sources=(
            EvidenceSource("CDC: About Rabies", "https://www.cdc.gov/rabies/about/index.html"),
            EvidenceSource("CDC: Clinical Features of Rabies", "https://www.cdc.gov/rabies/hcp/clinical-signs/index.html"),
        ),
    ),
    ConditionGuidance(
        name="sepsis",
        summary=(
            "CDC describes sepsis as the body's extreme response to infection and lists confusion, shortness of breath, "
            "clammy skin, extreme pain, fever or feeling very cold, and high heart rate or weak pulse as warning signs."
        ),
        escalation="Immediate emergency evaluation is recommended because sepsis can rapidly progress to organ failure and death.",
        groups=(
            (
                "infection",
                "infected",
                "uti",
                "urinary infection",
                "pneumonia",
                "wound",
                "fever",
                "chills",
                "shivering",
                "shivery",
                "very cold",
            ),
            ("confusion", "disorientation", "altered mental status"),
            ("shortness of breath", "short of breath", "trouble breathing", "difficulty breathing"),
            ("clammy", "sweaty", "sweating"),
            ("high heart rate", "weak pulse", "pulse is weak", "racing heart", "heart is racing", "rapid heart rate", "tachycardia"),
            ("extreme pain", "severe pain", "extreme discomfort"),
        ),
        group_labels=(
            "possible infection or fever",
            "confusion or disorientation",
            "shortness of breath",
            "clammy or sweaty skin",
            "high heart rate or weak pulse",
            "extreme pain or discomfort",
        ),
        minimum_groups=3,
        sources=(
            EvidenceSource("CDC: About Sepsis", "https://www.cdc.gov/sepsis/about/index.html"),
        ),
    ),
    ConditionGuidance(
        name="stroke",
        summary=(
            "CDC highlights sudden one-sided weakness or numbness, confusion or speech difficulty, vision change, "
            "trouble walking or loss of balance, and sudden severe headache as stroke warning signs."
        ),
        escalation="Call emergency services right away because stroke treatment is time-sensitive and every minute counts.",
        groups=(
            ("face droop", "facial droop", "one side of the face", "facial numbness"),
            (
                "arm weakness",
                "leg weakness",
                "one sided weakness",
                "one-sided weakness",
                "weakness in my left arm",
                "weakness in my right arm",
                "weakness in my left leg",
                "weakness in my right leg",
                "numbness in the arm",
                "numbness in the leg",
                "numbness on one side",
            ),
            ("trouble speaking", "slurred speech", "difficulty understanding speech", "difficulty speaking", "cannot speak", "speech difficulty"),
            ("vision loss", "trouble seeing", "blurred vision", "double vision"),
            ("loss of balance", "dizziness", "trouble walking", "lack of coordination"),
            ("sudden severe headache", "worst headache", "severe headache"),
        ),
        group_labels=(
            "facial droop or facial numbness",
            "one-sided arm or leg weakness",
            "speech or language difficulty",
            "vision change",
            "balance or gait problem",
            "sudden severe headache",
        ),
        minimum_groups=2,
        sources=(
            EvidenceSource("CDC: Signs and Symptoms of Stroke", "https://www.cdc.gov/stroke/signs-symptoms/index.html"),
        ),
    ),
    ConditionGuidance(
        name="meningitis",
        summary=(
            "CDC lists fever, headache, and stiff neck as common meningitis symptoms, often with confusion, nausea, "
            "vomiting, photophobia, or rapidly worsening illness."
        ),
        escalation="Urgent emergency evaluation is recommended because bacterial meningitis can worsen within hours.",
        groups=(
            ("fever", "high fever"),
            ("headache", "severe headache"),
            ("stiff neck", "neck stiffness", "cannot bend my neck"),
            ("confusion", "altered mental status", "delirium"),
            ("photophobia", "light hurts my eyes", "sensitive to light"),
            ("vomiting", "nausea", "rash", "petechial", "purpuric"),
        ),
        group_labels=(
            "fever",
            "headache",
            "stiff neck",
            "confusion or altered mental status",
            "photophobia",
            "vomiting or rash",
        ),
        minimum_groups=3,
        sources=(
            EvidenceSource("CDC: Meningococcal Disease Symptoms", "https://www.cdc.gov/meningococcal/symptoms/index.html"),
        ),
    ),
    ConditionGuidance(
        name="acute coronary syndrome",
        summary=(
            "NHLBI warns that heart attack symptoms can include chest pain or heaviness, pain in the arms, back, neck, "
            "jaw, or upper abdomen, shortness of breath, sweating, nausea, dizziness, and irregular heartbeat."
        ),
        escalation="Emergency evaluation is recommended because possible acute coronary syndrome should not be managed as a routine outpatient complaint.",
        groups=(
            ("chest pain", "chest pressure", "chest heaviness", "chest discomfort", "tightness in my chest"),
            ("arm pain", "left arm", "right arm", "back pain", "shoulder pain", "neck pain", "jaw pain"),
            ("shortness of breath", "breathless", "difficulty breathing"),
            ("sweating", "cold sweat", "clammy"),
            ("nausea", "vomiting"),
            ("dizziness", "light headed", "lightheaded", "faint"),
        ),
        group_labels=(
            "chest pain or pressure",
            "radiating arm, back, shoulder, neck, or jaw pain",
            "shortness of breath",
            "sweating or clamminess",
            "nausea or vomiting",
            "dizziness or faintness",
        ),
        minimum_groups=2,
        required_group_indices=(0,),
        sources=(
            EvidenceSource("NHLBI: Heart Attack Symptoms", "https://www.nhlbi.nih.gov/health/heart-attack/symptoms"),
        ),
    ),
    ConditionGuidance(
        name="ectopic pregnancy",
        summary=(
            "ACOG notes that ectopic pregnancy may present with pelvic or abdominal pain and abnormal bleeding early on, "
            "with severe pain, shoulder pain, weakness, dizziness, or fainting if rupture occurs."
        ),
        escalation="Immediate emergency evaluation is recommended because ectopic pregnancy can rupture and cause life-threatening bleeding.",
        groups=(
            ("pregnant", "positive pregnancy test", "missed period", "late period"),
            ("vaginal bleeding", "spotting", "abnormal bleeding"),
            ("pelvic pain", "lower abdominal pain", "abdominal pain", "one sided pelvic pain", "cramping"),
            ("shoulder pain", "fainting", "dizziness", "weakness", "passed out"),
        ),
        group_labels=(
            "pregnancy context",
            "abnormal vaginal bleeding",
            "pelvic or abdominal pain",
            "rupture warning signs such as shoulder pain or fainting",
        ),
        minimum_groups=2,
        required_group_indices=(0,),
        sources=(
            EvidenceSource("ACOG: Ectopic Pregnancy", "https://www.acog.org/womens-health/faqs/ectopic-pregnancy"),
            EvidenceSource("MedlinePlus: Ectopic Pregnancy", "https://medlineplus.gov/ectopicpregnancy.html"),
        ),
    ),
)

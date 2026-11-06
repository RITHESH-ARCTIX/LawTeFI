"""Statutory Cross-Referencing Engine (Transitional Provision Mapper)."""

from __future__ import annotations

import re
from typing import Any
from fastapi import APIRouter, HTTPException, Query

from .retrieval import retrieve

router = APIRouter(prefix="/api/v1/cross-reference", tags=["cross-reference"])

# Curated transitional mapping dictionary for major penal and procedural provisions
CONCORDANCE_MAP: dict[str, dict[str, Any]] = {
    # IPC -> BNS
    "ipc_302": {
        "old_act": "Indian Penal Code, 1860",
        "old_section": "Section 302",
        "old_title": "Punishment for murder",
        "new_act": "Bharatiya Nyaya Sanhita, 2023",
        "new_section": "Section 103(1)",
        "new_title": "Punishment for murder",
        "substantive_changes": [
            "Section 103(1) prescribes death or imprisonment for life, and fine (mirroring IPC 302).",
            "Section 103(2) introduces a distinct offense and penalty for murder committed by a group of five or more persons on grounds of race, caste, sex, place of birth, language, or personal belief (mob lynching/hate killing)."
        ],
        "search_query": "Section 103 Bharatiya Nyaya Sanhita punishment for murder"
    },
    "ipc_304": {
        "old_act": "Indian Penal Code, 1860",
        "old_section": "Section 304",
        "old_title": "Punishment for culpable homicide not amounting to murder",
        "new_act": "Bharatiya Nyaya Sanhita, 2023",
        "new_section": "Section 105",
        "new_title": "Punishment for culpable homicide not amounting to murder",
        "substantive_changes": [
            "Penal structure split across Part I (intent) and Part II (knowledge).",
            "Minimum term of imprisonment of not less than 5 years (extendable to 10 years or life) for intentional acts."
        ],
        "search_query": "Section 105 Bharatiya Nyaya Sanhita punishment for culpable homicide"
    },
    "ipc_304a": {
        "old_act": "Indian Penal Code, 1860",
        "old_section": "Section 304A",
        "old_title": "Causing death by negligence",
        "new_act": "Bharatiya Nyaya Sanhita, 2023",
        "new_section": "Section 106",
        "new_title": "Causing death by negligence",
        "substantive_changes": [
            "Section 106(1) provides up to 5 years imprisonment and fine for general rash/negligent acts.",
            "Section 106(2) provides up to 10 years imprisonment for rash and negligent driving who escapes without reporting to police (hit and run)."
        ],
        "search_query": "Section 106 Bharatiya Nyaya Sanhita causing death by negligence"
    },
    "ipc_378": {
        "old_act": "Indian Penal Code, 1860",
        "old_section": "Section 378",
        "old_title": "Theft (definition)",
        "new_act": "Bharatiya Nyaya Sanhita, 2023",
        "new_section": "Section 303(1)",
        "new_title": "Theft",
        "substantive_changes": [
            "Consolidated definitions and punishment under a single statutory head.",
            "Retains the five core explanations of movable property and dishonest taking."
        ],
        "search_query": "Section 303 Bharatiya Nyaya Sanhita theft definition"
    },
    "ipc_379": {
        "old_act": "Indian Penal Code, 1860",
        "old_section": "Section 379",
        "old_title": "Punishment for theft",
        "new_act": "Bharatiya Nyaya Sanhita, 2023",
        "new_section": "Section 303(2)",
        "new_title": "Punishment for theft",
        "substantive_changes": [
            "Imprisonment up to 3 years, or fine, or both.",
            "Provides community service for first-time offenders where the stolen property value is less than ₹5,000 upon return or restitution."
        ],
        "search_query": "Section 303 Bharatiya Nyaya Sanhita punishment for theft community service"
    },
    "ipc_420": {
        "old_act": "Indian Penal Code, 1860",
        "old_section": "Section 420",
        "old_title": "Cheating and dishonestly inducing delivery of property",
        "new_act": "Bharatiya Nyaya Sanhita, 2023",
        "new_section": "Section 318(4)",
        "new_title": "Cheating and dishonestly inducing delivery of property",
        "substantive_changes": [
            "Offense of cheating restructured under Section 318 with sub-clauses for simple cheating and aggravated cheating.",
            "Punishment retains imprisonment up to 7 years and fine."
        ],
        "search_query": "Section 318 Bharatiya Nyaya Sanhita cheating"
    },
    "ipc_375": {
        "old_act": "Indian Penal Code, 1860",
        "old_section": "Section 375 / 376",
        "old_title": "Rape / Punishment for rape",
        "new_act": "Bharatiya Nyaya Sanhita, 2023",
        "new_section": "Section 63 / Section 64",
        "new_title": "Rape / Punishment for rape",
        "substantive_changes": [
            "Grouped under Chapter V ('Of Offences Against Woman and Child').",
            "Added Section 69 punishing sexual intercourse by employing deceitful means or false promise of marriage."
        ],
        "search_query": "Section 64 Bharatiya Nyaya Sanhita punishment for rape"
    },
    "ipc_124a": {
        "old_act": "Indian Penal Code, 1860",
        "old_section": "Section 124A",
        "old_title": "Sedition",
        "new_act": "Bharatiya Nyaya Sanhita, 2023",
        "new_section": "Section 152",
        "new_title": "Act endangering sovereignty, unity and integrity of India",
        "substantive_changes": [
            "The term 'sedition' is omitted from the statutory text.",
            "Explicitly penalizes acts exciting secession, armed rebellion, or subversive activities endangering sovereignty.",
            "Punishment increased from 3 years to up to 7 years or life imprisonment."
        ],
        "search_query": "Section 152 Bharatiya Nyaya Sanhita sovereignty unity integrity"
    },
    # CrPC -> BNSS
    "crpc_154": {
        "old_act": "Code of Criminal Procedure, 1973",
        "old_section": "Section 154",
        "old_title": "Information in cognizable cases (FIR)",
        "new_act": "Bharatiya Nagarik Suraksha Sanhita, 2023",
        "new_section": "Section 173",
        "new_title": "Information in cognizable cases (Zero FIR / e-FIR)",
        "substantive_changes": [
            "Formal recognition of Zero FIR regardless of police station territorial jurisdiction.",
            "Statutory allowance for electronic information (e-FIR), requiring signature within three days.",
            "Preliminary inquiry permitted within 14 days before registering an FIR for offenses punishable by 3 to 7 years imprisonment."
        ],
        "search_query": "Section 173 Bharatiya Nagarik Suraksha Sanhita information cognizable cases"
    },
    "crpc_41a": {
        "old_act": "Code of Criminal Procedure, 1973",
        "old_section": "Section 41A",
        "old_title": "Notice of appearance before police officer",
        "new_act": "Bharatiya Nagarik Suraksha Sanhita, 2023",
        "new_section": "Section 35(3)",
        "new_title": "Notice of appearance",
        "substantive_changes": [
            "Prior approval of an officer not below Deputy Superintendent of Police required before arresting elderly individuals or those suffering from infirmity for offenses punishable under 3 years."
        ],
        "search_query": "Section 35 Bharatiya Nagarik Suraksha Sanhita arrest"
    },
    "crpc_167": {
        "old_act": "Code of Criminal Procedure, 1973",
        "old_section": "Section 167",
        "old_title": "Procedure when investigation cannot be completed in twenty-four hours",
        "new_act": "Bharatiya Nagarik Suraksha Sanhita, 2023",
        "new_section": "Section 187",
        "new_title": "Procedure when investigation cannot be completed in twenty-four hours",
        "substantive_changes": [
            "Police custody of 15 days can now be sought in whole or in parts across the initial 40 or 60 days of detention."
        ],
        "search_query": "Section 187 Bharatiya Nagarik Suraksha Sanhita custody"
    },
    # IEA -> BSA
    "iea_65b": {
        "old_act": "Indian Evidence Act, 1872",
        "old_section": "Section 65B",
        "old_title": "Admissibility of electronic records",
        "new_act": "Bharatiya Sakshya Adhiniyam, 2023",
        "new_section": "Section 63",
        "new_title": "Admissibility of electronic records",
        "substantive_changes": [
            "Electronic records placed on equal legal footing with primary documentary evidence.",
            "Introduced standardized certificate format in the Schedule to the Adhiniyam."
        ],
        "search_query": "Section 63 Bharatiya Sakshya Adhiniyam electronic records"
    }
}

TOPIC_KEYWORDS: dict[str, str] = {
    "theft": "ipc_378",
    "stealing": "ipc_378",
    "murder": "ipc_302",
    "culpable homicide": "ipc_304",
    "negligence": "ipc_304a",
    "hit and run": "ipc_304a",
    "cheating": "ipc_420",
    "fraud": "ipc_420",
    "rape": "ipc_375",
    "sedition": "ipc_124a",
    "fir": "crpc_154",
    "zero fir": "crpc_154",
    "e-fir": "crpc_154",
    "arrest": "crpc_41a",
    "remand": "crpc_167",
    "custody": "crpc_167",
    "electronic evidence": "iea_65b",
    "electronic record": "iea_65b",
}

def parse_cross_reference_query(query: str) -> str | None:
    """Normalize input query into a lookup key (e.g., 'IPC 302' -> 'ipc_302' or 'theft' -> 'ipc_378')."""
    clean = query.strip().casefold()
    clean_no_sec = clean.replace("section", "").replace("sec", "").replace("s.", "").strip()

    # 1. Exact statute + number match (e.g. 'ipc 378', 'crpc 154')
    match_code_first = re.search(r"\b(ipc|crpc|iea)\s*(\d+[a-z]?)\b", clean_no_sec)
    if match_code_first:
        act, sec = match_code_first.groups()
        return f"{act}_{sec}"

    match_sec_first = re.search(r"\b(\d+[a-z]?)\s*(ipc|crpc|iea)\b", clean_no_sec)
    if match_sec_first:
        sec, act = match_sec_first.groups()
        return f"{act}_{sec}"

    # 2. Standalone section numbers where IPC is implied
    match_num = re.search(r"\b(\d{3}[a-z]?)\b", clean_no_sec)
    if match_num:
        candidate = f"ipc_{match_num.group(1)}"
        if candidate in CONCORDANCE_MAP:
            return candidate

    # 3. Topic keywords (e.g. 'theft', 'cheating', 'murder')
    for word, key in TOPIC_KEYWORDS.items():
        if word in clean:
            return key

    return None

@router.get("")
def lookup_cross_reference(
    query: str = Query(..., min_length=2, description="Statutory provision query, e.g., 'IPC 302' or 'CrPC 154'")
) -> dict[str, Any]:
    """Retrieve transitional mapping between old and new criminal law provisions."""
    key = parse_cross_reference_query(query)

    if not key or key not in CONCORDANCE_MAP:
        raise HTTPException(
            status_code=404,
            detail=f"No direct concordance mapping found for '{query}'. Try queries such as 'IPC 302', 'IPC 420', 'IPC 379', or 'CrPC 154'."
        )

    mapping = CONCORDANCE_MAP[key]

    # Retrieve live passage evidence from Chroma index
    retrieved_evidence = []
    try:
        passages = retrieve(mapping["search_query"], limit=2)
        for p in passages:
            retrieved_evidence.append({
                "title": p.get("title"),
                "section": p.get("section"),
                "heading": p.get("heading"),
                "pages": p.get("pages"),
                "excerpt": str(p.get("text") or "")[:500]
            })
    except Exception:
        pass

    return {
        "status": "ok",
        "query": query,
        "concordance": mapping,
        "grounded_evidence": retrieved_evidence
    }
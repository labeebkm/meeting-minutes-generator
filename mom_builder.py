from __future__ import annotations

import re
from typing import Dict, List


# --------------------------------------------------
# Sentence categorization rules
# --------------------------------------------------

DECISION_PATTERNS = [
    r"\bdecided\b",
    r"\bfinalized\b",
    r"\bagreed\b",
    r"\bconfirmed\b",
]

ACTION_PATTERNS = [
    r"\bwe will\b",
    r"\bneed to\b",
    r"\bshould\b",
    r"\bhas to\b",
    r"\bmust\b",
    r"\baction\b",
    r"\bdeadline\b",
    r"\bfollow up\b",
]

EXAMPLE_MARKERS = [
    "for example",
    "for instance",
    "suppose",
    "imagine",
    "like ",
]


# --------------------------------------------------
# Core helpers
# --------------------------------------------------

def _is_example(sentence: str) -> bool:
    return any(m in sentence.lower() for m in EXAMPLE_MARKERS)


def _formalize(sentence: str) -> str:
    """
    Convert casual speech into MoM-style business phrasing.
    """
    sentence = sentence.strip()

    replacements = [
        (r"\bi think\b", "It was noted that"),
        (r"\bwe discussed\b", "The team discussed"),
        (r"\bhe explained\b", "It was explained that"),
        (r"\bthey explained\b", "It was explained that"),
        (r"\bi am going to\b", "The team plans to"),
        (r"\bwe are going to\b", "The team plans to"),
    ]

    for pat, repl in replacements:
        sentence = re.sub(pat, repl, sentence, flags=re.IGNORECASE)

    # Capitalize first letter
    return sentence[:1].upper() + sentence[1:]


# --------------------------------------------------
# Public API
# --------------------------------------------------

def build_mom(transcript: str) -> Dict[str, List[str]]:
    """
    Build a structured Minutes of Meeting from cleaned transcript.
    """

    lines = [ln.strip() for ln in transcript.splitlines() if ln.strip()]

    discussion: List[str] = []
    decisions: List[str] = []
    actions: List[str] = []

    for ln in lines:
        # Remove speaker labels
        ln = re.sub(r"^Speaker\s+\d+:\s*", "", ln).strip()

        # Skip trivial or noisy lines
        if len(ln.split()) < 5:
            continue

        # Drop example-heavy content
        if _is_example(ln):
            continue

        # Categorize
        if any(re.search(p, ln, re.IGNORECASE) for p in DECISION_PATTERNS):
            decisions.append(_formalize(ln))
        elif any(re.search(p, ln, re.IGNORECASE) for p in ACTION_PATTERNS):
            actions.append(_formalize(ln))
        else:
            discussion.append(_formalize(ln))

    # De-duplicate while preserving order
    def dedupe(items: List[str]) -> List[str]:
        seen = set()
        out = []
        for i in items:
            key = i.lower()
            if key not in seen:
                seen.add(key)
                out.append(i)
        return out

    return {
        "discussion": dedupe(discussion)[:8],
        "decisions": dedupe(decisions)[:6],
        "actions": dedupe(actions)[:6],
    }

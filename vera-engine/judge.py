"""
judge.py — LLM evidence classification for TalentLens's BM25 + judge pipeline.

Given ONE requirement and the evidence chunks retrieved for it (retrieval.py),
asks a small local model to classify how well that evidence supports the
requirement. Same "LLM never does arithmetic" principle used throughout this
codebase (see extractor.py / experience.py): the judge only classifies —
Python (matcher.py) converts the classification into a numeric contribution
and calculates the actual score. The judge is also never shown the whole
resume, only the handful of chunks BM25 already retrieved, and is told
explicitly that retrieval can be wrong — this keeps a bad retrieval from
being rubber-stamped as a match.
"""

import json
import logging

import ollama

logger = logging.getLogger("talentlens.judge")

JUDGE_MODEL = "gemma3:1b"

_LEVELS = ("direct", "related", "weak", "none")

# Passed as `format` to ollama.chat instead of the bare string "json". "json" only forces
# *some* valid JSON object back — it does NOT force the object to contain match/confidence/
# reason. Confirmed in testing: gemma3:1b would happily return {"match": "direct"} and stop,
# satisfying "valid JSON" while silently dropping the two fields the prompt asked for. A real
# JSON Schema with "required" makes Ollama's structured-output grammar force those keys to
# actually be emitted, not just requested in prose.
_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "match": {"type": "string", "enum": list(_LEVELS)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string", "minLength": 1},
    },
    "required": ["match", "confidence", "reason"],
}

_SYSTEM_PROMPT = (
    "You are an expert technical recruiter's assistant. You will be given ONE job "
    "requirement and one or more pieces of evidence retrieved from a candidate's resume "
    "via keyword search — the retrieval is not always relevant, since it is based on word "
    "overlap, not meaning. Your job is to classify how well the evidence actually supports "
    "the candidate having that requirement.\n\n"
    "Use EXACTLY one of these four levels:\n"
    '- "direct": the evidence explicitly demonstrates this requirement — the skill, tool, or '
    "concept is clearly present, named, or unambiguously described.\n"
    '- "related": the evidence shows closely related or adjacent work that strongly implies '
    "this requirement, without explicitly naming or describing it.\n"
    '- "weak": the evidence is only tangentially related — plausible, but not a strong signal.\n'
    '- "none": the evidence does not support this requirement at all (including when the '
    "retrieved evidence is simply irrelevant to the requirement).\n\n"
    "Base your judgment ONLY on the evidence text provided. Do not assume a skill is present "
    "just because it is common for similar roles, and do not let the requirement's own wording "
    "influence your answer beyond what the evidence actually shows. Respond with ONLY a JSON "
    "object matching this schema, no other text:\n"
    '{"match": "direct" | "related" | "weak" | "none", "confidence": number between 0 and 1, '
    '"reason": "one short sentence explaining the classification"}'
)


def _build_user_prompt(requirement: str, evidence_chunks: list[dict]) -> str:
    evidence_text = "\n\n".join(
        f"[{c['source_type']} — {c['source_label']}]\n{c['text']}" for c in evidence_chunks
    )
    return f"REQUIREMENT:\n{requirement}\n\nRESUME EVIDENCE:\n{evidence_text}\n\nCLASSIFY the match level."


def judge_evidence(requirement: str, evidence_chunks: list[dict]) -> dict:
    """
    Returns {"match": "direct"|"related"|"weak"|"none", "confidence": float 0-1, "reason": str}.

    If there's no evidence to judge (BM25 retrieved nothing), this short-circuits to "none"
    without calling the model at all — there is nothing for the LLM to usefully classify, and
    skipping the call saves one round trip per missing requirement per candidate, which adds
    up fast across a JD with many requirements and a large resume batch.

    On any call/parse failure, or if the model returns something outside the 4 allowed levels,
    this fails closed to "none" rather than guessing — same principle as extractor.py's
    unparseable-JSON handling: an ambiguous judge output must never silently count as a match.
    """
    if not evidence_chunks:
        return {"match": "none", "confidence": 1.0, "reason": "No evidence retrieved for this requirement."}

    try:
        response = ollama.chat(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(requirement, evidence_chunks)},
            ],
            format=_RESPONSE_SCHEMA,
            options={"temperature": 0.0, "num_predict": 256},
        )
        raw = response["message"]["content"]
        parsed = json.loads(raw)

        match = str(parsed.get("match", "")).strip().lower()
        if match not in _LEVELS:
            logger.warning(
                "Judge returned unrecognized match level %r for requirement %r — treating as 'none'.",
                match, requirement,
            )
            return {"match": "none", "confidence": 0.0, "reason": f"Unrecognized judge output: {match!r}"}

        try:
            confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0

        return {"match": match, "confidence": confidence, "reason": str(parsed.get("reason", ""))}

    except (json.JSONDecodeError, KeyError, TypeError, ollama.ResponseError) as e:
        logger.warning("Judge call failed for requirement %r: %s — treating as 'none'.", requirement, e)
        return {"match": "none", "confidence": 0.0, "reason": f"Judge call failed: {e}"}
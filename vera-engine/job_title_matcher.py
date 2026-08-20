"""
job_title_matcher.py — job-title relevance scoring (the "Job Titles" scoring
category — one of the ATS-style comparison points alongside Industry Keywords
and Soft Skills).

Previous revision used pure token-overlap (Jaccard) — deliberately deterministic
and cheap, but with a known, explicitly-documented limitation: it misses true
synonyms/related roles that share no words at all (e.g. "AI Architect" vs
required "Data Engineer" scores 0 despite plausibly being a related role,
depending on context). That limitation is exactly what this revision fixes.

Now reuses the SAME judge (judge.py) already used for skill/requirement
evidence classification, rather than introducing a second similarity
mechanism (embeddings) or a synonym table to maintain. All of a candidate's
past titles — plus, if extracted, their current-role title pulled from a
resume Summary/Profile section (see extractor.py's
current_role_title_from_summary field) — are handed to the judge as evidence
chunks in ONE call, same shape as a BM25-retrieved evidence list for a skill.
This deliberately costs exactly one extra judge call per candidate (not one
per past title), keeping this in line with the rest of the pipeline's
"one model, minimal calls" design.

Jaccard token overlap is NOT gone — it's kept purely as a deterministic,
inspectable way to pick which past title to surface as "best_match" for
display (e.g. in the Evidence panel), sorted alongside every other title in
all_titles. It no longer drives the actual score; the judge's
direct/related/weak/none classification does, via the same
MATCH_LEVEL_CONTRIBUTION mapping matcher.py already uses for skills — so a
job-title match and a skill match mean the same thing on the same 0-1 scale.
"""

import re
import difflib
from judge import judge_evidence
from matcher import MATCH_LEVEL_CONTRIBUTION

STOPWORDS = {"the", "a", "an", "of", "and", "for"}


def _tokenize(text: str) -> set:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {w for w in words if w not in STOPWORDS}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


# --- Groundedness check: catch the judge citing the JD's own title as if it
# were evidence -----------------------------------------------------------
#
# Confirmed in production: for a candidate whose actual titles share zero
# words with the JD's target_role_title, the judge still returned "direct"
# with a reason like "the evidence explicitly states 'Manager' ... directly
# related to Data Engineer roles involving AI data platforms" - where
# "Data Engineer" / "AI data platforms" are the JD's OWN words, appearing
# nowhere in the candidate's actual titles, but cited as if they were found
# in the evidence. This happened for two different candidates with two
# different (unrelated) actual titles, both judged "direct" against the
# same JD title with the same tell: the reasoning quotes JD language back
# as though it were evidence.
#
# This is deliberately NOT a Jaccard floor - the whole point of this
# revision was to let a true zero-overlap synonym match (e.g. "AI
# Architect" for "Data Engineer") still earn credit via the judge, and a
# hard token-overlap gate would defeat that. Instead this checks something
# narrower and fully deterministic: does every quoted phrase in the judge's
# OWN reason actually appear in the evidence text it was given? A reason
# quoting something absent from the evidence is a demonstrably false
# citation regardless of whether the underlying match might otherwise be
# real - so this can't (falsely) reject a genuine synonym call as long as
# the judge's reasoning doesn't misattribute JD language to the evidence
# while making it.
_QUOTED_PHRASE_RE = re.compile(r"['\u2018\u2019]([^'\u2018\u2019]{2,80})['\u2018\u2019]|[\"\u201c\u201d]([^\"\u201c\u201d]{2,80})[\"\u201c\u201d]")


def _extract_quoted_phrases(text: str) -> list[str]:
    phrases = []
    for m in _QUOTED_PHRASE_RE.finditer(text or ""):
        phrase = m.group(1) or m.group(2)
        if phrase:
            phrases.append(phrase.strip())
    return phrases


def _normalize_for_containment(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[-_/(),]", " ", (text or "").lower())).strip()


def _phrase_grounded_in(phrase: str, source_text: str) -> bool:
    phrase_norm = _normalize_for_containment(phrase)
    if not phrase_norm:
        return True  # nothing meaningful to check - don't penalize on empty/punctuation-only quotes
    return phrase_norm in _normalize_for_containment(source_text)


def _reason_falsely_cites_jd_title(judge_reason: str, evidence_titles_text: str, target_role_title: str) -> str | None:
    """
    Returns a description of the false citation if the judge's reason
    contains language drawn from the JD's target_role_title that does not
    appear anywhere in the candidate's actual evidence titles - i.e. the
    judge attributed JD language to the candidate's evidence. Checks two
    shapes of this, since it showed up both ways in production:

    1. Quoted citation - the reason explicitly quotes a phrase (e.g. "the
       evidence explicitly states 'Data Engineer'") that isn't in the
       evidence but IS in the JD title.
    2. Unquoted leakage - the reason just uses JD-title language directly in
       its own prose (e.g. "...directly related to Data Engineer roles
       involving AI data platforms") without quoting it at all. This one
       needs the JD title split on its own delimiters (em-dash, slash, pipe)
       into segments, since a compound title like "Data Engineer — AI Data
       Platform" leaking as either half is just as ungrounded as the whole
       phrase leaking.

    Returns None if no such phrase is found (reasoning is grounded, or at
    least not provably mis-citing the JD title specifically).
    """
    quoted_phrases = _extract_quoted_phrases(judge_reason)

    title_segments = [seg.strip() for seg in re.split(r"[—\-/|]", target_role_title) if len(seg.strip().split()) >= 2]
    if len(target_role_title.split()) >= 2:
        title_segments.append(target_role_title.strip())

    for phrase in quoted_phrases + title_segments:
        if not phrase:
            continue
        if _phrase_grounded_in(phrase, evidence_titles_text):
            continue  # actually present in the candidate's evidence - fine
        if _phrase_grounded_in(phrase, judge_reason) and _phrase_grounded_in(phrase, target_role_title):
            return (
                f"judge's reasoning uses '{phrase}', which appears only in the JD's target role "
                f"title ('{target_role_title}'), not in any of the candidate's actual titles — "
                f"treating as an unsupported match rather than trusting it."
            )
    return None


def _collect_candidate_titles(candidate_data: dict) -> list[dict]:
    """
    Every distinct title worth judging against the JD's role title: the
    candidate's past experience-entry titles, plus — if extractor.py found
    one — a title stated in the resume's Summary/Profile section
    (current_role_title_from_summary). The summary title is included even
    when it duplicates the first experience entry's title (harmless — dedup
    below handles it) and matters most when a resume's most recent role
    ISN'T clearly the first dated experience entry (unusual ordering,
    a title that only appears in prose, etc.) — exactly the gap the summary
    field exists to cover.

    Deduplicates case-insensitively, preserving first-seen order (summary
    title first, since it's the most likely "current" signal).
    """
    titles: list[dict] = []
    seen = set()

    summary_title = (candidate_data.get("current_role_title_from_summary") or "").strip()
    if summary_title:
        titles.append({"title": summary_title, "company": "(from resume summary)"})
        seen.add(summary_title.lower())

    for entry in candidate_data.get("experience", []) or []:
        title = (entry.get("title") or "").strip()
        if not title:
            continue
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        titles.append({"title": title, "company": entry.get("company") or "unknown company"})

    return titles


def score_job_titles(candidate_data: dict, target_role_title: str, judge_fn=judge_evidence) -> dict:
    """
    Scores how closely a candidate's job titles (past roles + summary-stated
    current title) relate to the JD's role_title.
    
    Applies a deterministic check on the latest role first: >= 50% similarity 
    yields an 85% match. Falls back to a single judge call if not met.
    """
    titles = _collect_candidate_titles(candidate_data)
    target_role_title = (target_role_title or "").strip()

    if not titles or not target_role_title:
        return {
            "contribution": 0.0,
            "best_match": None,
            "status": "missing",
            "all_titles": [],
            "match_level": "none",
            "judge_reason": "" if target_role_title else "No JD role title to compare against.",
        }

    target_tokens = _tokenize(target_role_title)
    all_titles = [
        {"title": t["title"], "company": t["company"], "jaccard": round(_jaccard(target_tokens, _tokenize(t["title"])), 3)}
        for t in titles
    ]
    all_titles.sort(key=lambda t: t["jaccard"], reverse=True)

    # NEW: Deterministic bypass check on the latest/current job title
    latest_title_dict = titles[0]
    similarity_ratio = difflib.SequenceMatcher(
        None, 
        _normalize_for_containment(target_role_title), 
        _normalize_for_containment(latest_title_dict["title"])
    ).ratio()

    if similarity_ratio >= 0.50:
        # Bypass the judge and return deterministic score
        contribution = 1.0 if similarity_ratio == 1.0 else 0.85
        match_level = "direct" if similarity_ratio == 1.0 else "related"
        status = "matched" if contribution >= 1.0 else "related"
        judge_reason = f"Deterministic match: Latest title '{latest_title_dict['title']}' is {similarity_ratio:.0%} similar to required '{target_role_title}'."
        
        best_match = dict(latest_title_dict)
        best_match["jaccard"] = round(_jaccard(target_tokens, _tokenize(latest_title_dict["title"])), 3)
        best_match["match_level"] = match_level
        best_match["judge_reason"] = judge_reason
        
        return {
            "contribution": contribution,
            "best_match": best_match,
            "status": status,
            "all_titles": all_titles,
            "match_level": match_level,
            "judge_reason": judge_reason,
        }

    # FALLBACK: If similarity is < 50%, hand off to the judge to evaluate for synonyms
    evidence_chunks = [
        {"text": t["title"], "source_type": "job_title", "source_label": f"{t['title']} at {t['company']}"}
        for t in titles
    ]
    judgment = judge_fn(target_role_title, evidence_chunks)
    match_level = judgment.get("match", "none")
    judge_reason = judgment.get("reason", "")
    contribution = MATCH_LEVEL_CONTRIBUTION.get(match_level, 0.0)

    # Groundedness check (only runs when the judge claimed a real match)
    groundedness_warning = None
    if contribution > 0.0:
        evidence_titles_text = " | ".join(t["title"] for t in titles)
        groundedness_warning = _reason_falsely_cites_jd_title(judge_reason, evidence_titles_text, target_role_title)
        
        if groundedness_warning:
            best_jaccard = all_titles[0]["jaccard"] if all_titles else 0.0
            if best_jaccard > 0.0:
                match_level = "weak"
                contribution = MATCH_LEVEL_CONTRIBUTION["weak"]
                groundedness_warning += (
                    f" Falling back to a 'weak' match instead of zeroing out entirely, since the "
                    f"best-matching title ('{all_titles[0]['title']}') has independently-measured "
                    f"token overlap (jaccard={best_jaccard}) with the JD title — a real, if modest, "
                    f"signal the judge's fabricated reasoning didn't produce and can't discredit."
                )
            else:
                match_level = "none"
                contribution = 0.0

    if contribution >= 1.0:
        status = "matched"
    elif contribution > 0.0:
        status = "related"
    else:
        status = "missing"

    best_match = dict(all_titles[0])
    best_match["match_level"] = match_level
    best_match["judge_reason"] = judge_reason

    result = {
        "contribution": round(contribution, 3),
        "best_match": best_match,
        "status": status,
        "all_titles": all_titles,
        "match_level": match_level,
        "judge_reason": judge_reason,
    }
    if groundedness_warning:
        result["groundedness_warning"] = groundedness_warning
    return result
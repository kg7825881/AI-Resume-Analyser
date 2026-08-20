"""
retrieval.py — BM25-based evidence retrieval for candidate resumes.

Given one JD requirement (e.g. "Production ETL/ELT pipelines"), retrieves the
most relevant chunks of a candidate's resume — experience descriptions,
project descriptions, and their technologies_used, plus the raw skills list —
using BM25 lexical ranking (rank_bm25's BM25Okapi).

This module is retrieval ONLY. It does not decide whether the evidence it
finds actually supports the requirement — that judgment belongs to the LLM
judge (see judge.py). BM25's job is narrower and cheaper: cut a candidate's
whole resume down to the handful of chunks worth asking the judge about, and
do it in a way that's transparent — every retrieval has an inspectable score
and a source quote, unlike a bare cosine-similarity number.

Why BM25 and not embeddings for this stage: it's a first-stage retrieval
mechanism precisely because it's cheap, fast, and needs no calibration
(compare calibrate_embeddings.py, which exists ONLY because embedding cosine
similarity needs per-model threshold tuning). BM25 is lexical, not semantic —
it will miss e.g. "data pipeline development" (JD) vs "Built ETL workflows
using Apache Airflow" (resume) if the vocabulary doesn't overlap. That's
expected and fine here: a retrieval miss just means an empty evidence list,
which judge.py already treats as "no evidence" — a safe, inspectable default,
not a wrong answer smuggled in from an unrelated match.
"""

import re

from rank_bm25 import BM25Plus

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


class EvidenceChunk:
    """One retrievable unit of a candidate's resume."""

    __slots__ = ("text", "source_type", "source_label")

    def __init__(self, text: str, source_type: str, source_label: str):
        self.text = text
        self.source_type = source_type    # "experience" | "project" | "skills"
        self.source_label = source_label  # e.g. "Data Engineer at Foo Corp"

    def to_dict(self) -> dict:
        return {"text": self.text, "source_type": self.source_type, "source_label": self.source_label}


def build_evidence_chunks(candidate_data: dict) -> list[EvidenceChunk]:
    """
    Builds the BM25 corpus for one candidate from their structured resume
    record (the dict shape produced by extractor.py's ingest_resume). One
    chunk per experience entry, one per project, plus a single chunk for the
    raw skills list — kept coarse (whole-role/whole-project) rather than
    sentence-level, since the judge needs enough surrounding context to
    actually assess a requirement, not an isolated fragment.
    """
    chunks: list[EvidenceChunk] = []

    for entry in candidate_data.get("experience", []) or []:
        title = entry.get("title") or "unknown title"
        company = entry.get("company") or "unknown company"
        label = f"{title} at {company}"
        parts = []
        desc = entry.get("description")
        if desc:
            parts.append(desc)
        domain = entry.get("domain")
        if domain:
            # Added so industry/domain requirements (e.g. "Fintech", "Healthcare")
            # have something to retrieve against — Stage 2 already infers this per
            # role, but until now it was extracted and stored, never fed into the
            # BM25 corpus, so an industry_keywords requirement always retrieved
            # nothing for it specifically (it could only piggyback on skills/desc
            # text that happened to share a word with the industry name).
            parts.append(f"Industry/domain: {domain}")
        techs = entry.get("technologies_used") or []
        if techs:
            parts.append("Tools/technologies used: " + ", ".join(techs))
        text = " ".join(parts).strip()
        if text:
            chunks.append(EvidenceChunk(text, "experience", label))

    for proj in candidate_data.get("projects", []) or []:
        label = proj.get("title") or "unknown project"
        parts = []
        desc = proj.get("description")
        if desc:
            parts.append(desc)
        techs = proj.get("technologies_used") or []
        if techs:
            parts.append("Tools/technologies used: " + ", ".join(techs))
        text = " ".join(parts).strip()
        if text:
            chunks.append(EvidenceChunk(text, "project", label))

    # The skills list itself is also retrievable evidence — a required skill
    # can legitimately be satisfied by "candidate literally lists this" even
    # when no experience/project bullet happens to mention it. score_single_skill
    # in matcher.py already checks the skills list exactly/word-boundary first;
    # this chunk exists for the judge's looser matching (e.g. required "cloud
    # data warehousing" against a skills list that contains "Snowflake").
    skills = candidate_data.get("skills_all_sources") or candidate_data.get("skills") or []
    if skills:
        chunks.append(EvidenceChunk(", ".join(skills), "skills", "Skills section"))

    return chunks


class CandidateEvidenceIndex:
    """
    One BM25 index built ONCE per candidate and reused across every
    requirement scored against that candidate in a single scoring run — a JD
    can easily have 15-20 mandatory + preferred requirements, so rebuilding
    the index per-requirement would be wasted repeated work over the same
    fixed corpus.
    """

    def __init__(self, candidate_data: dict):
        self.chunks = build_evidence_chunks(candidate_data)
        self._corpus_tokens = [_tokenize(c.text) for c in self.chunks]
        # BM25Plus, not the classic BM25Okapi. This corpus is small by construction —
        # a handful of chunks per candidate, not thousands of documents — and Okapi's
        # classic IDF, log((N - n + 0.5) / (n + 0.5)), collapses to exactly 0 whenever a
        # term appears in precisely half the chunks (e.g. N=2, n=1: log(1.5/1.5) = 0),
        # silently zeroing out the score for a term that DOES appear in the evidence.
        # Confirmed on this exact corpus shape during testing. BM25Plus's IDF,
        # log((N + 1) / n), stays positive for any n <= N, which is the property this
        # small-corpus retrieval actually needs.
        self._bm25 = BM25Plus(self._corpus_tokens) if self._corpus_tokens else None

    def retrieve(self, requirement: str, top_k: int = 3, min_score: float = 0.0) -> list[dict]:
        """
        Returns up to top_k evidence chunks for `requirement`, each as a dict
        (see EvidenceChunk.to_dict) plus a "bm25_score" field, sorted by score
        descending. Returns [] if the candidate has no evidence chunks at all,
        the requirement tokenizes to nothing, or nothing scores above
        min_score (i.e. zero lexical overlap with anything on the resume).
        """
        if not self._bm25 or not self.chunks:
            return []

        query_tokens = _tokenize(requirement)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(zip(self.chunks, scores), key=lambda pair: pair[1], reverse=True)

        results = []
        for chunk, score in ranked[:top_k]:
            if score <= min_score:
                continue
            d = chunk.to_dict()
            d["bm25_score"] = round(float(score), 3)
            results.append(d)
        return results
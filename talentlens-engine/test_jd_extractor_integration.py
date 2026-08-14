"""
test_jd_extractor_integration.py — end-to-end test that extract_structured_jd() applies
the jd_skills.py safety net, using the model's ACTUAL observed (unsplit) output for the
Data Engineer — AI Data Platform JD as the mock response — i.e. reproducing exactly what
was seen in production, not an idealized already-correct response.
"""

import json
from unittest.mock import patch, MagicMock

import jd_extractor


# Exactly what the model actually returned in production (per the screenshot) — several
# mandatory_skills items un-split, confirming the LLM's own ATOMIC SKILL EXTRACTION
# instructions aren't reliably followed.
_ACTUAL_UNSPLIT_LLM_OUTPUT = {
    "role_title": "Data Engineer — AI Data Platform",
    "department": "Data Platform",
    "mandatory_skills": [
        "3-5 years of experience in data engineering, analytics engineering, or data platform roles.",
        "Strong SQL and good Python skills.",
        "Experience building ETL or ELT pipelines in production.",
        "Strong understanding of data modeling, schema design, and transformation logic.",
        "Experience working with structured and semi-structured data.",
        "Familiarity with data warehouse or lakehouse design concepts.",
        "Hands-on experience with data validation, deduplication, consistency checks, and data quality control",
        "Experience building data quality checks, monitoring, and alerting pipelines.",
        "Ability to prepare clean, reliable datasets for ML, RAG, evaluation, and production use.",
    ],
    "preferred_technical_skills": [
        "Experience with Spark, dbt, Airflow, Dagster, Kafka, or similar tools.",
        "Exposure to document processing, OCR, or unstructured data pipelines.",
        "Familiarity with vector databases, embedding pipelines, or retrieval datasets.",
        "Experience supporting ML or GenAI teams with training, evaluation, or feedback data.",
    ],
    "soft_preferred_skills": [
        "Comfort working with document-heavy, workflow-heavy business data.",
        "Ability to collaborate closely with AI engineers, backend engineers, and product teams.",
    ],
    "min_years_experience": 3,
    "education_requirements": [],
    "relevant_certifications": [],
    "responsibilities": [
        "Design and maintain ETL/ELT pipelines feeding the AI data platform.",
    ],
}


def _ollama_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.__getitem__.side_effect = lambda k: {"message": {"content": content}}[k]
    return resp


@patch("jd_extractor.ollama.chat")
def test_extract_structured_jd_atomizes_real_production_output(mock_chat):
    mock_chat.return_value = _ollama_response(json.dumps(_ACTUAL_UNSPLIT_LLM_OUTPUT))

    result = jd_extractor.extract_structured_jd("(irrelevant — response is mocked)")

    # mandatory_skills: years clause dropped, SQL/Python split, etc.
    assert "SQL" in result["mandatory_skills"]
    assert "Python" in result["mandatory_skills"]
    assert not any("years of experience" in s.lower() for s in result["mandatory_skills"])
    assert not any(s.strip().lower() == "python skills" for s in result["mandatory_skills"])

    # preferred_technical_skills: "similar tools" hedge dropped
    assert "Spark" in result["preferred_technical_skills"]
    assert not any("similar" in s.lower() for s in result["preferred_technical_skills"])

    # soft_preferred_skills: collaboration-with-people clause dropped entirely,
    # adjective-pair clause NOT wrongly split into two fake skills
    assert not any("collaborate" in s.lower() for s in result["soft_preferred_skills"])
    assert not any("ai engineers" in s.lower() for s in result["soft_preferred_skills"])
    assert "document-heavy, workflow-heavy business data" in result["soft_preferred_skills"]

    # responsibilities untouched (full sentence preserved, not atomized)
    assert result["responsibilities"] == _ACTUAL_UNSPLIT_LLM_OUTPUT["responsibilities"]

    # Other fields pass through unchanged
    assert result["role_title"] == "Data Engineer — AI Data Platform"
    assert result["min_years_experience"] == 3


@patch("jd_extractor.ollama.chat")
def test_extract_structured_jd_handles_already_atomic_llm_output(mock_chat):
    """If the LLM DOES follow its own instructions correctly, the safety net must be a
    no-op — not damage already-good output."""
    already_good = dict(_ACTUAL_UNSPLIT_LLM_OUTPUT)
    already_good["mandatory_skills"] = ["SQL", "Python", "ETL", "Data Modeling"]
    mock_chat.return_value = _ollama_response(json.dumps(already_good))

    result = jd_extractor.extract_structured_jd("(irrelevant)")
    assert result["mandatory_skills"] == ["SQL", "Python", "ETL", "Data Modeling"]


@patch("jd_extractor.ollama.chat")
def test_extract_structured_jd_handles_malformed_json_unchanged(mock_chat):
    """Existing incomplete-JSON error path must still work — the new code runs AFTER
    the json.loads try/except, not instead of it."""
    mock_chat.return_value = _ollama_response("{not valid json")

    result = jd_extractor.extract_structured_jd("(irrelevant)")
    assert result["error"] == "Incomplete JSON"
    assert "raw_output" in result


if __name__ == "__main__":
    import sys
    try:
        import pytest
        sys.exit(pytest.main([__file__, "-v"]))
    except ImportError:
        tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
        failures = 0
        for test in tests:
            try:
                test()
                print(f"PASS: {test.__name__}")
            except Exception as e:
                failures += 1
                print(f"FAIL: {test.__name__}: {e}")
        print("\nALL PASSED" if failures == 0 else f"\n{failures} FAILURE(S)")
        sys.exit(1 if failures else 0)

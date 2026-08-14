"""
test_jd_skills.py — regression tests for jd_skills.py, built from the 15 real
mandatory_skills lines shown un-split in production (Data Engineer — AI Data Platform
JD, screenshot 2026-08-14). Every case here is a REAL line, not a synthetic example.

Run: pytest test_jd_skills.py -v
"""

from jd_skills import atomize_skill_line, atomize_skill_list


# --- Cases that must split cleanly into atomic skills ---

def test_strips_years_of_experience_clause_entirely():
    result = atomize_skill_line(
        "3-5 years of experience in data engineering, analytics engineering, or data platform roles."
    )
    assert result == [], "years-of-experience clause should be dropped (redundant with min_years_experience)"


def test_splits_strong_x_and_good_y_pattern():
    result = atomize_skill_line("Strong SQL and good Python skills.")
    assert result == ["SQL", "Python"], f"got {result!r}"


def test_splits_three_way_and_list():
    result = atomize_skill_line(
        "Strong understanding of data modeling, schema design, and transformation logic."
    )
    assert result == ["data modeling", "schema design", "transformation logic"]


def test_splits_four_way_and_list():
    result = atomize_skill_line(
        "Hands-on experience with data validation, deduplication, consistency checks, "
        "and data quality control"
    )
    assert result == ["data validation", "deduplication", "consistency checks", "data quality control"]


def test_splits_three_way_and_list_building_pipelines():
    result = atomize_skill_line(
        "Experience building data quality checks, monitoring, and alerting pipelines."
    )
    assert result == ["data quality checks", "monitoring", "alerting pipelines"]


def test_drops_similar_tools_hedge_after_or_list():
    result = atomize_skill_line("Experience with Spark, dbt, Airflow, Dagster, Kafka, or similar tools.")
    assert result == ["Spark", "dbt", "Airflow", "Dagster", "Kafka"]
    assert "similar tools" not in result
    assert not any("similar" in item.lower() for item in result)


def test_splits_or_list_document_processing():
    result = atomize_skill_line("Exposure to document processing, OCR, or unstructured data pipelines.")
    assert result == ["document processing", "OCR", "unstructured data pipelines"]


def test_splits_or_list_vector_databases():
    result = atomize_skill_line(
        "Familiarity with vector databases, embedding pipelines, or retrieval datasets."
    )
    assert result == ["vector databases", "embedding pipelines", "retrieval datasets"]


def test_strips_experience_working_with_prefix():
    result = atomize_skill_line("Experience working with structured and semi-structured data.")
    assert result == ["structured", "semi-structured data"]


# --- Cases that must NOT be force-split (nested structure / shared trailing phrase) ---
# These are kept as ONE item rather than fragmented — a single non-atomic skill just
# semantic-matches a bit more weakly; a fabricated fragment would never match anything.

def test_does_not_fragment_nested_datasets_for_clause():
    result = atomize_skill_line(
        "Ability to prepare clean, reliable datasets for ML, RAG, evaluation, and production use."
    )
    assert len(result) == 1, f"should stay whole, got {result!r}"
    # The specific garbage fragments this must NOT produce:
    assert not any("reliable datasets for" == item.lower()[:len("reliable datasets for")] for item in result)
    joined = " ".join(result).lower()
    assert "clean" not in [i.strip().lower() for i in result]  # "clean" alone would be a fabricated fragment


def test_does_not_fragment_shared_trailing_phrase_ml_genai_teams():
    result = atomize_skill_line(
        "Experience supporting ML or GenAI teams with training, evaluation, or feedback data."
    )
    assert len(result) == 1, f"should stay whole, got {result!r}"


def test_does_not_split_etl_elt_shared_trailing_phrase():
    result = atomize_skill_line("Experience building ETL or ELT pipelines in production.")
    assert len(result) == 1, f"should stay whole (known limitation), got {result!r}"


# --- Cases that must be filtered out entirely (not skills at all) ---

def test_drops_collaboration_with_people_clause_entirely():
    result = atomize_skill_line(
        "Ability to collaborate closely with AI engineers, backend engineers, and product teams."
    )
    assert result == [], f"people/team collaboration clause should be dropped entirely, got {result!r}"


# --- The critical negative case: must NOT over-split adjective pairs into fake skills ---

def test_does_not_split_hyphenated_adjective_pair_with_no_conjunction():
    """This is the inverse failure mode from the nested-clause cases above: a comma
    with NO and/or conjunction is stacked adjectives on one noun, not an enumeration."""
    result = atomize_skill_line("Comfort working with document-heavy, workflow-heavy business data.")
    assert result == ["document-heavy, workflow-heavy business data"], f"got {result!r}"
    assert result != ["document-heavy", "workflow-heavy business data"], (
        "must not have split into two fake separate skills"
    )


# --- List-level behavior ---

def test_atomize_skill_list_flattens_and_dedupes():
    raw = [
        "Strong SQL and good Python skills.",
        "Experience with Python and SQL databases.",  # deliberately overlaps SQL/Python
    ]
    result = atomize_skill_list(raw)
    lowered = [r.lower() for r in result]
    assert lowered.count("sql") == 1, f"SQL should be deduped, got {result!r}"
    assert lowered.count("python") == 1, f"Python should be deduped, got {result!r}"


def test_atomize_skill_list_handles_empty_and_none():
    assert atomize_skill_list([]) == []
    assert atomize_skill_list(None) == []
    assert atomize_skill_list(["", None, "   "]) == []


def test_full_15_line_jd_produces_no_garbage_fragments():
    """End-to-end sanity check across all 15 real lines: no result item should be a
    single meaningless leftover word (a common signature of a bad split)."""
    lines = [
        "3-5 years of experience in data engineering, analytics engineering, or data platform roles.",
        "Strong SQL and good Python skills.",
        "Experience building ETL or ELT pipelines in production.",
        "Strong understanding of data modeling, schema design, and transformation logic.",
        "Experience working with structured and semi-structured data.",
        "Familiarity with data warehouse or lakehouse design concepts.",
        "Hands-on experience with data validation, deduplication, consistency checks, and data quality control",
        "Experience building data quality checks, monitoring, and alerting pipelines.",
        "Ability to prepare clean, reliable datasets for ML, RAG, evaluation, and production use.",
        "Experience with Spark, dbt, Airflow, Dagster, Kafka, or similar tools.",
        "Exposure to document processing, OCR, or unstructured data pipelines.",
        "Familiarity with vector databases, embedding pipelines, or retrieval datasets.",
        "Experience supporting ML or GenAI teams with training, evaluation, or feedback data.",
        "Comfort working with document-heavy, workflow-heavy business data.",
        "Ability to collaborate closely with AI engineers, backend engineers, and product teams.",
    ]
    result = atomize_skill_list(lines)
    known_garbage_fragments = {"clean", "reliable", "ability", "experience", "strong", "good", "and", "or"}
    lowered = {r.strip().lower() for r in result}
    overlap = lowered & known_garbage_fragments
    assert not overlap, f"garbage fragments leaked through: {overlap} (full result: {result})"


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

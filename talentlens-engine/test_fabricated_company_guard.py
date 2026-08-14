"""
test_fabricated_company_guard.py

Reproduces the bug confirmed on Ulka Jambhulkar's resume: _list_companies_present
mistook CONTACT-sidebar labels ("Phone", "Email", "Address") and a project subheading
("ML-Based Fuel Delivery Automation & Optimization" — actually inside the Foster
Technologies India block) for real employers, and _reextract_single_job fabricated
full experience entries for them because it only checked entry.get("company") was
truthy, never that it matched what was actually requested.

Proves the two guardrails close this against the REAL extractor.py functions:
  1. _looks_like_non_company_label — denylist, skips known contact/section labels
     before ever calling the LLM.
  2. _reextract_single_job's company-match check — rejects an entry if the model
     returns one for a DIFFERENT company than requested (the project-subheading case,
     which the denylist alone doesn't catch).

Run: pytest test_fabricated_company_guard.py -v
"""

import json
from unittest.mock import patch, MagicMock

import extractor

RESUME_TEXT = """
ULKA JAMBHULKAR
CONTACT
8510802840
Phone
ulkajambhulkar280@gmail.com
Email
Amrapali Dream Valley, Gautam Buddha Nagar, Techzone -4, Greater Noida
Address

WORK EXPERIENCE
Chetu, Inc. OCT 2023 - PRESENT
Multi-Agent Travel Itinerary AI Planner
... (abbreviated) ...

Foster Technologies India Dec 2020 - Oct 2023
ML-Based Fuel Delivery Automation & Optimization
Integrated Python and .NET APIs to fetch and preprocess real-time tank, vendor,
vehicle, and driver data using Pandas...
"""

ALREADY_EXTRACTED = [
    {"company": "Chetu, Inc", "technologies_used": ["LangGraph", "OpenAI"]},
    {"company": "Foster Technologies India", "technologies_used": ["Python", "OpenCV"]},
]


def _ollama_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.__getitem__.side_effect = lambda k: {"message": {"content": content}}[k]
    return resp


def _fake_ollama_chat(*, model, messages, format=None, options=None):
    system_prompt = messages[0]["content"]

    if "List every company name" in system_prompt:
        # Reproduces the confirmed bug: contact labels AND a project subheading get
        # reported as if they were companies.
        return _ollama_response(
            "Chetu, Inc\nFoster Technologies India\n"
            "ML-Based Fuel Delivery Automation & Optimization\nPhone\nEmail\nAddress"
        )

    if "extract ONLY the work experience entry" in system_prompt:
        if "\"ML-Based Fuel Delivery Automation & Optimization\"" in system_prompt:
            # Reproduces the confirmed bug: asked for a nonexistent "company" (really a
            # project subheading), the model doesn't fail — it hallucinates an entry
            # under a DIFFERENT, already-real company instead of the empty-string
            # fields the improved prompt asks for.
            fabricated = {
                "title": "ML-Based Fuel Delivery Automation & Optimization",
                "company": "Foster Technologies India",  # <-- mismatch, the bug
                "start_date_raw": "Dec 2020",
                "end_date_raw": "Oct 2023",
                "domain": "Machine Learning",
                "description": "...",
                "technologies_used": ["Python", ".NET", "Pandas"],
            }
            return _ollama_response(json.dumps(fabricated))

        # Any other company_name reaching this branch (e.g. Phone/Email/Address) means
        # the denylist failed to skip it — that's a test failure, not something to mock
        # a plausible response for.
        raise AssertionError(
            f"_reextract_single_job was called for a company that should have been "
            f"denylisted before reaching the LLM. Prompt: {system_prompt[:200]!r}"
        )

    raise AssertionError(f"Unexpected system prompt routed to mock: {system_prompt[:80]!r}")


def test_looks_like_non_company_label_catches_contact_fields():
    for label in ["Phone", "Email", "Address", "phone", "  Email  ", "CONTACT", "LinkedIn"]:
        assert extractor._looks_like_non_company_label(label), f"{label!r} should be flagged"


def test_looks_like_non_company_label_does_not_flag_real_companies():
    for name in ["Chetu, Inc", "Foster Technologies India", "Phone Inc", "Email Corp", "Google"]:
        assert not extractor._looks_like_non_company_label(name), f"{name!r} should NOT be flagged"


def test_companies_match_basic_cases():
    assert extractor._companies_match("Foster Technologies India", "Foster Technologies India")
    assert extractor._companies_match("Foster Technologies", "Foster Technologies India")  # substring
    assert not extractor._companies_match("Foster Technologies India", "Chetu, Inc")
    assert not extractor._companies_match(
        "Foster Technologies India", "ML-Based Fuel Delivery Automation & Optimization"
    )


@patch("extractor.ollama.chat", side_effect=_fake_ollama_chat)
def test_reextract_single_job_rejects_mismatched_company(mock_chat):
    """Core regression test for guardrail #2: asked to recover a project subheading
    mistaken for a company, the function must reject the model's fabricated entry
    (returned under a different, already-real company) rather than accepting it."""
    result = extractor._reextract_single_job(
        RESUME_TEXT, "ML-Based Fuel Delivery Automation & Optimization"
    )
    assert result is None, "Mismatched-company entry should have been rejected, not accepted"


@patch("extractor.ollama.chat", side_effect=_fake_ollama_chat)
def test_fill_missing_experience_does_not_fabricate_or_duplicate(mock_chat):
    """End-to-end regression test reproducing the full Ulka Jambhulkar scenario:
    _list_companies_present returns 3 contact labels + 1 project-subheading false
    positive alongside the 2 real companies (both already extracted). After the fix,
    structured['experience'] must be UNCHANGED (no fabricated Phone/Email/Address
    entries, no duplicate Foster Technologies India entry)."""
    structured = {"experience": [dict(e) for e in ALREADY_EXTRACTED]}

    warnings = extractor._fill_missing_experience(RESUME_TEXT, structured)

    # No new entries were fabricated or duplicated
    assert len(structured["experience"]) == 2
    companies_in_result = [e["company"] for e in structured["experience"]]
    assert companies_in_result == ["Chetu, Inc", "Foster Technologies India"]

    # Contact labels produced NO warnings at all (silently skipped)
    assert not any("Phone" in w for w in warnings)
    assert not any("Email" in w for w in warnings)
    assert not any("Address" in w for w in warnings)

    # The project-subheading false positive DID produce exactly one warning, and it's
    # the informative "could not confirm as a real employer" warning, not the old
    # (misleading, in this case) "total_years_experience is likely understated" wording
    ml_warnings = [w for w in warnings if "ML-Based Fuel Delivery Automation & Optimization" in w]
    assert len(ml_warnings) == 1
    assert "could not confirm it as a real employer" in ml_warnings[0]
    assert "project name or section label" in ml_warnings[0]


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

"""
experience.py — deterministic total-experience calculation (Stage 3).

The LLM (Stage 2) never computes a duration or a total. It only copies each role's
start/end dates exactly as written in the resume text. This module turns those raw
strings into a single total-years figure, entirely in code, using a date-range UNION —
not a summation.

Why union and not summation: summing each role's duration double-counts continuous
back-to-back roles (role A ends where role B starts) whenever anything else in the
pipeline also derives a total from the earliest-to-latest span — which is exactly the
~2x inflation bug this replaces (12.0 yrs / 13.0 yrs / 10.0 yrs shown for candidates
whose resumes state "5+ years" and whose own date ranges confirm ~5-6 years). Union
also correctly handles genuinely overlapping roles (e.g. freelance work alongside a
full-time job) by not counting the overlapping days twice.
"""

import re
from datetime import date, datetime

from dateutil import parser as dateutil_parser

PRESENT_KEYWORDS = {"present", "current", "currently", "till date", "to date", "till now", "ongoing", "now", "date"}

_MONTH = r"[A-Za-z]{3,9}"
# Resumes commonly abbreviate years to 2 digits ("Mar 24", "Feb 23"). dateutil's default
# parser reads "Mar 24" as day=24 (a valid day-of-month) with the year left at whatever
# default is passed in — NOT as "March 2024". That silently produced a year-1900 date,
# which is exactly the kind of wrong-but-plausible-looking value this module exists to
# prevent. These patterns catch that shape and expand it to a real 4-digit year first.
_TWO_DIGIT_YEAR_PATTERNS = [
    re.compile(rf"^({_MONTH})[\s\-/]+(\d{{2}})$"),   # "Mar 24", "Mar-24", "Mar/24"
    re.compile(rf"^(\d{{2}})[\s\-/]+({_MONTH})$"),   # "24 Mar", "24-Mar"
]


def _expand_two_digit_year(raw: str) -> str:
    """Rewrites e.g. 'Mar 24' -> 'Mar 2024' so dateutil can't mistake the year for a day.
    Leaves anything else (full 4-digit years, bare years, 'Present', etc.) untouched."""
    s = raw.strip()
    for idx, pattern in enumerate(_TWO_DIGIT_YEAR_PATTERNS):
        m = pattern.match(s)
        if m:
            month, yy = m.groups() if idx == 0 else (m.group(2), m.group(1))
            return f"{month} {2000 + int(yy)}"
    return raw


def _parse_date(raw: str):
    """
    Parses a single date string exactly as extracted from the resume (e.g. "Oct 2023",
    "Mar 2022", "Mar 24", "2019", "Present"). Returns a date, or None if it can't be
    parsed — unparseable entries are dropped (with a warning) upstream, never silently
    guessed, since a wrong guess here would reintroduce the exact problem this module
    exists to fix.
    """
    if not raw or not isinstance(raw, str) or not raw.strip():
        return None

    normalized = raw.strip().lower()
    if normalized in PRESENT_KEYWORDS:
        return date.today()

    raw = _expand_two_digit_year(raw)

    try:
        # default=datetime(1900, 1, 1) means: any date component NOT present in `raw`
        # (e.g. day is never in a resume date, sometimes month isn't either) falls back
        # to 1 rather than to today's day/month — so "2019" parses as 2019-01-01, not
        # "today's month/day in 2019".
        parsed = dateutil_parser.parse(raw, default=datetime(1900, 1, 1), fuzzy=True)
        return parsed.date()
    except (ValueError, OverflowError, TypeError):
        return None


def compute_total_years(experience_entries: list) -> tuple:
    """
    experience_entries: the "experience" list from the Stage 2 JSON, each entry expected
    to have "start_date_raw" and "end_date_raw" (plus title/company/etc, which are ignored
    here except for warning messages).

    Returns (total_years: float, warnings: list[str]).

    Ranges are merged (union) after sorting by start date — overlapping or back-to-back
    ranges are combined into one span before the total is summed, so continuous roles at
    different companies count once, not once-per-role plus once-for-the-whole-span.

    Safety net: an entry with a valid start date but an unparseable/empty end date is
    normally excluded (never silently guessed). The one exception is the single entry
    whose start date is the LATEST among all entries — that's almost always the current
    role, and resumes frequently drop "Present" from the raw text in ways the extraction
    step can miss. That one entry is treated as ongoing (end = today), flagged with a
    clear warning so it stays reviewable rather than silently trusted.
    """
    warnings = []
    parsed = []

    for entry in experience_entries or []:
        company = entry.get("company") or "unknown company"
        title = entry.get("title") or "unknown title"
        label = f"{title} at {company}"

        start_raw = entry.get("start_date_raw", "")
        end_raw = entry.get("end_date_raw", "")

        start = _parse_date(start_raw)
        if start is None:
            warnings.append(f"Could not parse start date '{start_raw}' for '{label}' — excluded from experience total.")
            continue

        end = _parse_date(end_raw)
        parsed.append({"start": start, "end": end, "label": label, "end_raw": end_raw})

    if not parsed:
        return 0.0, warnings

    latest_start = max(p["start"] for p in parsed)
    missing_end_at_latest_start = [p for p in parsed if p["end"] is None and p["start"] == latest_start]
    # Only auto-assume "ongoing" if there's exactly one entry sitting at the latest start
    # date with no end — if two roles both start latest with no end, that's ambiguous
    # enough that guessing could just as easily introduce a new error, so it's left to
    # fall through to the normal excluded-with-warning path instead.
    if len(missing_end_at_latest_start) == 1:
        p = missing_end_at_latest_start[0]
        p["end"] = date.today()
        warnings.append(
            f"End date missing for '{p['label']}' — it has the most recent start date of all "
            f"roles, so it was assumed to still be ongoing (Present) for the total. Verify manually."
        )

    ranges = []
    for p in parsed:
        if p["end"] is None:
            warnings.append(f"Could not parse end date '{p['end_raw']}' for '{p['label']}' — excluded from experience total.")
            continue
        if p["end"] < p["start"]:
            warnings.append(f"End date ({p['end_raw']}) is before start date for '{p['label']}' — excluded from experience total.")
            continue
        ranges.append((p["start"], min(p["end"], date.today())))  # never count future-dated experience

    if not ranges:
        return 0.0, warnings

    ranges.sort(key=lambda r: r[0])
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    total_days = sum((end - start).days for start, end in merged)
    total_years = round(total_days / 365.25, 1)
    return total_years, warnings
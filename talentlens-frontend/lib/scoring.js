// lib/scoring.js
// Presentation helpers that derive everything from the real score records
// returned by GET /results/{role_id} — nothing here invents data.

// Max points per category, mirroring scorer.py's WEIGHTS * 100.
export const CATEGORY_MAX = {
  mandatory_skills: 50,
  relevant_experience: 30,
  education: 10,
  preferred_skills: 10,
};

export const CATEGORY_LABELS = {
  mandatory_skills: "Mandatory skills",
  relevant_experience: "Experience",
  education: "Education",
  preferred_skills: "Preferred skills",
};

// Shared 3-state status → accent color, used by the Evidence panel's sub-sections.
// "weak_match" covers a semantic hit too soft to clear the hard-gate threshold
// distinct from a confident match.
export const STATUS_COLOR = {
  matched: "var(--g)",
  weak_match: "#f5bd62",
  missing: "var(--r)",
};

export const STATUS_ICON = {
  matched: "✓",
  weak_match: "~",
  missing: "×",
};

export function statusFor(finalScore) {
  if (finalScore >= 90) return { key: "excellent", label: "Excellent Match" };
  if (finalScore >= 80) return { key: "strong", label: "Strong Match" };
  if (finalScore >= 60) return { key: "review", label: "Review" };
  return { key: "low", label: "Below threshold" };
}

export function initials(name) {
  if (!name) return "?";
  const parts = name.trim().split(/\s+/);
  return parts
    .slice(0, 2)
    .map((p) => p[0]?.toUpperCase() || "")
    .join("");
}

/** True when every mandatory skill was matched (nothing in `missing`). */
export function passesMandatory(record) {
  const missing = record.category_scores?.mandatory_skills?.missing || [];
  return missing.length === 0;
}

/** Sort every scored record (ranked + excluded) by final_score desc, and attach a 1-based rank. */
export function rankAll(ranked, excluded) {
  const all = [...ranked, ...excluded].sort((a, b) => b.final_score - a.final_score);
  return all.map((r, i) => ({ ...r, rank: i + 1 }));
}

/** Top few matched skills across categories, for a compact "top evidence" table cell. */
export function topEvidence(record, limit = 3) {
  const cs = record.category_scores || {};
  const pool = [
    ...(cs.mandatory_skills?.matched || []),
    ...(cs.preferred_skills?.matched || []),
  ];
  return pool.slice(0, limit).join(" · ");
}

/**
 * Every matched skill across all skill-based categories (mandatory, preferred),
 * deduplicated. Used for the "Skills matched" comparison field.
 */
export function allMatchedSkills(record) {
  const cs = record.category_scores || {};
  const pool = [
    ...(cs.mandatory_skills?.matched || []),
    ...(cs.preferred_skills?.matched || []),
  ];
  return [...new Set(pool)];
}

/** The candidate's most recent role (first entry in their experience list), or null. */
export function mostRecentRole(record) {
  const experience = record.experience || [];
  return experience.length > 0 ? experience[0] : null;
}

/** Flat list of { skill, matched: bool, category } across every skill-based category, for evidence panels.
 * Kept for any existing callers relying on the flat shape — evidenceSections() below is the
 * richer replacement used by the Candidate Detail evidence panel. */
export function evidenceList(record) {
  const cs = record.category_scores || {};
  const cats = [
    ["mandatory_skills", "Mandatory"],
    ["preferred_skills", "Preferred"],
  ];
  const items = [];
  for (const [key, label] of cats) {
    for (const skill of cs[key]?.matched || []) {
      items.push({ skill, matched: true, category: label });
    }
    for (const skill of cs[key]?.missing || []) {
      items.push({ skill, matched: false, category: label });
    }
  }
  return items;
}

function mapSkillRow(r) {
  return {
    label: r.skill,
    status: r.status, // "matched" | "weak_match" | "missing"
    detail: r.matched_against ? `matched against: ${r.matched_against}` : null,
    matchType: r.match_type, // "exact" | "semantic" | "none"
  };
}

/**
 * Structured evidence for the Candidate Detail page's Evidence panel, built from
 * record.evidence (scorer.py's per-category itemized rows) rather than the flatter
 * category_scores.matched/missing lists — this is what lets the panel show *why* a
 * skill matched (exact vs. semantic, and against which candidate skill), plus the
 * experience/projects/education rows that category_scores alone never carried.
 *
 * Shape:
 * {
 *   skillSections: [{ key, label, items: [{ label, status, detail, matchType }] }],
 *   experience: { years: { total_years_experience, min_years_required, status } | null,
 *                 roles: [{ label, detail, status, similarity }] },
 *   education: [{ label, status }],
 *   additionalSkills: string[],
 * }
 */
export function evidenceSections(record) {
  const ev = record.evidence || {};

  const skillSections = [
    { key: "mandatory_skills", label: "Mandatory skills", items: (ev.mandatory_skills || []).map(mapSkillRow) },
    { key: "preferred_skills", label: "Preferred skills", items: (ev.preferred_skills || []).map(mapSkillRow) },
  ];

  const experienceEv = ev.experience || {};
  const experienceRoles = (experienceEv.roles || []).map((r) => ({
    label: [r.title, r.company].filter(Boolean).join(" @ ") || "Role",
    detail: r.duration,
    status: r.status,
    similarity: r.relevance_similarity,
  }));

  const educationItems = (ev.education || []).map((e) => ({
    label: [e.required_degree_level, e.required_field].filter(Boolean).join(" in ") || "Requirement",
    status: e.status,
  }));

  return {
    skillSections,
    experience: { years: experienceEv.years || null, roles: experienceRoles },
    education: educationItems,
    additionalSkills: ev.additional_candidate_skills || [],
  };
}

export function formatEducation(education) {
  if (!education || education.length === 0) return "—";
  return education.map((e) => [e.degree_level, e.field].filter(Boolean).join(" — ")).join(", ");
}

/** Grounded, template-built explanation of a candidate's rank — built only from fields the API returned. */
export function explainRank(record, roleTitle) {
  const cs = record.category_scores || {};
  const missingMandatory = cs.mandatory_skills?.missing || [];
  const sentences = [];

  sentences.push(
    `${record.candidate_name || "This candidate"} ranks #${record.rank} for ${roleTitle || "this role"} with a Job Fit score of ${record.final_score}%.`
  );

  if (record.hard_gate_failed) {
    sentences.push(record.hard_gate_reason || "Failed the mandatory-skills hard gate.");
  } else if (missingMandatory.length === 0) {
    sentences.push("All mandatory requirements are satisfied.");
  } else {
    sentences.push(`Missing ${missingMandatory.length} mandatory skill(s): ${missingMandatory.join(", ")}.`);
  }

  if (cs.relevant_experience?.notes) {
    sentences.push(cs.relevant_experience.notes.replace(/^./, (c) => c.toUpperCase()) + ".");
  }

  const strongest = Object.entries(cs)
    .map(([key, v]) => ({ key, fraction: (v.score || 0) / (CATEGORY_MAX[key] || 1) }))
    .sort((a, b) => b.fraction - a.fraction)[0];
  if (strongest) {
    sentences.push(`Strongest category: ${CATEGORY_LABELS[strongest.key] || strongest.key}.`);
  }

  return sentences.join(" ");
}
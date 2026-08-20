// lib/scoring.js
// Presentation helpers that derive everything from the real score records
// returned by GET /results/{role_id} — mirrors backend weights and categories.

// Max points per category mirroring scorer.py's WEIGHTS * 100
export const CATEGORY_MAX = {
  mandatory_skills: 25,
  relevant_experience: 25,
  education: 20,
  industry_keywords: 10,
  soft_skills: 10,
  job_title_match: 5,
  preferred_skills: 5,
};

export const CATEGORY_LABELS = {
  mandatory_skills: "Mandatory Skills",
  relevant_experience: "Experience",
  education: "Education",
  industry_keywords: "Industry Keywords",
  soft_skills: "Soft Skills",
  job_title_match: "Job Title Match",
  preferred_skills: "Preferred Skills",
};

// Shared 3-state status → accent color, used by the Evidence panel's sub-sections.
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
  if (finalScore >= 87) return { key: "excellent", label: "Excellent Match" };
  if (finalScore >= 82) return { key: "strong", label: "Strong Match" };
  if (finalScore >= 72) return { key: "review", label: "Review" };
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
  const all = [...(ranked || []), ...(excluded || [])].sort((a, b) => b.final_score - a.final_score);
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

/** Every matched skill across skill categories, deduplicated. */
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

/** Flat list across all skill-based categories for backward compatibility. */
export function evidenceList(record) {
  const cs = record.category_scores || {};
  const cats = [
    ["mandatory_skills", "Mandatory"],
    ["preferred_skills", "Preferred"],
    ["soft_skills", "Soft Skills"],
    ["industry_keywords", "Industry"],
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
    matchType: r.match_type, // "exact" | "evidence" | "none"
  };
}

/**
 * Structured evidence for the Candidate Detail page's Evidence panel
 */
export function evidenceSections(record) {
  const ev = record.evidence || {};

  const skillSections = [
    { key: "mandatory_skills", label: "Mandatory Skills", items: (ev.mandatory_skills || []).map(mapSkillRow) },
    //{ key: "industry_keywords", label: "Industry Keywords", items: (ev.industry_keywords || []).map(mapSkillRow) },
    { key: "soft_skills", label: "Soft Skills", items: (ev.soft_skills || []).map(mapSkillRow) },
    { key: "preferred_skills", label: "Preferred Skills", items: (ev.preferred_skills || []).map(mapSkillRow) },
  ];

  const experienceEv = ev.experience || {};
  const experienceRoles = (experienceEv.roles || []).map((r) => ({
    label: [r.title, r.company].filter(Boolean).join(" @ ") || "Role",
    detail: r.duration,
    status: r.status,
    similarity: r.relevance_similarity,
  }));

  let educationItems = [];
  if (ev.education && ev.education.length > 0) {
    educationItems = ev.education.map((e) => ({
      label: [e.required_degree_level, e.required_field].filter(Boolean).join(" in ") || "Requirement",
      status: e.status, // "matched" or "missing"
      detail: e.status === "matched" ? "Satisfied by candidate degree" : "Not found in resume",
    }));
  } else if (record.education && record.education.length > 0) {
    educationItems = record.education.map((e) => ({
      label: [e.degree_level, e.field].filter(Boolean).join(" in ") || "Degree",
      status: "matched",
      detail: e.institution || "Extracted from resume",
    }));
  }

  return {
    skillSections,
    experience: { years: experienceEv.years || null, roles: experienceRoles },
    jobTitle: ev.job_title || null,
    education: educationItems,
    additionalSkills: ev.additional_candidate_skills || [],
  };
}

export function formatEducation(education) {
  if (!education || education.length === 0) return "—";
  return education.map((e) => [e.degree_level, e.field].filter(Boolean).join(" — ")).join(", ");
}

/** Grounded, template-built explanation of a candidate's rank. */
export function explainRank(record, roleTitle) {
  const cs = record.category_scores || {};
  const missingMandatory = cs.mandatory_skills?.gate_missing || cs.mandatory_skills?.missing || [];
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

  if (cs.job_title_match?.notes) {
    sentences.push(cs.job_title_match.notes + ".");
  }

  // Return the array directly instead of sentences.join(" ")
  return sentences;
}
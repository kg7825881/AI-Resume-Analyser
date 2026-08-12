// lib/scoring.js
// Presentation helpers that derive everything from the real score records
// returned by GET /results/{role_id} — nothing here invents data.

// Max points per category, mirroring scorer.py's WEIGHTS * 100.
export const CATEGORY_MAX = {
  mandatory_skills: 35,
  relevant_experience: 25,
  technical_preferred_skills: 15,
  projects: 10,
  education: 5,
  preferred_skills_soft: 5,
  certifications_other: 5,
};

export const CATEGORY_LABELS = {
  mandatory_skills: "Mandatory skills",
  relevant_experience: "Experience",
  technical_preferred_skills: "Technical skills",
  projects: "Projects",
  education: "Education",
  preferred_skills_soft: "Preferred",
  certifications_other: "Certifications",
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
    ...(cs.technical_preferred_skills?.matched || []),
  ];
  return pool.slice(0, limit).join(" · ");
}

/**
 * Every matched skill across all skill-based categories (mandatory, technical/preferred,
 * soft preferred), deduplicated. Used for the "Skills matched" comparison field.
 */
export function allMatchedSkills(record) {
  const cs = record.category_scores || {};
  const pool = [
    ...(cs.mandatory_skills?.matched || []),
    ...(cs.technical_preferred_skills?.matched || []),
    ...(cs.preferred_skills_soft?.matched || []),
  ];
  return [...new Set(pool)];
}

/** The candidate's most recent role (first entry in their experience list), or null. */
export function mostRecentRole(record) {
  const experience = record.experience || [];
  return experience.length > 0 ? experience[0] : null;
}

/** Flat list of { skill, matched: bool, category } across every skill-based category, for evidence panels. */
export function evidenceList(record) {
  const cs = record.category_scores || {};
  const cats = [
    ["mandatory_skills", "Mandatory"],
    ["technical_preferred_skills", "Technical"],
    ["preferred_skills_soft", "Preferred"],
    ["certifications_other", "Certification"],
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

export function formatEducation(education) {
  if (!education || education.length === 0) return "—";
  return education.map((e) => [e.degree_level, e.field].filter(Boolean).join(" — ")).join(", ");
}

export function formatCertifications(certifications) {
  if (!certifications || certifications.length === 0) return "—";
  return certifications.join(", ");
}

export function formatProjects(projects) {
  if (!projects || projects.length === 0) return "—";
  return projects.map((p) => p.title).filter(Boolean).join(", ");
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

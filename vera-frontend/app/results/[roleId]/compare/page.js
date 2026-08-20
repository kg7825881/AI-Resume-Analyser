"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { getResults, listJDs } from "../../../../lib/api";
import { useAppState } from "../../../providers";
import {
  initials,
  rankAll,
  mostRecentRole,
  formatEducation,
} from "../../../../lib/scoring";

const THRESHOLD = 75;

// Rotating accent colors so each candidate's column reads as visually distinct at a glance.
const COLUMN_ACCENTS = ["#55d6c2", "#7c8cff", "#f5bd62", "#57d38c", "#ff7185", "#8ea4bb"];

// Same pill treatment as JDSummaryCard.js — solid var(--b) for mandatory (this candidate
// actually matched a hard requirement), plain "tag pref" for preferred/soft (nice-to-have,
// lower-stakes visually too). Kept local since it's only used in this one table.
function SkillPills({ skills, variant }) {
  if (!skills || skills.length === 0) return <span style={{ color: "var(--m)" }}>—</span>;
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
      {skills.map((skill) => (
        <span
          key={skill}
          className={variant === "mandatory" ? "tag" : "tag pref"}
          style={
            variant === "mandatory"
              ? { background: "var(--b)", color: "#fff", fontWeight: 600 }
              : undefined
          }
        >
          {skill}
        </span>
      ))}
    </div>
  );
}

export default function ComparisonPage({ params }) {
  const { roleId } = params;
  const router = useRouter();
  const { state, setCurrentRole, setResultsForRole } = useAppState();

  const cached = state.resultsCache[roleId];
  const [loading, setLoading] = useState(!cached);
  const [roleTitle, setRoleTitle] = useState(cached?.role_title || state.currentRole?.role_title || "");
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        setLoading(true);
        const [full, roles] = await Promise.all([getResults(roleId), listJDs().catch(() => [])]);
        if (cancelled) return;
        const matchedRole = roles.find((r) => r.role_id === roleId);
        const title = matchedRole?.role_title || roleTitle || roleId;
        setRoleTitle(title);
        setResultsForRole(roleId, { ...full, role_title: title });
        setCurrentRole({ role_id: roleId, role_title: title });
        setError(null);
      } catch (err) {
        if (!cancelled) setError(err.message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [roleId]);

  const data = state.resultsCache[roleId];
  const all = useMemo(() => (data ? rankAll(data.ranked, data.excluded_hard_gate_failed) : []), [data]);
  const qualified = useMemo(() => all.filter((r) => r.final_score >= THRESHOLD), [all]);

  if (loading && !data) {
    return (
      <section className="view active">
        <div className="center-pad">
          <span className="spinner" /> Loading comparison…
        </div>
      </section>
    );
  }

  if (error && !data) {
    return (
      <section className="view active">
        <div className="hero">
          <div>
            <div className="ey">Candidate comparison</div>
            <h1>No scores yet</h1>
            <p>{error}</p>
          </div>
          <button className="btn primary" onClick={() => router.push("/screen")}>
            ▶ Run an analysis
          </button>
        </div>
      </section>
    );
  }

  return (
    <section className="view active">
      <div className="hero">
        <div>
          <div className="ey">Side-by-side evaluation</div>
          <h1>Candidate Comparison</h1>
          <p>
            {roleTitle} · Job Fit ≥ {THRESHOLD} · {qualified.length} qualifying candidate(s)
          </p>
        </div>
        <button className="btn ghost" onClick={() => router.push(`/results/${roleId}`)}>
          ← Back to ranking
        </button>
      </div>

      <div className="panel">
        <div className="body">
          {qualified.length === 0 ? (
            <div className="center-pad">
              No candidates have scored {THRESHOLD} or above yet for this role.
            </div>
          ) : (
            <div style={{ overflow: "auto" }}>
              <table className="table" style={{ minWidth: qualified.length * 240 + 200 }}>
                <thead>
                  <tr>
                    <th style={{ position: "sticky", left: 0, background: "var(--p)", zIndex: 1 }}>
                      Field
                    </th>
                    {qualified.map((r, i) => (
                      <th
                        key={r.candidate_id}
                        style={{
                          borderTop: `3px solid ${COLUMN_ACCENTS[i % COLUMN_ACCENTS.length]}`,
                          cursor: "pointer",
                        }}
                        onClick={() => router.push(`/candidate/${r.candidate_id}?role=${roleId}`)}
                      >
                        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                          <div
                            className="mini"
                            style={{ background: COLUMN_ACCENTS[i % COLUMN_ACCENTS.length] }}
                          >
                            {initials(r.candidate_name)}
                          </div>
                          <div>
                            <div>{r.candidate_name || "Unnamed"}</div>
                            <small style={{ color: "var(--m)", fontWeight: 400 }}>#{r.rank}</small>
                          </div>
                        </div>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td style={{ position: "sticky", left: 0, background: "var(--p)", fontWeight: 700 }}>
                      Job Fit Score
                    </td>
                    {qualified.map((r) => (
                      <td key={r.candidate_id} className="score">
                        {r.final_score}%
                      </td>
                    ))}
                  </tr>
                  <tr>
                    <td style={{ position: "sticky", left: 0, background: "var(--p)", fontWeight: 700 }}>
                      Education
                    </td>
                    {qualified.map((r) => (
                      <td key={r.candidate_id}>{formatEducation(r.education)}</td>
                    ))}
                  </tr>
                  <tr>
                    <td style={{ position: "sticky", left: 0, background: "var(--p)", fontWeight: 700 }}>
                      Total experience
                    </td>
                    {qualified.map((r) => (
                      <td key={r.candidate_id}>
                        {r.total_years_experience != null ? `${r.total_years_experience} yrs` : "—"}
                      </td>
                    ))}
                  </tr>
                  <tr>
                    <td style={{ position: "sticky", left: 0, background: "var(--p)", fontWeight: 700 }}>
                      Last / recent role
                    </td>
                    {qualified.map((r) => {
                      const role = mostRecentRole(r);
                      const label = role ? [role.title, role.company].filter(Boolean).join(" @ ") : "";
                      return <td key={r.candidate_id}>{label || "—"}</td>;
                    })}
                  </tr>
                  <tr>
                    <td style={{ position: "sticky", left: 0, background: "var(--p)", fontWeight: 700 }}>
                      Mandatory skills matched
                    </td>
                    {qualified.map((r) => (
                      <td key={r.candidate_id}>
                        <SkillPills
                          skills={r.category_scores?.mandatory_skills?.matched}
                          variant="mandatory"
                        />
                      </td>
                    ))}
                  </tr>
                  <tr>
                    <td style={{ position: "sticky", left: 0, background: "var(--p)", fontWeight: 700 }}>
                      Preferred skills matched
                    </td>
                    {qualified.map((r) => {
                      const preferred = r.category_scores?.preferred_skills?.matched || [];
                      return (
                        <td key={r.candidate_id}>
                          <SkillPills skills={preferred} variant="preferred" />
                        </td>
                      );
                    })}
                  </tr>
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
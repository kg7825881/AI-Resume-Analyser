"use client";

import { Suspense, useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { getResults } from "../../../lib/api";
import { useAppState } from "../../providers";
import Pill from "../../../components/Pill";
import {
  statusFor,
  rankAll,
  evidenceSections,
  explainRank,
  CATEGORY_MAX,
  CATEGORY_LABELS,
} from "../../../lib/scoring";

export default function CandidateDetailPage({ params }) {
  return (
    <Suspense
      fallback={
        <section className="view active">
          <div className="center-pad">
            <span className="spinner" /> Loading candidate…
          </div>
        </section>
      }
    >
      <CandidateDetail params={params} />
    </Suspense>
  );
}

// Same small rounded chip shape as JDSummaryCard's requirement pills (.tag class),
// recolored by match status instead of mandatory/preferred. Full detail (match type,
// matched-against, relevance score, duration) sits in the title attribute as a hover
// tooltip so the chip itself stays compact, same as a JD requirement pill.
const STATUS_CHIP_STYLE = {
  matched: { background: "#12362b", color: "#67e19b" },
  weak_match: { background: "#3a2e18", color: "#f6c66c" },
  missing: { background: "#3a1f27", color: "#ff8192" },
};

const STATUS_ICON = {
  matched: "✓",
  weak_match: "~",
  missing: "×",
};

function EvidenceChip({ status, label, detail }) {
  const style = STATUS_CHIP_STYLE[status] || { background: "#14332f", color: "var(--a)" };
  const icon = STATUS_ICON[status] || "";
  return (
    <span className="tag" style={{ ...style, fontWeight: 600 }} title={detail || undefined}>
      {icon} {label}
    </span>
  );
}

function EvidenceSection({ title, emptyMessage, children }) {
  return (
    <div className="evidence-section">
      <h4
        style={{
          color: "var(--m)",
          fontSize: 11,
          textTransform: "uppercase",
          letterSpacing: 0.5,
          margin: "16px 0 8px",
        }}
      >
        {title}
      </h4>
      {emptyMessage ? (
        <div className="muted">{emptyMessage}</div>
      ) : (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>{children}</div>
      )}
    </div>
  );
}

function CandidateDetail({ params }) {
  const { candidateId } = params;
  const searchParams = useSearchParams();
  const router = useRouter();
  const { state, setResultsForRole } = useAppState();

  const roleId = searchParams.get("role") || state.currentRole?.role_id;
  const [loading, setLoading] = useState(!state.resultsCache[roleId]);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!roleId) return;
    let cancelled = false;
    async function load() {
      try {
        setLoading(true);
        const full = await getResults(roleId);
        if (cancelled) return;
        const roleTitle = state.resultsCache[roleId]?.role_title || state.currentRole?.role_title || "";
        setResultsForRole(roleId, { ...full, role_title: roleTitle });
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

  const data = roleId ? state.resultsCache[roleId] : null;
  const all = useMemo(() => (data ? rankAll(data.ranked, data.excluded_hard_gate_failed) : []), [data]);
  const record = all.find((r) => r.candidate_id === candidateId);
  const roleTitle = data?.role_title || state.currentRole?.role_title || "";

  if (!roleId) {
    return (
      <section className="view active">
        <div className="center-pad">
          No role in context. <Link href="/results">Pick a role</Link> to view candidates from.
        </div>
      </section>
    );
  }

  if (loading && !data) {
    return (
      <section className="view active">
        <div className="center-pad">
          <span className="spinner" /> Loading candidate…
        </div>
      </section>
    );
  }

  if ((error && !data) || (data && !record)) {
    return (
      <section className="view active">
        <div className="hero">
          <div>
            <div className="ey">Candidate analysis</div>
            <h1>Candidate not found</h1>
            <p>{error || "This candidate hasn't been scored for the current role."}</p>
          </div>
          <button className="btn primary" onClick={() => router.push(`/results/${roleId}`)}>
            ← Back to ranking
          </button>
        </div>
      </section>
    );
  }

  const status = statusFor(record.final_score);
  const sections = evidenceSections(record);
  const explanation = explainRank(record, roleTitle);
  const ringGradient = `conic-gradient(var(--a) 0 ${record.final_score}%, #173047 ${record.final_score}%)`;

  return (
    <section className="view active">
      <div className="hero">
        <div>
          <div className="ey">Candidate analysis</div>
          <h1>{record.candidate_name || "Unnamed candidate"}</h1>
          <p>
            {roleTitle} · Ranked #{record.rank} for current JD
          </p>
        </div>
        <button className="btn primary" onClick={() => router.push(`/results/${roleId}`)}>
          ← Back to ranking
        </button>
      </div>

      <div className="detail">
        <div className="panel scorecard">
          <div style={{ color: "var(--m)", fontSize: 11 }}>JOB FIT SCORE</div>
          <div className="ring" style={{ background: ringGradient }}>
            <b>{record.final_score}</b>
          </div>
          <Pill kind={status.key}>{status.label}</Pill>
          <p style={{ color: "var(--m)", lineHeight: 1.6 }}>{explanation}</p>
        </div>

        <div className="panel">
          <div className="head">
            <h3>Score breakdown</h3>
            <span className="tag">Explainable</span>
          </div>
          <div className="body metrics">
            {Object.entries(CATEGORY_MAX).map(([key, max]) => {
              const score = record.category_scores[key]?.score ?? 0;
              const pct = Math.round((score / max) * 100);
              return (
                <div className="metric" key={key}>
                  <small>{CATEGORY_LABELS[key]}</small>
                  <div className="bar">
                    <i style={{ width: `${pct}%` }} />
                  </div>
                  <strong>
                    {score}/{max}
                  </strong>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <div className="cols">
        <div className="panel">
          <div className="head">
            <h3>Evidence</h3>
          </div>
          <div className="body">
            {sections.skillSections.map((section) => (
              <EvidenceSection
                title={section.label}
                key={section.key}
                emptyMessage={section.items.length === 0 ? "No requirements in this category." : null}
              >
                {section.items.map((item, i) => (
                  <EvidenceChip
                    key={i}
                    status={item.status}
                    label={item.label}
                    detail={
                      item.detail ||
                      (item.status === "missing" ? "Not found in resume evidence" : item.matchType)
                    }
                  />
                ))}
              </EvidenceSection>
            ))}

            <EvidenceSection title="Experience">
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                {sections.experience.years && (
                  <EvidenceChip
                    status={sections.experience.years.status}
                    label={`${sections.experience.years.total_years_experience} yrs total`}
                    detail={`${sections.experience.years.min_years_required} yrs required`}
                  />
                )}
                {sections.experience.roles.map((r, i) => (
                  <EvidenceChip
                    key={i}
                    status={r.status}
                    label={r.label}
                    detail={[r.detail, `relevance ${r.similarity}`].filter(Boolean).join(" · ")}
                  />
                ))}
              </div>
              {sections.experience.roles.length === 0 && !sections.experience.years && (
                <div className="muted" style={{ marginTop: 6 }}>No experience entries recorded.</div>
              )}
            </EvidenceSection>

            <EvidenceSection
              title="Projects"
              emptyMessage={sections.projects.length === 0 ? "No projects listed." : null}
            >
              {sections.projects.map((p, i) => (
                <EvidenceChip key={i} status={p.status} label={p.label} detail={`relevance ${p.similarity}`} />
              ))}
            </EvidenceSection>

            <EvidenceSection
              title="Education"
              emptyMessage={sections.education.length === 0 ? "No specific education requirement in JD." : null}
            >
              {sections.education.map((e, i) => (
                <EvidenceChip key={i} status={e.status} label={e.label} />
              ))}
            </EvidenceSection>

            {sections.additionalSkills.length > 0 && (
              <EvidenceSection title="Additional candidate skills">
                {sections.additionalSkills.map((skill, i) => (
                  <span key={i} className="tag pref">
                    {skill}
                  </span>
                ))}
              </EvidenceSection>
            )}
          </div>
        </div>
        <div className="panel">
          <div className="head">
            <h3>Why ranked #{record.rank}?</h3>
          </div>
          <div className="body explain">{explanation}</div>
        </div>
      </div>
    </section>
  );
}
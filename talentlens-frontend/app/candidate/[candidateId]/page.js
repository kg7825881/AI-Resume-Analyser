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
  evidenceList,
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
  const evidence = evidenceList(record);
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
            <div className="reqs">
              {evidence.length === 0 && <div className="muted">No skill-based evidence recorded.</div>}
              {evidence.map((item, i) => (
                <div
                  className="match"
                  key={i}
                  style={{ borderLeft: `3px solid ${item.matched ? "var(--g)" : "var(--r)"}` }}
                >
                  {item.matched ? "✓" : "×"} {item.skill}{" "}
                  <span style={{ color: "var(--m)", fontSize: 11 }}>
                    — {item.category}
                    {item.matched ? " match" : " not found in resume evidence"}
                  </span>
                </div>
              ))}
            </div>
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

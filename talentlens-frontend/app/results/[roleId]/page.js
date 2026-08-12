"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { getResults, listJDs, getJD } from "../../../lib/api";
import { useAppState, useToast } from "../../providers";
import Pill from "../../../components/Pill";
import { statusFor, initials, passesMandatory, rankAll, topEvidence, CATEGORY_MAX } from "../../../lib/scoring";

const COMPARISON_THRESHOLD = 75;

export default function ResultsPage({ params }) {
  const { roleId } = params;
  const router = useRouter();
  const toast = useToast();
  const { state, setCurrentRole, setResultsForRole } = useAppState();

  const cached = state.resultsCache[roleId];
  const [loading, setLoading] = useState(!cached);
  const [roleTitle, setRoleTitle] = useState(cached?.role_title || state.currentRole?.role_title || "");
  const [filter, setFilter] = useState("all");
  const [error, setError] = useState(null);

  const [jd, setJd] = useState(null);
  const [jdLoading, setJdLoading] = useState(true);
  const [jdExpanded, setJdExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

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

  useEffect(() => {
    let cancelled = false;
    async function loadJd() {
      try {
        setJdLoading(true);
        const record = await getJD(roleId);
        if (!cancelled) setJd(record);
      } catch (err) {
        if (!cancelled) toast(`Couldn't load JD details: ${err.message}`, "error");
      } finally {
        if (!cancelled) setJdLoading(false);
      }
    }
    loadJd();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [roleId]);

  async function copyJdJson() {
    try {
      await navigator.clipboard.writeText(JSON.stringify(jd, null, 2));
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      toast("Couldn't copy to clipboard", "error");
    }
  }

  const data = state.resultsCache[roleId];
  const all = useMemo(() => (data ? rankAll(data.ranked, data.excluded_hard_gate_failed) : []), [data]);

  const stats = useMemo(() => {
    if (!all.length) return null;
    const total = all.length;
    const mandatoryPass = all.filter(passesMandatory).length;
    const strong = all.filter((r) => r.final_score >= 80).length;
    const avg = all.reduce((sum, r) => sum + r.final_score, 0) / total;
    return { total, mandatoryPass, strong, avg: avg.toFixed(1) };
  }, [all]);

  const qualifyingCount = useMemo(
    () => all.filter((r) => r.final_score >= COMPARISON_THRESHOLD).length,
    [all]
  );

  const filtered = all.filter((r) => {
    if (filter === "strong") return r.final_score >= 80;
    if (filter === "mandatory") return passesMandatory(r);
    return true;
  });

  if (loading && !data) {
    return (
      <section className="view active">
        <div className="center-pad">
          <span className="spinner" /> Loading ranking…
        </div>
      </section>
    );
  }

  if (error && !data) {
    return (
      <section className="view active">
        <div className="hero">
          <div>
            <div className="ey">Candidate ranking</div>
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
          <div className="ey">AI screening complete</div>
          <h1>Candidate ranking</h1>
          <p>
            {roleTitle} · {stats?.total ?? 0} resumes analyzed · same role criteria applied to every
            candidate.
          </p>
        </div>
        <div style={{ display: "flex", gap: 10 }}>
          <button className="btn ghost" onClick={() => router.push(`/results/${roleId}/compare`)}>
            ⇄ Compare (Job Fit ≥ {COMPARISON_THRESHOLD}) · {qualifyingCount}
          </button>
          <button className="btn primary" onClick={() => router.push("/screen")}>
            ＋ New screening
          </button>
        </div>
      </div>

      {stats && (
        <div className="stats">
          <div className="stat">
            <small>Resumes analyzed</small>
            <b>{stats.total}</b>
            <em>100% complete</em>
          </div>
          <div className="stat">
            <small>Mandatory pass</small>
            <b>{stats.mandatoryPass}</b>
            <em>{Math.round((stats.mandatoryPass / stats.total) * 100)}% of batch</em>
          </div>
          <div className="stat">
            <small>Strong matches</small>
            <b>{stats.strong}</b>
            <em>80+ Job Fit</em>
          </div>
          <div className="stat">
            <small>Avg. Job Fit</small>
            <b>{stats.avg}</b>
            <em>final score</em>
          </div>
        </div>
      )}

      <div className="panel" style={{ marginTop: 18 }}>
        <div className="head">
          <h3>Ranked candidates</h3>
          <div className="filters">
            <button className={filter === "all" ? "active" : ""} onClick={() => setFilter("all")}>
              All
            </button>
            <button className={filter === "strong" ? "active" : ""} onClick={() => setFilter("strong")}>
              80+
            </button>
            <button
              className={filter === "mandatory" ? "active" : ""}
              onClick={() => setFilter("mandatory")}
            >
              Mandatory pass
            </button>
          </div>
        </div>
        <div style={{ overflow: "auto" }}>
          <table className="table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Candidate</th>
                <th>Job Fit</th>
                <th>Mandatory</th>
                <th>Experience</th>
                <th>Top evidence</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((r) => {
                const status = statusFor(r.final_score);
                return (
                  <tr
                    key={r.candidate_id}
                    className="row-link"
                    onClick={() => router.push(`/candidate/${r.candidate_id}?role=${roleId}`)}
                  >
                    <td style={{ color: "#657f98", fontWeight: 800 }}>#{r.rank}</td>
                    <td>
                      <div className="cand">
                        <div className="mini">{initials(r.candidate_name)}</div>
                        <div>
                          <b>{r.candidate_name || "Unnamed candidate"}</b>
                          <small style={{ display: "block", color: "#7189a0" }}>{roleTitle}</small>
                        </div>
                      </div>
                    </td>
                    <td className="score">{r.final_score}%</td>
                    <td>
                      {r.category_scores.mandatory_skills.score}/{CATEGORY_MAX.mandatory_skills}
                    </td>
                    <td>
                      {r.category_scores.relevant_experience.score}/{CATEGORY_MAX.relevant_experience}
                    </td>
                    <td style={{ color: "#9fb2c5" }}>{topEvidence(r) || "—"}</td>
                    <td>
                      <Pill kind={status.key}>{status.label}</Pill>
                    </td>
                  </tr>
                );
              })}
              {filtered.length === 0 && (
                <tr>
                  <td colSpan={7} className="center-pad">
                    No candidates match this filter.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

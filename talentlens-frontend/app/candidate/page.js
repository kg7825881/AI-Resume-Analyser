"use client";

import Link from "next/link";
import { useAppState } from "../providers";

export default function CandidateIndexPage() {
  const { state } = useAppState();
  const resultsHref = state.currentRole ? `/results/${state.currentRole.role_id}` : "/results";

  return (
    <section className="view active">
      <div className="hero">
        <div>
          <div className="ey">Candidate analysis</div>
          <h1>No candidate selected.</h1>
          <p>Open a candidate from the ranking table to see their full score breakdown.</p>
        </div>
        <Link href={resultsHref} className="btn primary">
          ← Back to ranking
        </Link>
      </div>
    </section>
  );
}

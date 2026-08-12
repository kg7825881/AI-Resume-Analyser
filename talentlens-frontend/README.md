# TalentLens Frontend

A Next.js (App Router) frontend for the TalentLens FastAPI backend (`api.py`, no-auth version),
built to match the approved dark-dashboard design.

## Setup

```bash
npm install
cp .env.local.example .env.local   # point at your running FastAPI backend
npm run dev
```

Make sure the backend is running first:

```bash
uvicorn api:app --reload
```

By default the frontend expects the API at `http://localhost:8000` — change
`NEXT_PUBLIC_API_BASE_URL` in `.env.local` if it's elsewhere. Also make sure
`TALENTLENS_FRONTEND_ORIGINS` on the backend includes your frontend's origin
(defaults to `http://localhost:3000`, which matches `npm run dev`).

## Pages

- **`/screen`** — Upload a JD (`POST /jds/upload`) or pick one from the library
  (`GET /jds`), upload a batch of resumes (`POST /resumes/upload`), then run
  `POST /analyze` against just the candidates from this batch.
- **`/results/[roleId]`** — Ranked table for a role, from `GET /results/{role_id}`.
  Stats, filters (All / 80+ / Mandatory pass) and status pills are all computed
  client-side from the real score records — nothing is hardcoded.
- **`/candidate/[candidateId]`** — Score ring, category breakdown, evidence list,
  and a templated "why ranked #n" explanation, all built from the same
  `category_scores` payload the backend already computes in `scorer.py`.
- **`/copilot`** — A chat UI over the candidate pool. There's no LLM chat endpoint
  in the backend, so this is a small rule-based responder (`lib/copilot.js`) that
  answers ranking/comparison/skill-search/mandatory/"why" questions using only the
  real matched/missing/score fields already returned by `/results/{role_id}`.

## Notes

- State (current role, last upload batch, cached results per role) lives in a
  small React context (`app/providers.js`) persisted to `localStorage`, so a page
  refresh on `/results/[roleId]` or `/candidate/[candidateId]` still works —
  it just refetches from the API.
- `POST /analyze` scores **every** stored resume if `candidate_ids` is omitted.
  The Screen page always passes the ids from the current upload batch so a new
  screening doesn't re-score every resume ever uploaded.

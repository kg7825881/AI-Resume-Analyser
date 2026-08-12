// lib/api.js
// Thin client for the TalentLens FastAPI backend (api.py — no-auth version).
// All functions throw an Error with a readable message on non-2xx responses,
// so callers can just try/catch and show err.message.

const BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(`${BASE_URL}${path}`, options);
  } catch (err) {
    throw new Error(
      `Couldn't reach the TalentLens API at ${BASE_URL}. Is the backend running (uvicorn api:app --reload)?`
    );
  }

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      // response wasn't JSON — keep statusText
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }

  if (res.status === 204) return null;
  return res.json();
}

/** Upload a single JD file (.pdf/.docx). Returns { role_id, role_title, document_id, extraction_method, extraction_warnings }. */
export function uploadJD(file) {
  const form = new FormData();
  form.append("file", file);
  return request("/jds/upload", { method: "POST", body: form });
}

/** List every JD role currently in the library. */
export function listJDs() {
  return request("/jds");
}

/** Fetch the full extracted JD record (mandatory_skills, responsibilities, etc.) for one role. */
export function getJD(roleId) {
  return request(`/jds/${encodeURIComponent(roleId)}`);
}

/**
 * Upload a batch of resumes (.pdf/.docx). Individual failures don't stop the rest.
 * Returns { results: [{ file_name, status: 'ok'|'failed', candidate_id?, candidate_name?, extraction_warnings?, error? }] }.
 */
export function uploadResumes(files) {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  return request("/resumes/upload", { method: "POST", body: form });
}

/**
 * Resolve a role query and score resumes against it.
 * candidateIds: pass the ids from the current upload batch, or omit to score every stored resume.
 */
export function analyze(roleQuery, candidateIds) {
  return request("/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      role_query: roleQuery,
      candidate_ids: candidateIds && candidateIds.length ? candidateIds : null,
    }),
  });
}

/** Fetch the full ranked list of scores (with category breakdowns) for a role. */
export function getResults(roleId) {
  return request(`/results/${encodeURIComponent(roleId)}`);
}

export { BASE_URL };

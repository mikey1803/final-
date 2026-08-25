// API client for the blind-design study backend. Mirrors the existing
// dashboard's `src/lib/demoApi.js` request/post pattern (plain fetch, no
// axios) for consistency with the rest of the project, against the
// SEPARATE, standalone study FastAPI service (see
// study/app/backend/main.py's module docstring for why it's a distinct
// service rather than a route added to the demo API).

const API_BASE = import.meta.env.VITE_STUDY_API_BASE || "http://localhost:8100";

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, options);
  } catch (err) {
    throw new Error(
      `Could not reach the study server at ${API_BASE}. Is it running? (${err.message})`
    );
  }
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = body && body.detail ? body.detail : response.statusText;
    const error = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    error.status = response.status;
    error.body = body;
    throw error;
  }
  return body;
}

function post(path, body) {
  return request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function getHealth() {
  return request("/api/study/health");
}

export function getTaskContent(token) {
  return request(`/api/study/task-content?token=${encodeURIComponent(token)}`);
}

export function submitResponse(payload) {
  return post("/api/study/submit", payload);
}

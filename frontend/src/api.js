// Thin fetch wrapper over the read-only backend API. In dev, Vite proxies
// /api to the FastAPI server on :8000 (see vite.config.js), so these are
// relative URLs. The backend is the single source of truth — this client
// never computes anything, it only fetches.

async function getJSON(path) {
  const res = await fetch(path);
  if (!res.ok) {
    throw new Error(`${path} failed: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export const fetchModels = () => getJSON("/api/models");
export const fetchPerturbations = () => getJSON("/api/perturbations");
export const fetchResults = () => getJSON("/api/results");
export const fetchRobustnessIndex = () => getJSON("/api/robustness-index");
export const fetchInsights = () => getJSON("/api/insights");

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch (_) {
      /* ignore parse errors */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

export const api = {
  health: () => request("/api/health"),
  getDatasets: () => request("/api/datasets"),

  trainModel: (dataset, model_type) =>
    request("/api/models/train", {
      method: "POST",
      body: JSON.stringify({ dataset, model_type }),
    }),
  listModels: () => request("/api/models"),
  getModel: (modelId) => request(`/api/models/${modelId}`),

  runAttack: (payload) =>
    request("/api/attack/run", { method: "POST", body: JSON.stringify(payload) }),

  runPreprocessingDefense: (payload) =>
    request("/api/defense/preprocessing", { method: "POST", body: JSON.stringify(payload) }),

  runAdversarialTraining: (payload) =>
    request("/api/defense/adversarial-training", { method: "POST", body: JSON.stringify(payload) }),

  evaluateRobustness: (payload) =>
    request("/api/robustness/evaluate", { method: "POST", body: JSON.stringify(payload) }),

  listExperiments: (limit = 100) => request(`/api/experiments?limit=${limit}`),

  exportReportUrl: () => `${API_BASE}/api/experiments/export`,
};

export default api;

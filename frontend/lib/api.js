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

  // threat model
  getThreatModelMatrix: () => request("/api/threat-model/matrix"),
  checkThreatModel: (attack, capability) =>
    request("/api/threat-model/check", {
      method: "POST",
      body: JSON.stringify({ attack, capability }),
    }),

  // black-box lab
  runBlackBoxAttack: (payload) =>
    request("/api/attack/blackbox", { method: "POST", body: JSON.stringify(payload) }),

  // poisoning lab
  runLabelFlipPoisoning: (payload) =>
    request("/api/poisoning/label-flip", { method: "POST", body: JSON.stringify(payload) }),
  runBackdoorPoisoning: (payload) =>
    request("/api/poisoning/backdoor", { method: "POST", body: JSON.stringify(payload) }),

  // transferability
  computeTransferability: (payload) =>
    request("/api/robustness/transferability", { method: "POST", body: JSON.stringify(payload) }),

  // security assessment + leaderboard
  runSecurityAssessment: (payload) =>
    request("/api/security/assessment", { method: "POST", body: JSON.stringify(payload) }),
  getLeaderboard: (sortBy = "robustness_score") =>
    request(`/api/leaderboard?sort_by=${sortBy}`),

  // experiment grid
  runExperimentGrid: (payload) =>
    request("/api/experiments/grid", { method: "POST", body: JSON.stringify(payload) }),
  getExperimentGridLimits: () => request("/api/experiments/grid/limits"),

  listExperiments: (limit = 100) => request(`/api/experiments?limit=${limit}`),

  exportReportUrl: () => `${API_BASE}/api/experiments/export`,
};

export default api;

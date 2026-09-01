"use client";

import { useEffect, useState } from "react";
import PageHeader from "@/components/PageHeader";
import Panel from "@/components/Panel";
import Button from "@/components/Button";
import ModelPicker from "@/components/ModelPicker";
import MetricStat from "@/components/MetricStat";
import Pill from "@/components/Pill";
import api from "@/lib/api";

const ATTACKS = ["fgsm", "pgd", "deepfool", "random_noise"];

const RISK_TONE = { HIGH: "red", MEDIUM: "amber", LOW: "green" };
const SEVERITY_TONE = { HIGH: "red", MEDIUM: "amber", LOW: "cyan" };

export default function SecurityAssessmentPage() {
  const [models, setModels] = useState([]);
  const [modelId, setModelId] = useState(null);
  const [compareIds, setCompareIds] = useState([]);
  const [attacks, setAttacks] = useState(["fgsm", "pgd", "random_noise"]);
  const [epsilon, setEpsilon] = useState(0.15);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.listModels().then((r) => {
      setModels(r.models);
      if (r.models.length) setModelId(r.models[0].model_id);
    });
  }, []);

  const toggleAttack = (a) =>
    setAttacks((prev) => (prev.includes(a) ? prev.filter((x) => x !== a) : [...prev, a]));

  const toggleCompare = (id) =>
    setCompareIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));

  const comparableModels = models.filter(
    (m) => m.model_id !== modelId && m.dataset === models.find((x) => x.model_id === modelId)?.dataset
  );

  const handleRun = async () => {
    setRunning(true);
    setError(null);
    try {
      const res = await api.runSecurityAssessment({
        model_id: modelId,
        attacks,
        epsilon: Number(epsilon),
        compare_model_ids: compareIds,
      });
      setResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setRunning(false);
    }
  };

  return (
    <div>
      <PageHeader
        eyebrow="Assessment"
        title="Security Assessment"
        description="Runs a robustness benchmark (and, optionally, a transferability check against other models) and turns it into a risk rating with specific, defensible findings — not just a bare score."
      />

      <div className="grid grid-cols-3 gap-6">
        <Panel eyebrow="Configuration" title="Assessment scope" className="col-span-1 h-fit">
          <div className="space-y-4">
            <ModelPicker models={models} value={modelId} onChange={setModelId} />

            <div>
              <span className="block font-display text-[10px] tracking-widest text-muted uppercase mb-1.5">
                Attacks to include
              </span>
              <div className="space-y-1.5">
                {ATTACKS.map((a) => (
                  <label key={a} className="flex items-center gap-2 text-sm font-mono text-ink">
                    <input
                      type="checkbox"
                      checked={attacks.includes(a)}
                      onChange={() => toggleAttack(a)}
                      className="accent-cyan"
                    />
                    {a}
                  </label>
                ))}
              </div>
            </div>

            {comparableModels.length > 0 && (
              <div>
                <span className="block font-display text-[10px] tracking-widest text-muted uppercase mb-1.5">
                  Compare transferability against
                </span>
                <div className="space-y-1.5">
                  {comparableModels.map((m) => (
                    <label key={m.model_id} className="flex items-center gap-2 text-sm font-mono text-ink">
                      <input
                        type="checkbox"
                        checked={compareIds.includes(m.model_id)}
                        onChange={() => toggleCompare(m.model_id)}
                        className="accent-cyan"
                      />
                      {m.model_type} #{m.model_id}
                    </label>
                  ))}
                </div>
              </div>
            )}

            <div>
              <div className="flex justify-between items-baseline mb-1.5">
                <span className="font-display text-[10px] tracking-widest text-muted uppercase">
                  Epsilon
                </span>
                <span className="font-mono text-sm text-ink">{epsilon}</span>
              </div>
              <input
                type="range"
                min="0.01"
                max="0.4"
                step="0.01"
                value={epsilon}
                onChange={(e) => setEpsilon(e.target.value)}
                className="w-full accent-cyan"
              />
            </div>

            <Button
              variant="danger"
              onClick={handleRun}
              disabled={!modelId || running || attacks.length === 0}
              className="w-full"
            >
              {running ? "Assessing…" : "Run assessment"}
            </Button>
            {error && <p className="text-xs text-red font-mono">{error}</p>}
          </div>
        </Panel>

        <div className="col-span-2 space-y-4">
          {!result && (
            <Panel>
              <p className="text-sm text-muted font-mono">
                Configure and run an assessment to see a risk report here.
              </p>
            </Panel>
          )}

          {result && (
            <>
              <Panel
                eyebrow="Overall risk"
                title={`Recommendation: ${result.recommendation}`}
                right={<Pill tone={RISK_TONE[result.risk_rating]}>{result.risk_rating} RISK</Pill>}
              >
                <div className="grid grid-cols-3 gap-4 mb-2">
                  <MetricStat
                    label="Clean accuracy"
                    value={`${Math.round(result.clean_accuracy * 100)}%`}
                    tone="cyan"
                  />
                  <MetricStat
                    label="Mean robust accuracy"
                    value={
                      result.mean_robust_accuracy != null
                        ? `${Math.round(result.mean_robust_accuracy * 100)}%`
                        : "—"
                    }
                    tone="red"
                  />
                  <MetricStat label="Findings" value={result.findings.length} />
                </div>
                <div className="flex items-center justify-between pt-3 mt-1 border-t border-border">
                  <span className="font-display text-[10px] tracking-widest text-muted uppercase">
                    Aegis Robustness Index (composite, not a standard benchmark)
                  </span>
                  <span
                    className={`font-mono text-lg font-semibold ${
                      result.risk_rating === "LOW"
                        ? "text-green"
                        : result.risk_rating === "MEDIUM"
                        ? "text-amber"
                        : "text-red"
                    }`}
                  >
                    {result.aegis_robustness_index ?? result.robustness_score}/100
                  </span>
                </div>
              </Panel>

              <Panel eyebrow="Findings" title="What the evaluation found">
                <div className="space-y-2">
                  {result.findings.map((f, i) => (
                    <div key={i} className="flex gap-3 items-start border border-border p-3">
                      <Pill tone={SEVERITY_TONE[f.severity]}>{f.severity}</Pill>
                      <p className="text-sm text-ink font-mono leading-relaxed">{f.text}</p>
                    </div>
                  ))}
                </div>
              </Panel>

              <Panel eyebrow="Next steps" title="Recommended actions">
                <ul className="space-y-2">
                  {result.recommended_actions.map((action, i) => (
                    <li key={i} className="text-sm text-ink font-mono flex gap-2">
                      <span className="text-cyan">{i + 1}.</span>
                      {action}
                    </li>
                  ))}
                </ul>
              </Panel>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

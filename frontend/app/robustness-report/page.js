"use client";

import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import PageHeader from "@/components/PageHeader";
import Panel from "@/components/Panel";
import Button from "@/components/Button";
import ModelPicker from "@/components/ModelPicker";
import MetricStat from "@/components/MetricStat";
import Pill from "@/components/Pill";
import api from "@/lib/api";

const ATTACKS = ["fgsm", "pgd", "deepfool", "random_noise"];

export default function RobustnessReportPage() {
  const [models, setModels] = useState([]);
  const [modelId, setModelId] = useState(null);
  const [attacks, setAttacks] = useState(["fgsm", "pgd", "random_noise"]);
  const [epsilon, setEpsilon] = useState(0.15);
  const [nSamples, setNSamples] = useState(200);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.listModels().then((r) => {
      setModels(r.models);
      if (r.models.length) setModelId(r.models[0].model_id);
    });
  }, []);

  const toggleAttack = (a) => {
    setAttacks((prev) => (prev.includes(a) ? prev.filter((x) => x !== a) : [...prev, a]));
  };

  const handleRun = async () => {
    setRunning(true);
    setError(null);
    try {
      const res = await api.evaluateRobustness({
        model_id: modelId,
        attacks,
        epsilon: Number(epsilon),
        n_samples: Number(nSamples),
      });
      setResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setRunning(false);
    }
  };

  const chartData = result
    ? [
        { name: "clean", accuracy: result.clean_accuracy * 100 },
        ...Object.entries(result.attacks)
          .filter(([, v]) => v.accuracy != null)
          .map(([name, v]) => ({ name, accuracy: v.accuracy * 100 })),
      ]
    : [];

  const scoreTone = (score) => (score >= 70 ? "green" : score >= 40 ? "amber" : "red");

  return (
    <div>
      <PageHeader
        eyebrow="Step 4"
        title="Robustness Report"
        description="Run a batch evaluation across the model's held-out test set and get a single 0–100 robustness score plus a per-attack breakdown."
      />

      <div className="grid grid-cols-3 gap-6">
        <Panel eyebrow="Configuration" title="Batch evaluation" className="col-span-1 h-fit">
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

            <div>
              <div className="flex justify-between items-baseline mb-1.5">
                <span className="font-display text-[10px] tracking-widest text-muted uppercase">
                  Samples to test
                </span>
                <span className="font-mono text-sm text-ink">{nSamples}</span>
              </div>
              <input
                type="range"
                min="20"
                max="500"
                step="10"
                value={nSamples}
                onChange={(e) => setNSamples(e.target.value)}
                className="w-full accent-cyan"
              />
              <p className="text-[11px] text-muted mt-1">
                DeepFool is always capped at 64 samples — it's iterative and per-sample, so it's
                slower than the other attacks.
              </p>
            </div>

            <Button
              onClick={handleRun}
              disabled={!modelId || running || attacks.length === 0}
              className="w-full"
            >
              {running ? "Evaluating…" : "Run evaluation"}
            </Button>
            {error && <p className="text-xs text-red font-mono">{error}</p>}
          </div>
        </Panel>

        <div className="col-span-2 space-y-4">
          {!result && (
            <Panel>
              <p className="text-sm text-muted font-mono">
                Configure and run a batch evaluation to see a robustness report here.
              </p>
            </Panel>
          )}

          {result && (
            <>
              <Panel
                eyebrow="Overall"
                title="Robustness score"
                right={
                  <a href={api.exportReportUrl()} target="_blank" rel="noreferrer">
                    <Pill tone="cyan">export report ↓</Pill>
                  </a>
                }
              >
                <div className="grid grid-cols-3 gap-4 mb-4">
                  <MetricStat
                    label="Robustness score"
                    value={result.robustness_score}
                    suffix="/100"
                    tone={scoreTone(result.robustness_score)}
                  />
                  <MetricStat
                    label="Clean accuracy"
                    value={`${Math.round(result.clean_accuracy * 100)}%`}
                    tone="cyan"
                  />
                  <MetricStat label="Samples evaluated" value={result.n_samples_evaluated} />
                </div>
                <div className="h-2 bg-panel2 border border-border">
                  <div
                    className={`h-full ${
                      result.robustness_score >= 70
                        ? "bg-green"
                        : result.robustness_score >= 40
                        ? "bg-amber"
                        : "bg-red"
                    }`}
                    style={{ width: `${result.robustness_score}%` }}
                  />
                </div>
              </Panel>

              <Panel eyebrow="Breakdown" title="Accuracy by attack">
                <div className="h-52 mb-4">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#232D3D" />
                      <XAxis dataKey="name" stroke="#8B98A5" fontSize={11} tickLine={false} />
                      <YAxis stroke="#8B98A5" fontSize={11} tickLine={false} domain={[0, 100]} />
                      <Tooltip
                        contentStyle={{
                          background: "#111826",
                          border: "1px solid #232D3D",
                          fontSize: 12,
                        }}
                        labelStyle={{ color: "#E6EDF3" }}
                      />
                      <Bar dataKey="accuracy" fill="#FF4D5E" radius={[2, 2, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                <table className="w-full text-sm font-mono">
                  <thead>
                    <tr className="text-left text-muted text-[11px] uppercase tracking-wide">
                      <th className="pb-2 font-display font-normal">Attack</th>
                      <th className="pb-2 font-display font-normal">Accuracy</th>
                      <th className="pb-2 font-display font-normal">Success rate</th>
                      <th className="pb-2 font-display font-normal">Avg. perturbation</th>
                      <th className="pb-2 font-display font-normal">L2</th>
                      <th className="pb-2 font-display font-normal">L∞</th>
                      <th className="pb-2 font-display font-normal">Conf. shift</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(result.attacks).map(([name, v]) => (
                      <tr key={name} className="border-t border-border">
                        <td className="py-1.5 text-ink">{name}</td>
                        <td className="py-1.5 text-ink">
                          {v.accuracy != null ? `${Math.round(v.accuracy * 100)}%` : "—"}
                        </td>
                        <td className="py-1.5 text-amber">
                          {v.attack_success_rate != null
                            ? `${Math.round(v.attack_success_rate * 100)}%`
                            : v.error || "—"}
                        </td>
                        <td className="py-1.5 text-muted">{v.avg_perturbation ?? "—"}</td>
                        <td className="py-1.5 text-muted">{v.l2_norm ?? "—"}</td>
                        <td className="py-1.5 text-muted">{v.linf_norm ?? "—"}</td>
                        <td className="py-1.5 text-muted">{v.confidence_shift ?? "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Panel>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

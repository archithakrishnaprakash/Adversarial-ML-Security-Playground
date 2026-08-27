"use client";

import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import PageHeader from "@/components/PageHeader";
import Panel from "@/components/Panel";
import Select from "@/components/Select";
import Button from "@/components/Button";
import ModelPicker from "@/components/ModelPicker";
import MetricStat from "@/components/MetricStat";
import Pill from "@/components/Pill";
import api from "@/lib/api";

const ATTACKS = ["fgsm", "pgd", "deepfool", "random_noise"];
const PREPROCESS_OPTIONS = [
  { value: "gaussian_smoothing", label: "Gaussian smoothing" },
  { value: "feature_clipping", label: "Feature clipping" },
  { value: "normalization", label: "Normalization" },
];

export default function DefenseLabPage() {
  const [models, setModels] = useState([]);

  useEffect(() => {
    api.listModels().then((r) => setModels(r.models));
  }, []);

  return (
    <div>
      <PageHeader
        eyebrow="Step 3"
        title="Defense Lab"
        description="Compare clean vs. attacked vs. defended accuracy, or train a hardened version of a model and see how much robustness it actually gained."
      />
      <div className="grid grid-cols-2 gap-6 items-start">
        <PreprocessingDefense models={models} />
        <AdversarialTrainingDefense models={models} onTrained={() => api.listModels().then((r) => setModels(r.models))} />
      </div>
    </div>
  );
}

function PreprocessingDefense({ models }) {
  const [modelId, setModelId] = useState(null);
  const [attack, setAttack] = useState("fgsm");
  const [defense, setDefense] = useState("gaussian_smoothing");
  const [epsilon, setEpsilon] = useState(0.15);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (models.length && !modelId) setModelId(models[0].model_id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [models]);

  const handleRun = async () => {
    setRunning(true);
    setError(null);
    try {
      const res = await api.runPreprocessingDefense({
        model_id: modelId,
        attack,
        defense,
        epsilon: Number(epsilon),
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
        { name: "Clean", value: result.clean_accuracy * 100 },
        { name: "Attacked", value: result.adversarial_accuracy * 100 },
        { name: "Defended", value: result.defended_accuracy * 100 },
      ]
    : [];

  return (
    <Panel eyebrow="Input preprocessing" title="Attack → Defend → Compare">
      <div className="space-y-4 mb-5">
        <ModelPicker models={models} value={modelId} onChange={setModelId} />
        <div className="grid grid-cols-2 gap-3">
          <Select
            label="Attack"
            value={attack}
            onChange={setAttack}
            options={ATTACKS.map((a) => ({ value: a, label: a }))}
          />
          <Select
            label="Defense"
            value={defense}
            onChange={setDefense}
            options={PREPROCESS_OPTIONS}
          />
        </div>
        <Button onClick={handleRun} disabled={!modelId || running} className="w-full">
          {running ? "Running…" : "Run comparison"}
        </Button>
        {error && <p className="text-xs text-red font-mono">{error}</p>}
      </div>

      {result && (
        <>
          <div className="h-40 mb-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#232D3D" />
                <XAxis dataKey="name" stroke="#8B98A5" fontSize={11} tickLine={false} />
                <YAxis stroke="#8B98A5" fontSize={11} tickLine={false} domain={[0, 100]} />
                <Tooltip
                  contentStyle={{ background: "#111826", border: "1px solid #232D3D", fontSize: 12 }}
                  labelStyle={{ color: "#E6EDF3" }}
                />
                <Bar dataKey="value" fill="#22D3EE" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="flex items-center justify-between pt-3 border-t border-border">
            <span className="font-mono text-xs text-muted">Accuracy recovered by defense</span>
            <Pill tone={result.accuracy_recovered > 0 ? "green" : "amber"}>
              {result.accuracy_recovered > 0 ? "+" : ""}
              {Math.round(result.accuracy_recovered * 100)} pts
            </Pill>
          </div>
        </>
      )}
    </Panel>
  );
}

function AdversarialTrainingDefense({ models, onTrained }) {
  const [modelId, setModelId] = useState(null);
  const [epsilon, setEpsilon] = useState(0.15);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (models.length && !modelId) setModelId(models[0].model_id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [models]);

  const handleRun = async () => {
    setRunning(true);
    setError(null);
    try {
      const res = await api.runAdversarialTraining({
        model_id: modelId,
        epsilon: Number(epsilon),
        attacks_to_compare: ["fgsm", "pgd"],
      });
      setResult(res);
      onTrained?.();
    } catch (err) {
      setError(err.message);
    } finally {
      setRunning(false);
    }
  };

  return (
    <Panel eyebrow="Adversarial training" title="Train a hardened model">
      <div className="space-y-4 mb-5">
        <ModelPicker models={models} value={modelId} onChange={setModelId} />
        <div>
          <div className="flex justify-between items-baseline mb-1.5">
            <span className="font-display text-[10px] tracking-widest text-muted uppercase">
              Training epsilon
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
        <Button onClick={handleRun} disabled={!modelId || running} className="w-full">
          {running ? "Training robust model…" : "Train robust version"}
        </Button>
        {error && <p className="text-xs text-red font-mono">{error}</p>}
        <p className="text-[11px] text-muted leading-relaxed">
          Retrains a fresh copy of the model on a mix of clean and FGSM adversarial examples, then
          compares it against the original under attack. This can take longer than other actions.
        </p>
      </div>

      {result && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <MetricStat
              label="Clean acc · normal"
              value={`${Math.round(result.comparison.clean.normal_model * 100)}%`}
            />
            <MetricStat
              label="Clean acc · robust"
              value={`${Math.round(result.comparison.clean.robust_model * 100)}%`}
              tone="green"
            />
          </div>
          <table className="w-full text-sm font-mono">
            <thead>
              <tr className="text-left text-muted text-[11px] uppercase tracking-wide">
                <th className="pb-2 font-display font-normal">Under attack</th>
                <th className="pb-2 font-display font-normal">Normal model</th>
                <th className="pb-2 font-display font-normal">Robust model</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(result.comparison.attacks).map(([attack, vals]) => (
                <tr key={attack} className="border-t border-border">
                  <td className="py-1.5 text-ink">{attack}</td>
                  <td className="py-1.5 text-red">
                    {vals.normal_model != null ? `${Math.round(vals.normal_model * 100)}%` : "—"}
                  </td>
                  <td className="py-1.5 text-green">
                    {vals.robust_model != null ? `${Math.round(vals.robust_model * 100)}%` : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="text-[11px] text-muted font-mono">
            Robust model registered as #{result.robust_model_id} — also visible on the Models page.
          </p>
        </div>
      )}
    </Panel>
  );
}

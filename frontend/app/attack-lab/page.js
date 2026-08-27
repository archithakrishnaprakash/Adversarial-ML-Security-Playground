"use client";

import { useEffect, useState } from "react";
import PageHeader from "@/components/PageHeader";
import Panel from "@/components/Panel";
import Select from "@/components/Select";
import Button from "@/components/Button";
import Pill from "@/components/Pill";
import ConfidenceBar from "@/components/ConfidenceBar";
import ModelPicker from "@/components/ModelPicker";
import api from "@/lib/api";

const ATTACK_OPTIONS = [
  { value: "fgsm", label: "FGSM — Fast Gradient Sign" },
  { value: "pgd", label: "PGD — Projected Gradient Descent" },
  { value: "deepfool", label: "DeepFool — minimal perturbation" },
  { value: "random_noise", label: "Random Noise — baseline" },
];

export default function AttackLabPage() {
  const [models, setModels] = useState([]);
  const [modelId, setModelId] = useState(null);
  const [attack, setAttack] = useState("fgsm");
  const [epsilon, setEpsilon] = useState(0.15);
  const [pgdSteps, setPgdSteps] = useState(10);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.listModels().then((r) => {
      setModels(r.models);
      if (r.models.length && !modelId) setModelId(r.models[0].model_id);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const selectedModel = models.find((m) => m.model_id === modelId);

  const handleRun = async () => {
    setRunning(true);
    setError(null);
    try {
      const res = await api.runAttack({
        model_id: modelId,
        attack,
        epsilon: Number(epsilon),
        pgd_steps: Number(pgdSteps),
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
        eyebrow="Step 2"
        title="Attack Lab"
        description="Pick a trained model, choose an attack, and watch the original input, its perturbation, and the resulting prediction side by side."
      />

      <div className="grid grid-cols-3 gap-6">
        <div className="col-span-1 space-y-4">
          <Panel eyebrow="Target" title="Model">
            <ModelPicker models={models} value={modelId} onChange={setModelId} />
          </Panel>

          <Panel eyebrow="Configuration" title="Attack">
            <div className="space-y-4">
              <Select
                label="Attack method"
                value={attack}
                onChange={setAttack}
                options={ATTACK_OPTIONS}
              />
              {attack !== "deepfool" && (
                <div>
                  <div className="flex justify-between items-baseline mb-1.5">
                    <span className="font-display text-[10px] tracking-widest text-muted uppercase">
                      Attack strength (ε)
                    </span>
                    <span className="font-mono text-sm text-ink">{epsilon}</span>
                  </div>
                  <input
                    type="range"
                    min="0.01"
                    max="0.5"
                    step="0.01"
                    value={epsilon}
                    onChange={(e) => setEpsilon(e.target.value)}
                    className="w-full accent-cyan"
                  />
                </div>
              )}
              {attack === "pgd" && (
                <div>
                  <div className="flex justify-between items-baseline mb-1.5">
                    <span className="font-display text-[10px] tracking-widest text-muted uppercase">
                      PGD steps
                    </span>
                    <span className="font-mono text-sm text-ink">{pgdSteps}</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="30"
                    step="1"
                    value={pgdSteps}
                    onChange={(e) => setPgdSteps(e.target.value)}
                    className="w-full accent-cyan"
                  />
                </div>
              )}
              <Button
                variant="danger"
                onClick={handleRun}
                disabled={!modelId || running}
                className="w-full"
              >
                {running ? "Attacking…" : "Run attack"}
              </Button>
              {error && <p className="text-xs text-red font-mono">{error}</p>}
              {selectedModel?.attack_note && (
                <p className="text-[11px] text-amber leading-relaxed">
                  {selectedModel.attack_note}
                </p>
              )}
            </div>
          </Panel>
        </div>

        <div className="col-span-2 space-y-4">
          {!result && (
            <Panel>
              <p className="text-sm text-muted font-mono">
                Configure an attack on the left and run it to see results here.
              </p>
            </Panel>
          )}

          {result && (
            <>
              <Panel
                eyebrow={result.attack_succeeded ? "Attack succeeded" : "Attack failed"}
                title="Result"
                right={
                  <Pill tone={result.attack_succeeded ? "red" : "green"}>
                    {result.attack_succeeded ? "prediction flipped" : "prediction unchanged"}
                  </Pill>
                }
              >
                {result.original_image ? (
                  <div className="grid grid-cols-3 gap-4 mb-4">
                    <ImageCell label="Original" src={result.original_image} />
                    <ImageCell label="Perturbation" src={result.perturbation_heatmap} />
                    <ImageCell label="Adversarial" src={result.adversarial_image} />
                  </div>
                ) : (
                  <FeatureDeltaTable
                    deltas={result.feature_deltas}
                    classNames={result.class_names}
                  />
                )}

                <div className="grid grid-cols-2 gap-6 mt-2">
                  <ConfidenceBar
                    label={`Original prediction: ${labelFor(result, result.original_prediction)}`}
                    value={result.original_confidence}
                    tone="cyan"
                  />
                  <ConfidenceBar
                    label={`Adversarial prediction: ${labelFor(result, result.adversarial_prediction)}`}
                    value={result.adversarial_confidence}
                    tone="red"
                  />
                </div>

                <div className="grid grid-cols-3 gap-4 mt-4 pt-4 border-t border-border">
                  <Stat label="True label" value={labelFor(result, result.true_label)} />
                  <Stat label="Perturbation magnitude" value={result.perturbation_magnitude} />
                  <Stat label="Epsilon" value={result.epsilon} />
                </div>
              </Panel>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function labelFor(result, classIdx) {
  if (result.class_names) return result.class_names[classIdx] ?? classIdx;
  return classIdx;
}

function ImageCell({ label, src }) {
  return (
    <div className="text-center">
      <div className="font-display text-[10px] tracking-widest text-muted uppercase mb-2">
        {label}
      </div>
      <div className="border border-border bg-panel2 p-2 inline-block">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={src} alt={label} className="w-28 h-28 object-contain" />
      </div>
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div>
      <div className="font-display text-[10px] tracking-widest text-muted uppercase mb-1">
        {label}
      </div>
      <div className="font-mono text-sm text-ink">{value}</div>
    </div>
  );
}

function FeatureDeltaTable({ deltas, classNames }) {
  if (!deltas) return null;
  return (
    <div className="mb-4">
      <table className="w-full text-sm font-mono">
        <thead>
          <tr className="text-left text-muted text-[11px] uppercase tracking-wide">
            <th className="pb-2 font-display font-normal">Feature</th>
            <th className="pb-2 font-display font-normal">Original</th>
            <th className="pb-2 font-display font-normal">Change</th>
          </tr>
        </thead>
        <tbody>
          {deltas.map((row) => (
            <tr key={row.feature} className="border-t border-border">
              <td className="py-1.5 text-ink">{row.feature}</td>
              <td className="py-1.5 text-muted">{row.original_value}</td>
              <td className={`py-1.5 ${row.significant ? "text-amber" : "text-muted"}`}>
                {row.change > 0 ? "+" : ""}
                {row.change}
                {row.significant ? " ⚠" : ""}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

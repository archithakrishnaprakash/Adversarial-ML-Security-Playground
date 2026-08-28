"use client";

import { useEffect, useState } from "react";
import PageHeader from "@/components/PageHeader";
import Panel from "@/components/Panel";
import Select from "@/components/Select";
import Button from "@/components/Button";
import Pill from "@/components/Pill";
import api from "@/lib/api";

const MODEL_TYPES = ["logistic_regression", "random_forest", "small_nn", "cnn"];
const ATTACKS = ["fgsm", "pgd", "deepfool", "random_noise"];
const DEFENSES = ["none", "gaussian_smoothing", "feature_clipping", "normalization"];
const EPSILON_CHOICES = [0.03, 0.05, 0.1, 0.15, 0.2, 0.3];

export default function ExperimentsPage() {
  const [datasets, setDatasets] = useState([]);
  const [dataset, setDataset] = useState("cyber");
  const [modelTypes, setModelTypes] = useState(["logistic_regression"]);
  const [attacks, setAttacks] = useState(["fgsm"]);
  const [epsilons, setEpsilons] = useState([0.1, 0.15]);
  const [defenses, setDefenses] = useState(["none"]);
  const [limits, setLimits] = useState(null);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.getDatasets().then((r) => setDatasets(r.datasets));
    api.getExperimentGridLimits().then((r) => setLimits(r));
  }, []);

  const currentDataset = datasets.find((d) => d.id === dataset);
  const validModelTypes = currentDataset?.valid_models || MODEL_TYPES;

  const toggle = (list, setList, value) =>
    setList(list.includes(value) ? list.filter((x) => x !== value) : [...list, value]);

  const nConfigs = modelTypes.length * attacks.length * epsilons.length * defenses.length;
  const overLimit = limits && nConfigs > limits.max_configurations;

  const handleRun = async () => {
    setRunning(true);
    setError(null);
    try {
      const res = await api.runExperimentGrid({
        dataset,
        model_types: modelTypes,
        attacks,
        epsilons,
        defenses,
        n_samples: 150,
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
        eyebrow="Automation"
        title="Experiment Runner"
        description="Sweep every combination of model x attack x epsilon x defense in one run instead of clicking through Attack Lab one configuration at a time. Every configuration is logged individually and reproducible later."
      />

      <div className="grid grid-cols-3 gap-6">
        <Panel eyebrow="Grid configuration" title="Sweep" className="col-span-1 h-fit">
          <div className="space-y-4">
            <Select
              label="Dataset"
              value={dataset}
              onChange={setDataset}
              options={datasets.map((d) => ({ value: d.id, label: d.name }))}
            />

            <CheckboxGroup
              label="Model architectures"
              options={validModelTypes}
              selected={modelTypes}
              onToggle={(v) => toggle(modelTypes, setModelTypes, v)}
            />
            <CheckboxGroup
              label="Attacks"
              options={ATTACKS}
              selected={attacks}
              onToggle={(v) => toggle(attacks, setAttacks, v)}
            />
            <CheckboxGroup
              label="Epsilon values"
              options={EPSILON_CHOICES.map(String)}
              selected={epsilons.map(String)}
              onToggle={(v) => toggle(epsilons, setEpsilons, Number(v))}
            />
            <CheckboxGroup
              label="Defenses"
              options={DEFENSES}
              selected={defenses}
              onToggle={(v) => toggle(defenses, setDefenses, v)}
            />

            <div className="flex items-center justify-between text-xs font-mono pt-2 border-t border-border">
              <span className="text-muted">Configurations</span>
              <span className={overLimit ? "text-red" : "text-ink"}>
                {nConfigs} {limits ? `/ ${limits.max_configurations} max` : ""}
              </span>
            </div>

            <Button
              onClick={handleRun}
              disabled={running || nConfigs === 0 || overLimit}
              className="w-full"
            >
              {running ? "Running grid…" : "Run experiment grid"}
            </Button>
            {overLimit && (
              <p className="text-[11px] text-red leading-relaxed">
                This grid runs synchronously in a single request — reduce the number of selected
                options to stay under the cap, or run it in smaller batches.
              </p>
            )}
            {error && <p className="text-xs text-red font-mono">{error}</p>}
          </div>
        </Panel>

        <div className="col-span-2">
          {!result && (
            <Panel>
              <p className="text-sm text-muted font-mono">
                Configure a grid on the left and run it to see every result here.
              </p>
            </Panel>
          )}

          {result && (
            <Panel
              eyebrow="Results"
              title={`${result.n_configurations_run} configurations`}
              right={<Pill tone="cyan">{result.dataset}</Pill>}
            >
              <div className="overflow-x-auto">
                <table className="w-full text-sm font-mono">
                  <thead>
                    <tr className="text-left text-muted text-[11px] uppercase tracking-wide">
                      <th className="pb-2 font-display font-normal">Model</th>
                      <th className="pb-2 font-display font-normal">Attack</th>
                      <th className="pb-2 font-display font-normal">ε</th>
                      <th className="pb-2 font-display font-normal">Defense</th>
                      <th className="pb-2 font-display font-normal">Clean</th>
                      <th className="pb-2 font-display font-normal">Attacked</th>
                      <th className="pb-2 font-display font-normal">Defended</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.records.map((r, i) => (
                      <tr key={i} className="border-t border-border">
                        <td className="py-1.5 text-ink">{r.model_type}</td>
                        <td className="py-1.5 text-ink">{r.attack || r.error ? r.attack ?? "—" : "—"}</td>
                        <td className="py-1.5 text-muted">{r.epsilon ?? "—"}</td>
                        <td className="py-1.5 text-muted">{r.defense ?? "—"}</td>
                        {r.error ? (
                          <td colSpan={3} className="py-1.5 text-red">
                            {r.error}
                          </td>
                        ) : (
                          <>
                            <td className="py-1.5 text-cyan">
                              {r.clean_accuracy != null ? `${Math.round(r.clean_accuracy * 100)}%` : "—"}
                            </td>
                            <td className="py-1.5 text-red">
                              {r.adversarial_accuracy != null
                                ? `${Math.round(r.adversarial_accuracy * 100)}%`
                                : "—"}
                            </td>
                            <td className="py-1.5 text-green">
                              {r.defended_accuracy != null
                                ? `${Math.round(r.defended_accuracy * 100)}%`
                                : "—"}
                            </td>
                          </>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Panel>
          )}
        </div>
      </div>
    </div>
  );
}

function CheckboxGroup({ label, options, selected, onToggle }) {
  return (
    <div>
      <span className="block font-display text-[10px] tracking-widest text-muted uppercase mb-1.5">
        {label}
      </span>
      <div className="space-y-1.5">
        {options.map((opt) => (
          <label key={opt} className="flex items-center gap-2 text-sm font-mono text-ink">
            <input
              type="checkbox"
              checked={selected.includes(opt)}
              onChange={() => onToggle(opt)}
              className="accent-cyan"
            />
            {opt}
          </label>
        ))}
      </div>
    </div>
  );
}

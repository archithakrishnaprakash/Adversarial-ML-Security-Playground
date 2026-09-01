"use client";

import { useEffect, useState } from "react";
import PageHeader from "@/components/PageHeader";
import Panel from "@/components/Panel";
import Select from "@/components/Select";
import Button from "@/components/Button";
import MetricStat from "@/components/MetricStat";
import Pill from "@/components/Pill";
import api from "@/lib/api";

const MODEL_LABEL = {
  logistic_regression: "Logistic Regression",
  random_forest: "Random Forest",
  small_nn: "Small NN",
  cnn: "CNN",
};

export default function PoisoningLabPage() {
  const [datasets, setDatasets] = useState([]);

  useEffect(() => {
    api.getDatasets().then((r) => setDatasets(r.datasets));
  }, []);

  return (
    <div>
      <PageHeader
        eyebrow="Training-time attacks"
        title="Poisoning Lab"
        description="Evasion attacks (Attack Lab) tamper with an input at inference time. These tamper with the training data itself, so the model learns something wrong from the start."
      />
      <div className="grid grid-cols-2 gap-6 items-start">
        <LabelFlipCard datasets={datasets} />
        <BackdoorCard datasets={datasets} />
      </div>
    </div>
  );
}

function useDatasetModelPicker(datasets) {
  const [dataset, setDataset] = useState("cyber");
  const [modelType, setModelType] = useState("logistic_regression");
  const currentDataset = datasets.find((d) => d.id === dataset);
  const validModelTypes = currentDataset?.valid_models || [];

  useEffect(() => {
    if (validModelTypes.length && !validModelTypes.includes(modelType)) {
      setModelType(validModelTypes[0]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset, datasets]);

  return { dataset, setDataset, modelType, setModelType, validModelTypes };
}

function LabelFlipCard({ datasets }) {
  const { dataset, setDataset, modelType, setModelType, validModelTypes } =
    useDatasetModelPicker(datasets);
  const [poisonFraction, setPoisonFraction] = useState(0.05);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleRun = async () => {
    setRunning(true);
    setError(null);
    try {
      const res = await api.runLabelFlipPoisoning({
        dataset,
        model_type: modelType,
        poison_fraction: Number(poisonFraction),
      });
      setResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setRunning(false);
    }
  };

  return (
    <Panel eyebrow="Label flipping" title="Flip training labels">
      <p className="text-sm text-muted mb-4">
        Relabels a fraction of the training set to a random wrong class, then trains a fresh model
        on the tampered data. Trains a clean baseline alongside it so the gap is directly
        comparable.
      </p>
      <div className="space-y-4 mb-5">
        <div className="grid grid-cols-2 gap-3">
          <Select
            label="Dataset"
            value={dataset}
            onChange={setDataset}
            options={datasets.map((d) => ({ value: d.id, label: d.name }))}
          />
          <Select
            label="Model"
            value={modelType}
            onChange={setModelType}
            options={validModelTypes.map((t) => ({ value: t, label: MODEL_LABEL[t] || t }))}
          />
        </div>
        <div>
          <div className="flex justify-between items-baseline mb-1.5">
            <span className="font-display text-[10px] tracking-widest text-muted uppercase">
              Poison fraction
            </span>
            <span className="font-mono text-sm text-ink">{Math.round(poisonFraction * 100)}%</span>
          </div>
          <input
            type="range"
            min="0.01"
            max="0.5"
            step="0.01"
            value={poisonFraction}
            onChange={(e) => setPoisonFraction(e.target.value)}
            className="w-full accent-cyan"
          />
        </div>
        <Button onClick={handleRun} disabled={running} className="w-full">
          {running ? "Training clean + poisoned models…" : "Run label-flip experiment"}
        </Button>
        {error && <p className="text-xs text-red font-mono">{error}</p>}
      </div>

      {result && (
        <div className="space-y-3 pt-4 border-t border-border">
          <div className="grid grid-cols-3 gap-4">
            <MetricStat
              label="Clean model acc."
              value={`${Math.round(result.clean_accuracy_before_poisoning * 100)}%`}
              tone="green"
            />
            <MetricStat
              label="Poisoned model acc."
              value={`${Math.round(result.clean_accuracy_after_poisoning * 100)}%`}
              tone="red"
            />
            <MetricStat
              label="Degradation"
              value={`${Math.round(result.accuracy_degradation * 100)} pts`}
              tone="amber"
            />
          </div>
          <p className="text-[11px] text-muted font-mono">
            {result.n_poisoned_samples} samples relabeled · poisoned model #{result.poisoned_model_id}
          </p>
        </div>
      )}
    </Panel>
  );
}

function BackdoorCard({ datasets }) {
  const { dataset, setDataset, modelType, setModelType, validModelTypes } =
    useDatasetModelPicker(datasets);
  const [poisonFraction, setPoisonFraction] = useState(0.1);
  const [targetLabel, setTargetLabel] = useState(0);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const currentDataset = datasets.find((d) => d.id === dataset);
  const numClasses = currentDataset?.num_classes || 2;
  const classNames = currentDataset?.class_names;

  const handleRun = async () => {
    setRunning(true);
    setError(null);
    try {
      const res = await api.runBackdoorPoisoning({
        dataset,
        model_type: modelType,
        poison_fraction: Number(poisonFraction),
        target_label: Number(targetLabel),
      });
      setResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setRunning(false);
    }
  };

  return (
    <Panel eyebrow="Backdoor trigger" title="Inject a hidden trigger">
      <p className="text-sm text-muted mb-4">
        Stamps a fixed pattern onto a fraction of training samples and relabels only those to a
        target class. A model trained on this learns to associate the trigger with the target
        label, while behaving normally otherwise — invisible until the trigger appears.
      </p>
      <div className="space-y-4 mb-5">
        <div className="grid grid-cols-2 gap-3">
          <Select
            label="Dataset"
            value={dataset}
            onChange={setDataset}
            options={datasets.map((d) => ({ value: d.id, label: d.name }))}
          />
          <Select
            label="Model"
            value={modelType}
            onChange={setModelType}
            options={validModelTypes.map((t) => ({ value: t, label: MODEL_LABEL[t] || t }))}
          />
        </div>
        <Select
          label="Target label"
          value={String(targetLabel)}
          onChange={(v) => setTargetLabel(Number(v))}
          options={Array.from({ length: numClasses }, (_, i) => ({
            value: String(i),
            label: classNames ? classNames[i] : `Class ${i}`,
          }))}
        />
        <div>
          <div className="flex justify-between items-baseline mb-1.5">
            <span className="font-display text-[10px] tracking-widest text-muted uppercase">
              Poison fraction
            </span>
            <span className="font-mono text-sm text-ink">{Math.round(poisonFraction * 100)}%</span>
          </div>
          <input
            type="range"
            min="0.01"
            max="0.5"
            step="0.01"
            value={poisonFraction}
            onChange={(e) => setPoisonFraction(e.target.value)}
            className="w-full accent-cyan"
          />
        </div>
        <Button variant="danger" onClick={handleRun} disabled={running} className="w-full">
          {running ? "Training clean + backdoored models…" : "Run backdoor experiment"}
        </Button>
        {error && <p className="text-xs text-red font-mono">{error}</p>}
      </div>

      {result && (
        <div className="space-y-3 pt-4 border-t border-border">
          <div className="grid grid-cols-3 gap-4">
            <MetricStat
              label="Clean acc. (backdoored model)"
              value={`${Math.round(result.clean_accuracy_after_poisoning * 100)}%`}
              tone="green"
            />
            <MetricStat
              label="Backdoor success"
              value={`${Math.round(result.backdoor_success_rate * 100)}%`}
              tone="red"
            />
            <div className="flex items-center">
              <Pill tone={result.backdoor_success_rate > 0.5 ? "red" : "amber"}>
                {result.backdoor_success_rate > 0.5 ? "trigger works" : "weak trigger"}
              </Pill>
            </div>
          </div>
          <p className="text-[11px] text-muted font-mono">
            Clean accuracy barely moved while the trigger reliably flips predictions — that's
            what makes backdoors hard to spot from accuracy alone.
          </p>
        </div>
      )}
    </Panel>
  );
}

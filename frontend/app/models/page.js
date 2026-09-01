"use client";

import { useEffect, useState } from "react";
import PageHeader from "@/components/PageHeader";
import Panel from "@/components/Panel";
import Select from "@/components/Select";
import Button from "@/components/Button";
import Pill from "@/components/Pill";
import MetricStat from "@/components/MetricStat";
import api from "@/lib/api";

const MODEL_LABEL = {
  logistic_regression: "Logistic Regression",
  random_forest: "Random Forest",
  small_nn: "Small Neural Network",
  cnn: "CNN",
};

export default function ModelsPage() {
  const [datasets, setDatasets] = useState([]);
  const [dataset, setDataset] = useState("image");
  const [modelType, setModelType] = useState("cnn");
  const [models, setModels] = useState([]);
  const [training, setTraining] = useState(false);
  const [error, setError] = useState(null);

  const refreshModels = () => api.listModels().then((r) => setModels(r.models));

  useEffect(() => {
    api.getDatasets().then((r) => setDatasets(r.datasets));
    refreshModels();
  }, []);

  const currentDataset = datasets.find((d) => d.id === dataset);
  const validModelTypes = currentDataset?.valid_models || [];

  useEffect(() => {
    if (validModelTypes.length && !validModelTypes.includes(modelType)) {
      setModelType(validModelTypes[0]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset, datasets]);

  const handleTrain = async () => {
    setTraining(true);
    setError(null);
    try {
      await api.trainModel(dataset, modelType);
      await refreshModels();
    } catch (err) {
      setError(err.message);
    } finally {
      setTraining(false);
    }
  };

  return (
    <div>
      <PageHeader
        eyebrow="Step 1"
        title="Train a model"
        description="Pick a dataset and an architecture. Training runs live in the backend and usually takes a few seconds."
      />

      <div className="grid grid-cols-3 gap-6">
        <Panel eyebrow="Configuration" title="New model" className="col-span-1 h-fit">
          <div className="space-y-4">
            <Select
              label="Dataset"
              value={dataset}
              onChange={setDataset}
              options={datasets.map((d) => ({
                value: d.id,
                label: d.available === false ? `${d.name} (needs setup)` : d.name,
              }))}
            />
            {currentDataset?.available === false && (
              <p className="text-[11px] text-amber leading-relaxed">
                {currentDataset.note}
              </p>
            )}
            {currentDataset?.available !== false && currentDataset?.note && (
              <p className="text-[11px] text-muted leading-relaxed">{currentDataset.note}</p>
            )}
            <Select
              label="Model architecture"
              value={modelType}
              onChange={setModelType}
              options={validModelTypes.map((t) => ({ value: t, label: MODEL_LABEL[t] || t }))}
            />
            <Button
              onClick={handleTrain}
              disabled={training || currentDataset?.available === false}
              className="w-full"
            >
              {training ? "Training…" : "Train model"}
            </Button>
            {error && <p className="text-xs text-red font-mono">{error}</p>}
            <p className="text-[11px] text-muted leading-relaxed">
              Random Forest is non-differentiable, so a small surrogate network is trained
              alongside it automatically — that's what gradient attacks target and transfer
              results back onto the real forest.
            </p>
          </div>
        </Panel>

        <Panel eyebrow="Registry" title={`Trained models (${models.length})`} className="col-span-2">
          {models.length === 0 ? (
            <p className="text-sm text-muted font-mono">No models trained yet.</p>
          ) : (
            <div className="space-y-3">
              {models.map((m) => (
                <div key={m.model_id} className="border border-border p-3">
                  <div className="flex items-center justify-between mb-3">
                    <div>
                      <div className="font-mono text-sm text-ink">
                        {MODEL_LABEL[m.model_type] || m.model_type}{" "}
                        <span className="text-muted">#{m.model_id}</span>
                      </div>
                      <div className="text-[11px] text-muted mt-0.5">{m.dataset}</div>
                    </div>
                    <div className="flex gap-2">
                      {m.metrics?.adversarially_trained && <Pill tone="green">robust</Pill>}
                      {!m.is_differentiable && <Pill tone="amber">surrogate</Pill>}
                    </div>
                  </div>
                  <div className="grid grid-cols-3 gap-4">
                    <MetricStat
                      label="Clean accuracy"
                      value={
                        m.metrics?.clean_accuracy != null
                          ? `${Math.round(m.metrics.clean_accuracy * 100)}%`
                          : "—"
                      }
                      tone="green"
                    />
                    <MetricStat label="Train samples" value={m.metrics?.n_train ?? "—"} />
                    <MetricStat label="Test samples" value={m.metrics?.n_test ?? "—"} />
                  </div>
                </div>
              ))}
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}

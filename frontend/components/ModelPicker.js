import Pill from "./Pill";

const DATASET_LABEL = { image: "Digits (image)", cyber: "Network Intrusion" };
const MODEL_LABEL = {
  logistic_regression: "Logistic Regression",
  random_forest: "Random Forest",
  small_nn: "Small NN",
  cnn: "CNN",
};

export default function ModelPicker({ models, value, onChange, emptyHint }) {
  if (!models || models.length === 0) {
    return (
      <div className="border border-dashed border-border p-4 text-sm text-muted font-mono">
        {emptyHint || "No trained models yet. Go to Models and train one first."}
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {models.map((m) => {
        const active = m.model_id === value;
        return (
          <button
            key={m.model_id}
            onClick={() => onChange(m.model_id)}
            className={`w-full text-left px-3 py-2.5 border transition-colors flex items-center justify-between gap-3 ${
              active ? "border-cyan bg-panel2" : "border-border hover:border-muted"
            }`}
          >
            <div>
              <div className="font-mono text-sm text-ink">
                {MODEL_LABEL[m.model_type] || m.model_type}{" "}
                <span className="text-muted">#{m.model_id}</span>
              </div>
              <div className="text-[11px] text-muted mt-0.5">
                {DATASET_LABEL[m.dataset] || m.dataset}
              </div>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              {m.metrics?.clean_accuracy != null && (
                <Pill tone="green">{Math.round(m.metrics.clean_accuracy * 100)}% acc</Pill>
              )}
              {!m.is_differentiable && <Pill tone="amber">surrogate-attacked</Pill>}
            </div>
          </button>
        );
      })}
    </div>
  );
}

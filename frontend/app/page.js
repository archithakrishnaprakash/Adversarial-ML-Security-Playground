"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import PageHeader from "@/components/PageHeader";
import Panel from "@/components/Panel";
import MetricStat from "@/components/MetricStat";
import Pill from "@/components/Pill";
import api from "@/lib/api";

export default function DashboardPage() {
  const [models, setModels] = useState([]);
  const [experiments, setExperiments] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([api.listModels(), api.listExperiments(10), api.getDatasets()])
      .then(([m, e, d]) => {
        setModels(m.models);
        setExperiments(e.experiments);
        setDatasets(d.datasets);
      })
      .catch((err) => setError(err.message));
  }, []);

  const realIds = datasets.find((d) => d.id === "real_ids");

  return (
    <div>
      <PageHeader
        eyebrow="Overview"
        title="AegisML — Adversarial ML Security Assessment Framework"
        description="Train models, attack them with FGSM / PGD / DeepFool / poisoning / black-box methods, harden them with defenses, and get a risk-rated security assessment — not just a score."
      />

      {error && (
        <Panel className="mb-6 border-red">
          <p className="text-sm text-red font-mono">
            Can't reach the backend at the configured API URL — {error}. Make sure the FastAPI
            server is running (see README).
          </p>
        </Panel>
      )}

      <div className="grid grid-cols-3 gap-4 mb-6">
        <Panel>
          <MetricStat label="Trained models" value={models.length} tone="cyan" />
        </Panel>
        <Panel>
          <MetricStat label="Experiments logged" value={experiments.length} tone="green" />
        </Panel>
        <Panel>
          <MetricStat label="Attack methods" value="6" suffix=" available" tone="red" />
        </Panel>
      </div>

      <div className="grid grid-cols-3 gap-4 mb-6">
        <Panel eyebrow="Playground A" title="Computer Vision">
          <p className="text-sm text-muted mb-3">
            8×8 handwritten digit classification (an MNIST-style dataset bundled with
            scikit-learn — no download required). Attack pixels, watch predictions flip while
            the image barely changes to the human eye.
          </p>
          <Pill tone="cyan">10 classes</Pill>
        </Panel>
        <Panel eyebrow="Playground B" title="Cybersecurity ML — Quick-Start">
          <p className="text-sm text-muted mb-3">
            A synthetic network-intrusion dataset (BENIGN vs ATTACK) with realistic behavioural
            features. Zero setup — good for demonstrating the pipeline, not a substitute for
            real traffic.
          </p>
          <Pill tone="red">2 classes · synthetic</Pill>
        </Panel>
        <Panel eyebrow="Playground C" title="Cybersecurity ML — Real Data">
          <p className="text-sm text-muted mb-3">
            UNSW-NB15, a published intrusion-detection dataset, once you've downloaded it (see
            README §5b). Everything else in the app works identically against it.
          </p>
          <Pill tone={realIds?.available ? "green" : "amber"}>
            {realIds?.available ? "available" : "needs setup"}
          </Pill>
        </Panel>
      </div>

      <div className="grid grid-cols-4 gap-4 mb-8">
        {[
          { href: "/models", label: "1. Train a model", desc: "Pick a dataset + architecture" },
          { href: "/attack-lab", label: "2. Attack it", desc: "Threat models, evasion, black-box" },
          { href: "/defense-lab", label: "3. Defend it", desc: "Preprocessing + adv. training" },
          { href: "/security-assessment", label: "4. Assess it", desc: "Risk rating + findings" },
        ].map((step) => (
          <Link key={step.href} href={step.href}>
            <div className="border border-border bg-panel hover:border-cyan transition-colors p-4 h-full">
              <div className="font-display text-xs text-cyan mb-2">{step.label}</div>
              <div className="text-xs text-muted">{step.desc}</div>
            </div>
          </Link>
        ))}
      </div>

      <Panel eyebrow="Activity" title="Recent experiments">
        {experiments.length === 0 ? (
          <p className="text-sm text-muted font-mono">
            Nothing logged yet — run an attack or evaluation to see it here.
          </p>
        ) : (
          <div className="divide-y divide-border">
            {experiments.map((e) => (
              <div key={e.id} className="py-2.5 flex items-center justify-between text-sm">
                <div className="font-mono text-ink">
                  {e.experiment_type} <span className="text-muted">on</span> {e.model_type}{" "}
                  <span className="text-muted">#{e.model_id}</span>
                  {e.attack && <span className="text-muted"> · {e.attack}</span>}
                </div>
                <div className="text-[11px] text-muted font-mono">
                  {new Date(e.timestamp).toLocaleTimeString()}
                </div>
              </div>
            ))}
          </div>
        )}
      </Panel>
    </div>
  );
}

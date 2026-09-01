"use client";

import { useEffect, useState } from "react";
import PageHeader from "@/components/PageHeader";
import Panel from "@/components/Panel";
import Select from "@/components/Select";
import Pill from "@/components/Pill";
import api from "@/lib/api";

const SORT_OPTIONS = [
  { value: "robustness_score", label: "Aegis Robustness Index (high to low)" },
  { value: "clean_accuracy", label: "Clean accuracy (high to low)" },
  { value: "attack_success_rate", label: "Attack success rate (low to high)" },
];

const RISK_TONE = { HIGH: "red", MEDIUM: "amber", LOW: "green" };

export default function LeaderboardPage() {
  const [sortBy, setSortBy] = useState("robustness_score");
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    api
      .getLeaderboard(sortBy)
      .then((r) => setRows(r.leaderboard))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [sortBy]);

  const maxScore = Math.max(1, ...rows.map((r) => (r.aegis_robustness_index ?? r.robustness_score) ?? 0));

  return (
    <div>
      <PageHeader
        eyebrow="Ranking"
        title="Robustness Leaderboard"
        description="Ranks every model that's had at least one robustness evaluation run against it (Robustness Report page), by Aegis Robustness Index — a composite project metric, not a standardized benchmark. Train and evaluate more models to grow this list."
      />

      <Panel
        right={
          <div className="w-72">
            <Select value={sortBy} onChange={setSortBy} options={SORT_OPTIONS} />
          </div>
        }
      >
        {error && <p className="text-sm text-red font-mono mb-3">{error}</p>}
        {loading && <p className="text-sm text-muted font-mono">Loading…</p>}

        {!loading && rows.length === 0 && (
          <p className="text-sm text-muted font-mono">
            No models have been evaluated yet — run a robustness report first.
          </p>
        )}

        {!loading && rows.length > 0 && (
          <div className="space-y-2">
            {rows.map((row, i) => (
              <div key={row.model_id} className="border border-border p-3 flex items-center gap-4">
                <div className="font-display text-lg text-muted w-8 text-center">{i + 1}</div>
                <div className="flex-1">
                  <div className="font-mono text-sm text-ink">
                    {row.model_type} <span className="text-muted">#{row.model_id}</span>
                  </div>
                  <div className="text-[11px] text-muted mt-0.5">{row.dataset}</div>
                </div>
                <div className="w-40">
                  <div className="h-2 bg-panel2 border border-border">
                    <div
                      className={`h-full ${
                        row.risk_rating === "LOW"
                          ? "bg-green"
                          : row.risk_rating === "MEDIUM"
                          ? "bg-amber"
                          : "bg-red"
                      }`}
                      style={{
                        width: `${(((row.aegis_robustness_index ?? row.robustness_score) ?? 0) / maxScore) * 100}%`,
                      }}
                    />
                  </div>
                </div>
                <div className="font-mono text-sm text-ink w-16 text-right">
                  {row.aegis_robustness_index ?? row.robustness_score ?? "—"}
                </div>
                <Pill tone={RISK_TONE[row.risk_rating]}>{row.risk_rating}</Pill>
              </div>
            ))}
          </div>
        )}
      </Panel>
    </div>
  );
}

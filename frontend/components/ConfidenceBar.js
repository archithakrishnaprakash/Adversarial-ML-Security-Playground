export default function ConfidenceBar({ label, value, tone = "cyan" }) {
  const pct = Math.round(value * 100);
  const barColor = { cyan: "bg-cyan", red: "bg-red", amber: "bg-amber", green: "bg-green" }[tone];
  return (
    <div>
      <div className="flex justify-between items-baseline mb-1">
        <span className="font-mono text-xs text-muted">{label}</span>
        <span className="font-mono text-sm text-ink">{pct}%</span>
      </div>
      <div className="h-2 bg-panel2 border border-border">
        <div className={`h-full ${barColor}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

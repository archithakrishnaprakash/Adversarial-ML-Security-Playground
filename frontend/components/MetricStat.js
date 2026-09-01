export default function MetricStat({ label, value, tone = "ink", suffix = "" }) {
  const toneClass =
    {
      ink: "text-ink",
      cyan: "text-cyan",
      red: "text-red",
      amber: "text-amber",
      green: "text-green",
    }[tone] || "text-ink";

  return (
    <div>
      <div className="font-display text-[10px] tracking-widest text-muted uppercase mb-1">
        {label}
      </div>
      <div className={`font-mono text-2xl font-semibold ${toneClass}`}>
        {value}
        <span className="text-sm text-muted">{suffix}</span>
      </div>
    </div>
  );
}

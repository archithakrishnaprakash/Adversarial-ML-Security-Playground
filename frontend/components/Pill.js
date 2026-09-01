const TONES = {
  cyan: "text-cyan border-cyan/40 bg-cyan/10",
  red: "text-red border-red/40 bg-red/10",
  amber: "text-amber border-amber/40 bg-amber/10",
  green: "text-green border-green/40 bg-green/10",
  muted: "text-muted border-border bg-panel2",
};

export default function Pill({ children, tone = "muted" }) {
  return (
    <span
      className={`inline-block font-mono text-[11px] px-2 py-1 border ${TONES[tone] || TONES.muted}`}
    >
      {children}
    </span>
  );
}

export default function Select({ label, value, onChange, options, disabled = false }) {
  return (
    <label className="block">
      {label && (
        <span className="block font-display text-[10px] tracking-widest text-muted uppercase mb-1.5">
          {label}
        </span>
      )}
      <select
        className="w-full bg-panel2 border border-border text-ink text-sm px-3 py-2.5 font-mono focus:outline-none focus:border-cyan disabled:opacity-40"
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </label>
  );
}

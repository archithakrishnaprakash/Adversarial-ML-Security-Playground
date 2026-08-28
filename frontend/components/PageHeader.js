export default function PageHeader({ eyebrow, title, description }) {
  return (
    <div className="mb-8">
      {eyebrow && (
        <div className="font-display text-[11px] tracking-widest text-cyan uppercase mb-2">
          {eyebrow}
        </div>
      )}
      <h1 className="text-2xl font-bold text-ink font-display">{title}</h1>
      {description && <p className="text-sm text-muted mt-2 max-w-2xl">{description}</p>}
    </div>
  );
}

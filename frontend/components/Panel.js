export default function Panel({ title, eyebrow, right, children, className = "" }) {
  return (
    <div className={`border border-border bg-panel ${className}`}>
      {(title || eyebrow) && (
        <div className="flex items-center justify-between px-4 py-3 border-b border-border">
          <div>
            {eyebrow && (
              <div className="font-display text-[10px] tracking-widest text-muted uppercase mb-0.5">
                {eyebrow}
              </div>
            )}
            {title && <div className="font-semibold text-ink text-sm">{title}</div>}
          </div>
          {right}
        </div>
      )}
      <div className="p-4">{children}</div>
    </div>
  );
}

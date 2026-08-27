"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV = [
  { href: "/", label: "Dashboard", glyph: "◈" },
  { href: "/models", label: "Models", glyph: "◫" },
  { href: "/attack-lab", label: "Attack Lab", glyph: "⚔" },
  { href: "/defense-lab", label: "Defense Lab", glyph: "◆" },
  { href: "/robustness-report", label: "Robustness Report", glyph: "▤" },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-60 shrink-0 border-r border-border bg-panel min-h-screen flex flex-col">
      <div className="px-5 py-6 border-b border-border">
        <div className="flex items-center gap-2">
          <span className="status-dot bg-red pulse" />
          <span className="font-display text-[11px] tracking-widest text-muted uppercase">
            Security Lab
          </span>
        </div>
        <h1 className="font-display font-bold text-lg text-ink mt-1 leading-tight">
          Adversarial ML
          <br />
          Playground
        </h1>
      </div>

      <nav className="flex-1 py-4">
        {NAV.map((item) => {
          const active = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 px-5 py-3 text-sm font-medium border-l-2 transition-colors ${
                active
                  ? "border-cyan text-ink bg-panel2"
                  : "border-transparent text-muted hover:text-ink hover:bg-panel2/50"
              }`}
            >
              <span className="font-display text-base">{item.glyph}</span>
              {item.label}
            </Link>
          );
        })}
      </nav>

      <div className="px-5 py-4 border-t border-border text-[11px] text-muted font-mono leading-relaxed">
        MNIST-style digits (8×8)
        <br />
        Synthetic network intrusion
      </div>
    </aside>
  );
}

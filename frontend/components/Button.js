export default function Button({ children, variant = "primary", className = "", ...props }) {
  const base =
    "font-display text-xs tracking-wide uppercase px-4 py-2.5 border transition-colors disabled:opacity-40 disabled:cursor-not-allowed";
  const variants = {
    primary: "bg-cyan text-bg border-cyan hover:bg-cyan/90 font-bold",
    danger: "bg-red text-bg border-red hover:bg-red/90 font-bold",
    ghost: "bg-transparent text-ink border-border hover:border-cyan hover:text-cyan",
  };
  return (
    <button className={`${base} ${variants[variant]} ${className}`} {...props}>
      {children}
    </button>
  );
}

/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,jsx}",
    "./components/**/*.{js,jsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: "#0B0E14",
        panel: "#111826",
        panel2: "#161F2E",
        border: "#232D3D",
        muted: "#8B98A5",
        ink: "#E6EDF3",
        cyan: "#22D3EE",
        red: "#FF4D5E",
        amber: "#F5A623",
        green: "#34D399",
      },
      fontFamily: {
        mono: ["'IBM Plex Mono'", "ui-monospace", "SFMono-Regular", "monospace"],
        display: ["'JetBrains Mono'", "ui-monospace", "monospace"],
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};

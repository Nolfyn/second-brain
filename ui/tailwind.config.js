/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        // Base surfaces
        base:     "#0A0A0F",
        surface:  "#111116",
        elevated: "#18181F",
        border:   "#1E1E28",
        "border-bright": "#2A2A38",

        // Text
        primary:   "#FFFFFF",
        secondary: "#9CA3AF",
        muted:     "#4B5563",

        // Accent — periwinkle violet
        accent:       "#7C6FF7",
        "accent-dim": "#3D3778",
        "accent-glow":"#7C6FF720",

        // Semantic
        success: "#34D399",
        warn:    "#FBBF24",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
      },
      maxWidth: {
        chat: "800px",
      },
    },
  },
  plugins: [],
};
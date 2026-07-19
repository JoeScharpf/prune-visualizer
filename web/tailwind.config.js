/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        stone: {
          50: "#fafaf9",
          100: "#f5f5f4",
          200: "#e7e5e4",
          300: "#d6d3d1",
          400: "#a8a29e",
          500: "#78716c",
          600: "#57534e",
          700: "#44403c",
          800: "#292524",
          900: "#1c1917",
          950: "#0c0a09",
        },
        bg: {
          DEFAULT: "#ffffff",
          subtle: "#fafaf9",
        },
        fg: {
          DEFAULT: "#0c0a09",
          muted: "#78716c",
          invert: "#ffffff",
        },
        border: {
          DEFAULT: "#e7e5e4",
        },
        accent: {
          DEFAULT: "#ea580c",
        },
      },
      fontFamily: {
        sans: ['"Geist"', "Inter", "ui-sans-serif", "system-ui", "-apple-system", "sans-serif"],
        mono: ['"JetBrains Mono"', "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
    },
  },
  plugins: [],
};

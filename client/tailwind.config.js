/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#101828",
        mist: "#eef4ff",
        glow: "#dff6df",
        coral: "#ff8762",
        teal: "#0f766e",
        oat: "#fff9ef"
      },
      fontFamily: {
        sans: ["Sora", "ui-sans-serif", "system-ui"],
        display: ["Fraunces", "ui-serif", "Georgia"]
      },
      boxShadow: {
        float: "0 30px 80px rgba(16, 24, 40, 0.16)"
      },
      backgroundImage: {
        hero: "radial-gradient(circle at top left, rgba(255,255,255,0.96), rgba(223,246,223,0.76) 38%, rgba(206,232,255,0.84) 78%, rgba(255,249,239,0.96) 100%)"
      }
    }
  },
  plugins: []
};

/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx,js,jsx}"],
  theme: {
    extend: {
      borderRadius: { '2xl': '1rem' },
      boxShadow: { soft: '0 10px 20px rgba(0,0,0,0.04)' }
    },
  },
  plugins: [],
};

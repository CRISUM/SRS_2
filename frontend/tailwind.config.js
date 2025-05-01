// frontend/tailwind.config.js
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
    "./public/index.html"
  ],
  theme: {
    extend: {
      colors: {
        steam: {
          blue: '#1b2838',
          dark: '#171a21',
          light: '#66c0f4',
          green: '#5c7e10'
        }
      },
    },
  },
  plugins: [],
}
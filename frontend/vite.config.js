import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Backend runs on :8000; proxy /api there so the frontend can use relative
// URLs and we avoid CORS entirely in dev.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
});

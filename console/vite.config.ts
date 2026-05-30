import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: "/console/",
  server: {
    port: 5173,
    proxy: {
      "/ingest": { target: "http://localhost:8000", changeOrigin: true },
      "/retrieve": { target: "http://localhost:8000", changeOrigin: true },
      "/meta": { target: "http://localhost:8000", changeOrigin: true },
      "/tasks": { target: "http://localhost:8000", changeOrigin: true },
      "/system": { target: "http://localhost:8000", changeOrigin: true },
    },
  },
});

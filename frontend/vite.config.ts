import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
  },
  server: {
    port: 5173,
    proxy: {
      "/jobs": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
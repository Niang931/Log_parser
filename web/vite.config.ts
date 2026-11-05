import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` proxies the API to a `logpipe server` on :8080; `npm run build` writes the
// bundle where the server serves it from.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "../logpipe/server/static", emptyOutDir: true },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8080",
      "/intake": "http://127.0.0.1:8080",
      "/demo": "http://127.0.0.1:8080",
    },
  },
});

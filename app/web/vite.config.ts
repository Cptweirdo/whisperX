import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { createRequire } from "node:module";
import { cp, readFile } from "node:fs/promises";
import { dirname, join, normalize } from "node:path";

const require = createRequire(import.meta.url);

// Resolve Shoelace's shipped asset dir (icons loaded by name at runtime).
// package.json isn't in the package "exports", so resolve an exported dist file
// (the base-path utility our shoelace.ts imports) and derive dist/assets from it.
function shoelaceAssetsDir(): string {
  const basePath = require.resolve("@shoelace-style/shoelace/dist/utilities/base-path.js");
  // .../dist/utilities/base-path.js -> .../dist/assets
  return join(dirname(dirname(basePath)), "assets");
}

// sl-icon fetches Bootstrap Icons by name at runtime from <basePath>/assets/.
// They're invisible to the bundler, so: serve them in dev under /shoelace, and
// copy them into the build output under <outDir>/shoelace at build time.
// setBasePath(import.meta.env.BASE_URL + 'shoelace') resolves to both.
function shoelaceAssets() {
  const assets = shoelaceAssetsDir();
  return {
    name: "shoelace-assets",
    configureServer(server: any) {
      server.middlewares.use("/shoelace/assets", async (req: any, res: any, next: any) => {
        const rel = normalize(decodeURIComponent((req.url || "/").split("?")[0]));
        if (rel.includes("..")) return next();
        try {
          const body = await readFile(join(assets, rel));
          const type = rel.endsWith(".svg") ? "image/svg+xml" : "application/octet-stream";
          res.setHeader("Content-Type", type);
          res.end(body);
        } catch {
          next();
        }
      });
    },
    async writeBundle(this: any, options: any) {
      const out = options.dir ?? "dist";
      await cp(assets, join(out, "shoelace", "assets"), { recursive: true });
    },
  };
}

export default defineConfig(({ command }) => ({
  // Built assets are served by Flask from /static/spa/; dev runs at root.
  base: command === "build" ? "/static/spa/" : "/",
  plugins: [svelte(), shoelaceAssets()],
  build: {
    outDir: "../static/spa",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    // Proxy the Flask backend: JSON API, health, binary/SSE session subresources,
    // model/backup SSE, and the OAuth callback pages. The bare SPA routes
    // (`/`, `/onboarding`, `/sessions/:id`, `/settings`) are handled by Vite/the
    // client router — note the regex leaves `/sessions/:id` to the SPA but proxies
    // every `/sessions/:id/<subresource>` (audio, events, downloads).
    proxy: {
      "/api": "http://127.0.0.1:5000",
      "/healthz": "http://127.0.0.1:5000",
      "/models": "http://127.0.0.1:5000",
      "/backup": "http://127.0.0.1:5000",
      "/oauth": "http://127.0.0.1:5000",
      "^/sessions/[^/]+/.+": "http://127.0.0.1:5000",
    },
  },
  test: {
    environment: "happy-dom",
    globals: true,
    // Unit tests only; the Playwright e2e specs (tests/e2e/*.spec.ts) run via
    // `bun run test:e2e`, not vitest.
    include: ["tests/**/*.test.ts"],
  },
}));

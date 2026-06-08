import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { defineConfig, devices } from "@playwright/test";

// The e2e suite drives the real SPA against a seeded native (C++) backend
// (build/whisperx_server, served single-origin). serve.sh builds the SPA, seeds
// a demo session via the Python store, and runs the binary; PYTHON only needs to
// import app.store for the one-shot seed (stdlib) — e.g. PYTHON=../../.venv/bin/python.
const PORT = process.env.PORT || "5099";
const baseURL = `http://127.0.0.1:${PORT}`;

// Own the data dir here (config load runs before the webServer spawns, so this
// value reaches it via webServer.env — a value set in globalSetup would be too
// late). We mint+clean our own, but honour a caller-supplied WHISPERX_DATA_DIR
// (and then leave it alone). globalTeardown removes only what we created —
// needed because Playwright SIGKILLs the server group, so serve.sh's EXIT trap
// can't be relied on for teardown.
const callerDataDir = process.env.WHISPERX_DATA_DIR;
const dataDir = callerDataDir || mkdtempSync(join(tmpdir(), "wx-e2e-"));
process.env.WX_E2E_OWN_DATA_DIR = callerDataDir ? "" : dataDir;

// Where downloading Playwright's own browser is blocked, point at an existing
// Chromium: PW_CHROMIUM_PATH=/path/to/chrome.
const executablePath = process.env.PW_CHROMIUM_PATH || undefined;

export default defineConfig({
  testDir: "./tests/e2e",
  timeout: 30_000,
  expect: { timeout: 10_000 },
  fullyParallel: false,
  retries: 0,
  reporter: "list",
  globalTeardown: "./tests/e2e/global-teardown.ts",
  use: {
    baseURL,
    trace: "retain-on-failure",
    ...(executablePath ? { launchOptions: { executablePath } } : {}),
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: "bash tests/e2e/serve.sh",
    url: `${baseURL}/healthz`,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
    env: { PORT, WHISPERX_DATA_DIR: dataDir },
  },
});

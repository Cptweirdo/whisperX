import { defineConfig, devices } from "@playwright/test";

// The e2e suite drives the real SPA against a seeded Flask backend (served
// single-origin by Flask). serve.sh builds the SPA, seeds a demo session, and
// runs `python -m app.server`; set PYTHON to an interpreter with the web deps
// (flask/keyring) — e.g. PYTHON=../../.venv-web/bin/python in a dev checkout.
const PORT = process.env.PORT || "5099";
const baseURL = `http://127.0.0.1:${PORT}`;

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
    env: { PORT },
  },
});

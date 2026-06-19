import { rmSync } from "node:fs";

// Remove the temp data dir the config minted (WX_E2E_OWN_DATA_DIR is empty when
// the caller supplied their own WHISPERX_DATA_DIR — left untouched then). Runs
// in the Node runner after the suite, so it fires regardless of how the
// webServer died (Playwright hard-kills it, so serve.sh's trap can't be trusted).
export default function globalTeardown() {
  const dir = process.env.WX_E2E_OWN_DATA_DIR;
  if (dir) rmSync(dir, { recursive: true, force: true });
}

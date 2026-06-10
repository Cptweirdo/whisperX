// Cloud-backup state: live via the persistent /backup/status/events stream, with
// the non-blocking OAuth connect flow watched on /backup/events (ports the old
// watchBackupConnect). All payloads are JSON now (no server-rendered card).
import { api, urls } from "../api";
import { persistentSSE, sseStream } from "../sse";
import { notify } from "./toast.svelte";
import type { BackupStatus } from "../types";

class BackupStore {
  status = $state<BackupStatus | null>(null);
  connecting = $state(false);
  #stop: (() => void) | null = null;

  get linked(): boolean {
    return !!this.status?.linked;
  }
  get state(): string {
    return this.status?.state ?? "unlinked";
  }

  async load() {
    this.status = await api.backup.status();
  }

  start() {
    if (this.#stop) return;
    this.#stop = persistentSSE(urls.backupStatusEvents(), (d) => {
      if (d.status) this.status = d.status;
    });
  }

  async connect(folder?: string) {
    this.connecting = true;
    await api.backup.connect(folder || "");
    // Watch the one-shot OAuth result stream for the terminal event.
    sseStream(urls.backupEvents(), {
      onData: (d) => {
        if (d.status === "linked" || d.status === "error") {
          this.connecting = false;
          if (d.backup) this.status = d.backup;
          if (d.status === "error") notify(d.message || "Connection failed.", "danger", 6000);
        }
      },
      terminal: (d) => d.status === "linked" || d.status === "error",
    });
  }

  async disconnect() {
    this.status = await api.backup.disconnect();
  }
  async now() {
    await api.backup.now();
  }
  async restore() {
    const r = await api.backup.restore();
    if (r.backup) this.status = r.backup;
    return r.restored;
  }
  async adopt() {
    const r = await api.backup.adopt();
    if (r.backup) this.status = r.backup;
    return r.restored;
  }
  async overwrite() {
    const r = await api.backup.overwrite();
    if (r.backup) this.status = r.backup;
    return r;
  }
  remoteInfo() {
    return api.backup.remoteInfo();
  }
}

export const backup = new BackupStore();

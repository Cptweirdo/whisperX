// Cloud-backup state: live via the persistent /backup/status/events stream, with
// the non-blocking OAuth connect flow watched on /backup/events (ports the old
// watchBackupConnect). All payloads are JSON now (no server-rendered card).
import { api, urls } from "../api";
import { openSSE, sseStream } from "../sse";
import { notify } from "./toast.svelte";

class BackupStore {
  status = $state<any>(null);
  connecting = $state(false);
  #es: EventSource | null = null;

  get linked(): boolean {
    return !!this.status?.linked;
  }
  get state(): string {
    return this.status?.state ?? "unlinked";
  }

  async load() {
    this.status = await api.get("/backup/status");
  }

  start() {
    if (this.#es) return;
    this.#es = openSSE(urls.backupStatusEvents(), (d) => {
      if (d.status) this.status = d.status;
    });
  }

  async connect(folder?: string) {
    this.connecting = true;
    await api.post("/backup/connect", { backup_folder: folder || "" });
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
    this.status = await api.post("/backup/disconnect");
  }
  async now() {
    await api.post("/backup/now");
  }
  async restore() {
    const r = await api.post("/backup/restore");
    if (r.backup) this.status = r.backup;
    return r.restored as number;
  }
  async adopt() {
    const r = await api.post("/backup/bootstrap/adopt");
    if (r.backup) this.status = r.backup;
    return r.restored as number;
  }
  async overwrite() {
    const r = await api.post("/backup/bootstrap/overwrite");
    if (r.backup) this.status = r.backup;
    return r;
  }
  remoteInfo() {
    return api.get("/backup/remote-info");
  }
}

export const backup = new BackupStore();

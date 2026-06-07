// Typed fetch wrapper for the Flask JSON API. All data routes live under /api;
// binary/SSE routes are at root and have their own URL helpers below.

export class ApiError extends Error {
  status: number;
  body: any;
  constructor(status: number, body: any) {
    super(typeof body?.error === "string" ? body.error : `HTTP ${status}`);
    this.status = status;
    this.body = body;
  }
}

async function parse(res: Response): Promise<any> {
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) return res.json();
  const text = await res.text();
  return text ? { message: text } : {};
}

async function request(method: string, path: string, body?: any): Promise<any> {
  const opts: RequestInit = { method, headers: {} };
  if (body !== undefined) {
    (opts.headers as Record<string, string>)["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(`/api${path}`, opts);
  const data = await parse(res);
  if (!res.ok) throw new ApiError(res.status, data);
  return data;
}

export const api = {
  get: (path: string) => request("GET", path),
  post: (path: string, body?: any) => request("POST", path, body),
  del: (path: string) => request("POST", path), // delete endpoints are POSTs

  /** Multipart upload via XHR so we get upload progress (fetch has none).
   *  Resolves with the parsed JSON body; rejects with ApiError on non-2xx. */
  upload(
    path: string,
    form: FormData,
    onProgress?: (fraction: number) => void,
  ): Promise<any> {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `/api${path}`);
      xhr.responseType = "text";
      if (onProgress && xhr.upload) {
        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable) onProgress(e.loaded / e.total);
        };
      }
      xhr.onload = () => {
        let data: any = {};
        try {
          data = xhr.responseText ? JSON.parse(xhr.responseText) : {};
        } catch {
          data = { message: xhr.responseText };
        }
        if (xhr.status >= 200 && xhr.status < 300) resolve(data);
        else reject(new ApiError(xhr.status, data));
      };
      xhr.onerror = () => reject(new ApiError(0, { error: "network error" }));
      xhr.send(form);
    });
  },
};

// --- Root (non-/api) URL helpers for the browser to hit directly ------------
export const urls = {
  audio: (id: string) => `/sessions/${id}/audio`,
  download: (id: string, fmt: string) => `/sessions/${id}/download/${fmt}`,
  exportMd: (id: string) => `/sessions/${id}/export.md`,
  translationDownload: (id: string, lang: string, fmt: string) =>
    `/sessions/${id}/translation/${lang}/download/${fmt}`,
  sessionEvents: (id: string) => `/sessions/${id}/events`,
  translateEvents: (id: string) => `/sessions/${id}/translate/events`,
  modelsEvents: () => `/models/events`,
  backupStatusEvents: () => `/backup/status/events`,
  backupEvents: () => `/backup/events`,
};

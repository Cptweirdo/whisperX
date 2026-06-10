// Server-Sent Events helpers, ported from the old static/sse.js. The HTML-swap
// helpers (sseSwap / watchBackupConnect) are intentionally dropped — the SPA
// renders from JSON state, not server-rendered fragments.

/** Open an EventSource and hand each parsed JSON message to onData(data, es).
 *  Malformed frames are dropped; the browser auto-reconnects on transient
 *  errors. Returns the EventSource so callers can close() it. */
export function openSSE(url: string, onData: (data: any, es: EventSource) => void): EventSource {
  const es = new EventSource(url);
  es.onmessage = (e) => {
    let d: any;
    try {
      d = JSON.parse(e.data);
    } catch {
      return;
    }
    onData(d, es);
  };
  return es;
}

/** Open an app-lifetime stream that releases its connection while the tab is
 *  hidden and reopens on return. Browsers cap HTTP/1.1 at ~6 connections per
 *  host:port shared across ALL tabs, so ever-open EventSources in background tabs
 *  starve every other request (fetches queue forever). Both persistent
 *  endpoints (/models/events, /backup/status/events) replay full state as the
 *  first frame on connect, so a reopen self-refreshes. Returns stop(). */
export function persistentSSE(url: string, onData: (data: any) => void): () => void {
  let es: EventSource | null = null;
  const open = () => {
    if (!es) es = openSSE(url, (d) => onData(d));
  };
  const close = () => {
    es?.close();
    es = null;
  };
  const onVisibility = () => {
    if (document.hidden) close();
    else open();
  };
  document.addEventListener("visibilitychange", onVisibility);
  if (!document.hidden) open();
  return () => {
    document.removeEventListener("visibilitychange", onVisibility);
    close();
  };
}

export interface StreamOpts {
  onData: (data: any) => void;
  /** Optional predicate; when it returns true the stream is closed after the
   *  event is delivered (use for one-shot / terminal streams). */
  terminal?: (data: any) => boolean;
}

/** Open a stream, delivering each event to onData. If `terminal` is given, the
 *  stream auto-closes once it returns true. Returns a stop() to close early —
 *  ideal for `$effect(() => sseStream(...))` cleanup. */
export function sseStream(url: string, { onData, terminal }: StreamOpts): () => void {
  const es = openSSE(url, (d) => {
    onData(d);
    if (terminal && terminal(d)) es.close();
  });
  return () => es.close();
}

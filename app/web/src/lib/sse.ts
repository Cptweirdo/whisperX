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

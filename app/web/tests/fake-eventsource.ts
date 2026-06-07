// Minimal controllable EventSource stand-in for unit tests (ported from the old
// app/tests). Install via `globalThis.EventSource = FakeEventSource` and drive it
// with `.emit(obj)` / `.emitRaw(str)`.
export class FakeEventSource {
  static last: FakeEventSource | null = null;
  url: string;
  onmessage: ((e: { data: string }) => void) | null = null;
  onerror: ((e: unknown) => void) | null = null;
  closed = false;

  constructor(url: string) {
    this.url = url;
    FakeEventSource.last = this;
  }
  emit(obj: unknown) {
    this.onmessage?.({ data: JSON.stringify(obj) });
  }
  emitRaw(data: string) {
    this.onmessage?.({ data });
  }
  close() {
    this.closed = true;
  }
}

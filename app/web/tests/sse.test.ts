import { beforeEach, describe, expect, it } from "vitest";
import { FakeEventSource } from "./fake-eventsource";
import { openSSE, sseStream } from "../src/lib/sse";

beforeEach(() => {
  (globalThis as any).EventSource = FakeEventSource;
  FakeEventSource.last = null;
});

describe("openSSE", () => {
  it("parses JSON frames and delivers them", () => {
    const seen: any[] = [];
    openSSE("/x", (d) => seen.push(d));
    FakeEventSource.last!.emit({ stage: "transcribing", eta: 12 });
    expect(seen).toEqual([{ stage: "transcribing", eta: 12 }]);
  });

  it("drops malformed frames without throwing", () => {
    const seen: any[] = [];
    openSSE("/x", (d) => seen.push(d));
    FakeEventSource.last!.emitRaw("not json");
    FakeEventSource.last!.emit({ ok: true });
    expect(seen).toEqual([{ ok: true }]);
  });
});

describe("sseStream", () => {
  it("auto-closes once terminal returns true", () => {
    const seen: any[] = [];
    sseStream("/s", {
      onData: (d) => seen.push(d),
      terminal: (d) => d.status === "done",
    });
    const es = FakeEventSource.last!;
    es.emit({ stage: "aligning" });
    expect(es.closed).toBe(false);
    es.emit({ status: "done" });
    expect(es.closed).toBe(true);
    expect(seen).toEqual([{ stage: "aligning" }, { status: "done" }]);
  });

  it("stop() closes the stream", () => {
    const stop = sseStream("/s", { onData: () => {} });
    const es = FakeEventSource.last!;
    stop();
    expect(es.closed).toBe(true);
  });
});

import { beforeEach, describe, expect, it } from "vitest";
import { FakeEventSource } from "./fake-eventsource";
import { openSSE, persistentSSE, sseStream } from "../src/lib/sse";

beforeEach(() => {
  (globalThis as any).EventSource = FakeEventSource;
  FakeEventSource.last = null;
  setHidden(false);
});

function setHidden(hidden: boolean) {
  Object.defineProperty(document, "hidden", { value: hidden, configurable: true });
}

function fireVisibility(hidden: boolean) {
  setHidden(hidden);
  document.dispatchEvent(new Event("visibilitychange"));
}

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

describe("persistentSSE", () => {
  it("delivers parsed frames while visible", () => {
    const seen: any[] = [];
    const stop = persistentSSE("/p", (d) => seen.push(d));
    FakeEventSource.last!.emit({ active: "small" });
    expect(seen).toEqual([{ active: "small" }]);
    stop();
  });

  it("releases the connection when the tab hides and reopens on return", () => {
    const stop = persistentSSE("/p", () => {});
    const first = FakeEventSource.last!;
    fireVisibility(true);
    expect(first.closed).toBe(true);
    fireVisibility(false);
    const second = FakeEventSource.last!;
    expect(second).not.toBe(first);
    expect(second.url).toBe("/p");
    expect(second.closed).toBe(false);
    stop();
  });

  it("does not open at all when constructed in a hidden tab", () => {
    setHidden(true);
    const stop = persistentSSE("/p", () => {});
    expect(FakeEventSource.last).toBeNull();
    fireVisibility(false);
    expect(FakeEventSource.last!.url).toBe("/p");
    stop();
  });

  it("stop() closes the stream and detaches the visibility listener", () => {
    const stop = persistentSSE("/p", () => {});
    const es = FakeEventSource.last!;
    stop();
    expect(es.closed).toBe(true);
    // A later visibility flip must not resurrect the stream.
    fireVisibility(true);
    fireVisibility(false);
    expect(FakeEventSource.last).toBe(es);
  });
});

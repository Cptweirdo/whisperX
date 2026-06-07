import { describe, expect, it } from "vitest";
import { fmtClock, fmtDuration, fmtTs, speakerLabel } from "../src/lib/format";

describe("fmtDuration", () => {
  it("formats sub-hour as Xm SSs", () => {
    expect(fmtDuration(83)).toBe("1m 23s");
    expect(fmtDuration(5)).toBe("0m 05s");
  });
  it("formats hours as Xh MMm", () => {
    expect(fmtDuration(3700)).toBe("1h 01m");
  });
  it("treats null/0 as 0m 00s", () => {
    expect(fmtDuration(null)).toBe("0m 00s");
  });
});

describe("fmtClock / fmtTs", () => {
  it("m:ss under an hour", () => {
    expect(fmtClock(75)).toBe("1:15");
    expect(fmtTs(5)).toBe("0:05");
  });
  it("h:mm:ss over an hour", () => {
    expect(fmtClock(3725)).toBe("1:02:05");
  });
  it("fmtTs returns --:-- for null", () => {
    expect(fmtTs(null)).toBe("--:--");
  });
});

describe("speakerLabel", () => {
  it("maps SPEAKER_00 -> Speaker 1", () => {
    expect(speakerLabel("SPEAKER_00")).toBe("Speaker 1");
    expect(speakerLabel("SPEAKER_03")).toBe("Speaker 4");
  });
  it("passes custom labels through", () => {
    expect(speakerLabel("Alice")).toBe("Alice");
    expect(speakerLabel(null)).toBe("Speaker");
  });
});

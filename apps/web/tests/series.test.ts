import { describe, expect, it } from "vitest";

import type { DatasetSummary } from "../src/lib/api";
import { seriesName, seriesNameOf } from "../src/lib/series";

const summary = (id: string, label: string | null): DatasetSummary => ({
  id,
  label,
  n_frames: 1,
  processed: true,
  conditions_set: true,
  frame_px: null,
  dt_frame_ms: null,
});

describe("seriesName", () => {
  it("shows the label when set", () => {
    expect(seriesName(summary("ds_a", "High-T FC-72"))).toBe("High-T FC-72");
  });

  it("falls back to the id when there is no label", () => {
    expect(seriesName(summary("ds_a", null))).toBe("ds_a");
  });
});

describe("seriesNameOf", () => {
  const datasets = [summary("ds_a", "High-T FC-72"), summary("ds_b", null)];

  it("resolves a bare id to the matching series' current label", () => {
    expect(seriesNameOf(datasets, "ds_a")).toBe("High-T FC-72");
  });

  it("returns the id when the matching series has no label", () => {
    expect(seriesNameOf(datasets, "ds_b")).toBe("ds_b");
  });

  it("falls back to the id when the series isn't in the list", () => {
    expect(seriesNameOf(datasets, "ghost")).toBe("ghost");
    expect(seriesNameOf([], "ds_a")).toBe("ds_a");
  });
});

/** The Results rail: what a run's row leads with, how experiments fold, and
 * which rows may honestly be ranked against each other. */

import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { RunSummary } from "../src/lib/api";
import { runHeadline, runProvenance } from "../src/views/results/format";
import { buildFamilies } from "../src/views/results/runFamilies";
import { RunRail } from "../src/views/results/RunRail";

const run = (over: Partial<RunSummary> & { id: string }): RunSummary => ({
  dataset: "Series-1",
  status: "trained",
  steps: 3000,
  iou_holdout: null,
  datasets: ["Series-1"],
  heldout_datasets: [],
  date: "2026-08-03T03:15:00+00:00",
  n_frames: 11,
  recipe: ["sharp"],
  ...over,
});

/** The project's real shape: two seeds of one experiment, two singletons, and
 * one run evaluated over a different frame count. */
const RUNS: RunSummary[] = [
  run({ id: "fvb-fix-s0", iou_val: 0.895, iou_mean: 0.9287, seed: 0 }),
  run({ id: "fvb-fix-s1", iou_val: 0.889, iou_mean: 0.9262, seed: 1 }),
  run({
    id: "r4-causal-s2",
    iou_val: 0.8733,
    iou_mean: 0.9377,
    seed: 0,
    recipe: ["sharp", "causal"],
    date: "2026-08-02T00:12:00+00:00",
  }),
  run({
    id: "r4-gate2-s2",
    iou_val: 0.919,
    iou_mean: 0.9262,
    seed: 2,
    n_frames: 10,
    date: "2026-08-01T00:59:00+00:00",
  }),
];

describe("runHeadline", () => {
  it("falls through to the metrics an ordinary run actually writes", () => {
    // iou_holdout belongs to the retired single-frame holdout and val_iou_mean
    // only to joint runs; leading with those two alone showed no number at all.
    expect(runHeadline(run({ id: "a", iou_val: 0.879 }))).toEqual({
      value: "0.879",
      label: "val IoU",
    });
    expect(runHeadline(run({ id: "b", iou_mean: 0.9377 }))).toEqual({
      value: "0.938",
      label: "mean IoU",
    });
    expect(runHeadline(run({ id: "c" }))).toBeNull();
  });

  it("prefers a score on frames the model never saw", () => {
    const both = run({ id: "d", iou_val: 0.87, iou_mean: 0.94 });
    expect(runHeadline(both)?.label).toBe("val IoU");
  });

  it("leads a training run with its progress", () => {
    const training = run({
      id: "e",
      status: "running",
      steps_done: 400,
      steps_total: 2000,
    });
    expect(runHeadline(training)).toEqual({ value: "20%", label: "progress" });
  });
});

describe("runProvenance", () => {
  it("names only what distinguishes the run", () => {
    // "1 cond" on every row of a single-series project said nothing.
    expect(runProvenance(run({ id: "a" }))).toEqual([]);
    expect(
      runProvenance(
        run({ id: "b", datasets: ["a", "b"], heldout_datasets: ["b"] }),
      ),
    ).toEqual(["2 cond", "1 held out"]);
  });
});

describe("buildFamilies", () => {
  it("folds seed replicates into one experiment carrying their spread", () => {
    const families = buildFamilies(RUNS, { metric: "val" });
    const fvb = families.find((family) => family.name === "fvb-fix")!;

    expect(fvb.runs).toHaveLength(2);
    expect(fvb.score).toBeCloseTo(0.892, 3);
    expect(fvb.spread).toBeCloseTo(0.003, 3);
  });

  it("keeps runs of the same stem but different recipes apart", () => {
    const families = buildFamilies(
      [
        run({ id: "x-s0", iou_val: 0.9, recipe: ["sharp"] }),
        run({ id: "x-s1", iou_val: 0.8, recipe: ["sharp", "causal"] }),
      ],
      { metric: "val" },
    );
    // Same name, different physics: one experiment each, not one of two seeds.
    expect(families).toHaveLength(2);
  });

  it("ranks by the chosen metric, and the ranking changes with it", () => {
    // The fact the old rail could not show: this run is first on one metric
    // and near-last on the other.
    const byVal = buildFamilies(RUNS, { metric: "val" });
    const byMean = buildFamilies(RUNS, { metric: "mean" });

    expect(byMean[0].name).toBe("r4-causal");
    expect(byVal.find((f) => f.name === "r4-causal")!.rank).toBeGreaterThan(1);
  });

  it("never makes an incomparable run the datum the others are short of", () => {
    const families = buildFamilies(RUNS, { metric: "val" });
    const odd = families.find((family) => family.name === "r4-gate2")!;
    const leader = families.find(
      (family) => family.behind === null && !family.incomparable,
    )!;

    // It scores highest of all, but on 10 frames rather than 11.
    expect(odd.score).toBeGreaterThan(leader.score!);
    expect(odd.incomparable).toMatch(/10 frames/);
    // So the leader is the best COMPARABLE run, and nothing is reported as
    // short of a measurement it was never taken against.
    expect(leader.name).toBe("fvb-fix");
    expect(odd.behind).toBeNull();
  });

  it("filters on the recipe as well as the name", () => {
    expect(
      buildFamilies(RUNS, { metric: "val", query: "causal" }),
    ).toHaveLength(1);
    expect(buildFamilies(RUNS, { metric: "val", query: "gate" })).toHaveLength(
      1,
    );
    expect(buildFamilies(RUNS, { metric: "val", query: "nope" })).toHaveLength(
      0,
    );
  });
});

describe("RunRail", () => {
  const renderRail = (props: Partial<Parameters<typeof RunRail>[0]> = {}) => {
    const onOpen = vi.fn();
    const onCompare = vi.fn();
    render(
      <RunRail
        runs={RUNS}
        datasetLabels={new Map()}
        selectedId="fvb-fix-s0"
        onOpen={onOpen}
        onCompare={onCompare}
        {...props}
      />,
    );
    return { onOpen, onCompare };
  };

  it("leads each row with a number and names which metric it is", () => {
    renderRail();
    const row = screen.getByRole("option", { name: /fvb-fix/ });
    expect(row).toHaveTextContent("0.892");
    expect(row).toHaveTextContent(/best val IoU/);
    expect(row).toHaveTextContent("2 seeds");
  });

  it("shows the recipe, and says so when there is none to show", () => {
    renderRail({
      runs: [
        run({ id: "bench", iou_val: 0.9, recipe: null }),
        ...RUNS.slice(2),
      ],
    });
    expect(screen.getByRole("option", { name: /bench/ })).toHaveTextContent(
      "config not recorded",
    );
    expect(screen.getByRole("option", { name: /r4-causal/ })).toHaveTextContent(
      "causal",
    );
  });

  it("cycles the sort order, and always says which one it is in", () => {
    renderRail();
    // The row that holds rank 1, not merely the first row: the incomparable
    // run sorts high on val IoU but is deliberately left unnumbered.
    const ranked = () =>
      screen
        .getAllByRole("option")
        .find((row) => row.textContent?.startsWith("1"))!;
    expect(ranked()).toHaveTextContent("fvb-fix");

    // One control that cycles, naming the order it is in and the next one, so
    // it is never a guess what a click will do.
    const sort = screen.getByRole("button", { name: /Sort order: val IoU/ });
    fireEvent.click(sort);
    expect(ranked()).toHaveTextContent("r4-causal");
    expect(
      screen.getByRole("button", { name: /Sort order: mean IoU/ }),
    ).toBeInTheDocument();

    // Round the cycle: newest, then back to where it started.
    fireEvent.click(
      screen.getByRole("button", { name: /Sort order: mean IoU/ }),
    );
    fireEvent.click(screen.getByRole("button", { name: /Sort order: newest/ }));
    expect(
      screen.getByRole("button", { name: /Sort order: val IoU/ }),
    ).toBeInTheDocument();
  });

  it("gives an incomparable run no rank number at all", () => {
    renderRail();
    const odd = screen.getByRole("option", { name: /r4-gate2/ });
    expect(odd).toHaveTextContent("10 frames · not comparable");
    // It scores highest, and claims neither a rank nor a lead over anyone.
    expect(odd.textContent?.startsWith("1")).toBe(false);
    expect(odd).not.toHaveTextContent(/best val IoU/);
  });

  it("opens an experiment to its own runs, each with its recorded seed", () => {
    const { onOpen } = renderRail();
    fireEvent.click(screen.getByRole("option", { name: /fvb-fix/ }));

    // Opening the row also opens the run it leads with.
    expect(onOpen).toHaveBeenCalledWith("fvb-fix-s0");
    const kids = screen.getByRole("list");
    expect(within(kids).getByText("fvb-fix-s1")).toBeInTheDocument();
    // The seed is read from the run, not parsed off a name that may disagree.
    expect(within(kids).getAllByText(/seed \d/)).toHaveLength(2);
  });

  it("assembles a comparison without leaving the rail", () => {
    const { onCompare, onOpen } = renderRail();
    const compare = screen.getByRole("button", { name: "Compare" });
    expect(compare).toBeDisabled();

    fireEvent.click(
      screen.getByRole("checkbox", { name: /Compare r4-causal/ }),
    );
    // Ticking a run must not navigate away from the one being read.
    expect(onOpen).not.toHaveBeenCalled();
    expect(compare).toBeDisabled(); // one run is not a comparison

    fireEvent.click(screen.getByRole("checkbox", { name: /Compare r4-gate2/ }));
    fireEvent.click(compare);
    expect(onCompare).toHaveBeenCalledWith(["r4-causal-s2", "r4-gate2-s2"]);
  });

  it("filters, and says what to do when nothing matches", () => {
    renderRail();
    fireEvent.change(screen.getByLabelText("Filter runs"), {
      target: { value: "causal" },
    });
    expect(screen.getAllByRole("option")).toHaveLength(1);

    fireEvent.change(screen.getByLabelText("Filter runs"), {
      target: { value: "zzz" },
    });
    expect(screen.getByText(/No run matches/)).toBeInTheDocument();
  });
});

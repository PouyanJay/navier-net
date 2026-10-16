/** The acceptance audit on the Results page: verdicts against thresholds. */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { RunMetrics } from "../src/lib/api";
import {
  AcceptancePanel,
  auditVerdicts,
} from "../src/views/results/AcceptancePanel";

const AUDIT = {
  mirror: { worst: 0.031, max: 0.05, pass: true },
  residuals: { worst: 0.12, max: 0.05, pass: false },
  flux: { worst: 0.02, max: 0.05, pass: true },
  heat: { worst: 0.4, max: 0.3, pass: false },
};
const OUTLINE = {
  outline: { worst: 0.91, min: 0.85, pass: true },
  outline_drop: { worst: 0.08, max: 0.05, pass: false },
  nose: { worst: 0.04, max: 0.1, pass: true },
};

/** Metrics as a file on disk may hold them: any shape at all. */
const metricsOf = (body: unknown) => body as RunMetrics;

describe("auditVerdicts", () => {
  it("reads the verdicts the evaluate stage wrote into the metrics", () => {
    const verdicts = auditVerdicts({ iou_mean: 0.9, audit: AUDIT });
    expect(verdicts?.mirror).toEqual({ worst: 0.031, max: 0.05, pass: true });
    expect(verdicts?.residuals).toEqual(
      expect.objectContaining({ pass: false }),
    );
  });

  it("is null for a run that wrote no audit", () => {
    expect(auditVerdicts({ iou_mean: 0.9 })).toBeNull();
    expect(auditVerdicts(null)).toBeNull();
    expect(auditVerdicts(metricsOf({ audit: "yes" }))).toBeNull();
  });

  it("reads a verdict held above a floor as well as one held below a cap", () => {
    const verdicts = auditVerdicts({ audit: { ...AUDIT, ...OUTLINE } });
    expect(verdicts?.outline).toEqual({ worst: 0.91, min: 0.85, pass: true });
    expect(verdicts?.nose).toEqual({ worst: 0.04, max: 0.1, pass: true });
    // A verdict with neither threshold is not a verdict.
    expect(
      auditVerdicts(
        metricsOf({ audit: { outline: { worst: 0.9, pass: true } } }),
      )?.outline,
    ).toBe("unreadable");
  });

  it("reads a partial or malformed audit as missing tests, never as passes", () => {
    const verdicts = auditVerdicts(
      metricsOf({ audit: { mirror: { worst: "a" }, flux: AUDIT.flux } }),
    );
    expect(verdicts?.mirror).toBe("unreadable");
    expect(verdicts?.residuals).toBe("absent");
    expect(verdicts?.flux).toEqual(AUDIT.flux);
  });
});

describe("AcceptancePanel", () => {
  it("states each verdict against its threshold and counts the passes", () => {
    render(<AcceptancePanel audit={auditVerdicts({ audit: AUDIT })} />);

    expect(screen.getByText("2 of 4 pass")).toBeInTheDocument();
    const mirror = screen.getByText("Mirror symmetry").closest(".check-row")!;
    expect(mirror).toHaveAttribute("data-ok", "true");
    expect(mirror).toHaveTextContent("0.0310");
    expect(mirror).toHaveTextContent("≤ 0.0500");
    expect(mirror).toHaveTextContent("within tolerance");
    const heat = screen.getByText("Heat budget").closest(".check-row")!;
    expect(heat).toHaveAttribute("data-ok", "false");
    expect(heat).toHaveTextContent("flagged");
    // The one test a single run cannot make is stated as not run, not failed.
    expect(
      screen.getByText("Seed agreement").closest(".check-row"),
    ).toHaveClass("unrun");
  });

  it("puts the outline first and states a floor as a floor", () => {
    render(
      <AcceptancePanel
        audit={auditVerdicts({ audit: { ...AUDIT, ...OUTLINE } })}
      />,
    );

    expect(screen.getByText("4 of 7 pass")).toBeInTheDocument();
    const rows = Array.from(document.querySelectorAll(".check-row b")).map(
      (b) => b.textContent,
    );
    expect(rows.slice(0, 3)).toEqual([
      "Outline against the masks",
      "Outline held from the warm-up",
      "Nose position",
    ]);
    const outline = screen
      .getByText("Outline against the masks")
      .closest(".check-row")!;
    expect(outline).toHaveAttribute("data-ok", "true");
    expect(outline).toHaveTextContent("≥ 0.850");
    const drop = screen
      .getByText("Outline held from the warm-up")
      .closest(".check-row")!;
    expect(drop).toHaveAttribute("data-ok", "false");
    expect(drop).toHaveTextContent("≤ 0.0500");
  });

  it("says when a run recorded no audit instead of showing four failures", () => {
    render(<AcceptancePanel audit={null} />);

    expect(screen.getByText("not run")).toBeInTheDocument();
    expect(
      screen.getByText(/recorded no acceptance audit/),
    ).toBeInTheDocument();
    expect(screen.queryByText("Mirror symmetry")).toBeNull();
  });

  it("shows a missing or unreadable verdict as not run, and counts only the measured", () => {
    render(
      <AcceptancePanel
        audit={auditVerdicts(
          metricsOf({ audit: { mirror: AUDIT.mirror, heat: { worst: "x" } } }),
        )}
      />,
    );

    expect(screen.getByText("1 of 1 pass")).toBeInTheDocument();
    expect(screen.getByText("Heat budget").closest(".check-row")).toHaveClass(
      "unrun",
    );
    expect(screen.getByText(/cannot read/)).toBeInTheDocument();
    expect(
      screen.getByText("Cross-section flux balance").closest(".check-row"),
    ).toHaveClass("unrun");
  });
});

/** The acceptance audit on the Results page: verdicts against thresholds. */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

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

describe("auditVerdicts", () => {
  it("reads the verdicts the evaluate stage wrote into the metrics", () => {
    const verdicts = auditVerdicts({ iou_mean: 0.9, audit: AUDIT });
    expect(verdicts?.mirror).toEqual({ worst: 0.031, max: 0.05, pass: true });
    expect(verdicts?.residuals?.pass).toBe(false);
  });

  it("is null for a run that wrote no audit, or a malformed one", () => {
    expect(auditVerdicts({ iou_mean: 0.9 })).toBeNull();
    expect(auditVerdicts(null)).toBeNull();
    expect(auditVerdicts({ audit: "yes" })).toBeNull();
    expect(auditVerdicts({ audit: { mirror: { worst: "a" } } })).toBeNull();
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
    const heat = screen.getByText("Heat budget").closest(".check-row")!;
    expect(heat).toHaveAttribute("data-ok", "false");
    // The one test a single run cannot make is stated as not run, not failed.
    expect(
      screen.getByText("Seed agreement").closest(".check-row"),
    ).toHaveClass("unrun");
  });

  it("says when a run recorded no audit instead of showing four failures", () => {
    render(<AcceptancePanel audit={null} />);

    expect(screen.getByText("not run")).toBeInTheDocument();
    expect(
      screen.getByText(/recorded no acceptance audit/),
    ).toBeInTheDocument();
    expect(screen.queryByText("Mirror symmetry")).toBeNull();
  });
});

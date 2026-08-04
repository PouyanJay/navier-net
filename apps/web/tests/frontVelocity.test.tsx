import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, api, type FrontVelocityReport } from "../src/lib/api";
import {
  FrontVelocityTab,
  speedReadout,
} from "../src/views/results/FrontVelocityTab";

const REPORT: FrontVelocityReport = {
  front_geometry: true,
  nose_speed: {
    t_ms: [0, 0.1, 0.2, 0.3],
    v_um_per_ms: [120.5, 138.2, 141.9, 96.4],
    measured: {
      t_ms: [0.05, 0.15, 0.25],
      v_um_per_ms: [118.0, 140.0, 99.0],
      heldout: [false, true, false],
    },
  },
  apex: {
    t_ms: [0, 0.1, 0.2, 0.3],
    x_um: [180, 192, 206, 216],
    y_um: [64, 64.1, 63.9, 64],
    vx_um_per_ms: [118.0, 137.0, 140.0, 95.0],
    vy_um_per_ms: [0.4, -0.2, 0.1, 0.0],
    measured: {
      t_ms: [0.05, 0.15, 0.25],
      vx_um_per_ms: [120.0, 139.0, 98.0],
      vy_um_per_ms: [0.3, -0.1, 0.05],
      heldout: [false, true, false],
    },
  },
};

function serve(report: FrontVelocityReport | ApiError) {
  return vi
    .spyOn(api, "getFrontVelocity")
    .mockImplementation(() =>
      report instanceof ApiError
        ? Promise.reject(report)
        : Promise.resolve(report),
    );
}

afterEach(() => vi.restoreAllMocks());

describe("FrontVelocityTab", () => {
  it("charts the nose speed with its unit", async () => {
    serve(REPORT);
    render(<FrontVelocityTab runId="run-a" />);

    expect(await screen.findByText("Nose speed")).toBeInTheDocument();
    // Every speed chart carries the same axis unit beside its heading.
    expect(screen.getAllByText("µm/ms")).toHaveLength(3);
  });

  it("says the report has not been written rather than drawing an empty axis", async () => {
    serve(new ApiError("no front-velocity report", 404));
    render(<FrontVelocityTab runId="run-a" />);

    expect(
      await screen.findByText(/re-run the evaluate stage/),
    ).toBeInTheDocument();
  });

  it("surfaces a real failure as an error, not as missing data", async () => {
    serve(new ApiError("checkpoint unreadable", 500));
    render(<FrontVelocityTab runId="run-a" />);

    expect(
      await screen.findByText("Could not load the front velocity"),
    ).toBeInTheDocument();
  });

  it("scopes the request to the viewed condition of a joint run", async () => {
    const spy = serve(REPORT);
    render(<FrontVelocityTab runId="run-a" dataset="series_2" />);

    await waitFor(() => expect(spy).toHaveBeenCalledWith("run-a", "series_2"));
  });

  it("draws a held-out interval apart from the trained ones", async () => {
    serve(REPORT);
    render(<FrontVelocityTab runId="run-a" />);
    const card = (await screen.findByText("Nose speed")).closest(".kin-chart")!;

    // Amber is the holdout tone; one of this chart's three measured pairs spans
    // the held-out frame, so exactly one marker carries it and two do not.
    await waitFor(() =>
      expect(card.querySelectorAll(".chart-sample.series-3")).toHaveLength(1),
    );
    expect(card.querySelectorAll(".chart-sample.series-1")).toHaveLength(2);
  });

  it("names the held-out interval, so the distinction is not colour alone", async () => {
    serve(REPORT);
    render(<FrontVelocityTab runId="run-a" />);

    expect(await screen.findByText(/held-out frame/)).toBeInTheDocument();
  });

  it("charts both apex components, one axis each", async () => {
    serve(REPORT);
    render(<FrontVelocityTab runId="run-a" />);

    expect(
      await screen.findByText("Apex velocity · along x"),
    ).toBeInTheDocument();
    expect(screen.getByText("Apex velocity · across y")).toBeInTheDocument();
  });

  it("explains a run with no explicit front instead of drawing an empty apex axis", async () => {
    serve({ ...REPORT, front_geometry: false, apex: null });
    render(<FrontVelocityTab runId="run-a" />);

    expect(await screen.findByText(/model.front_geometry/)).toBeInTheDocument();
    expect(screen.getByText(/Enable Front geometry/)).toBeInTheDocument();
    // The nose speed does not need a front, so it is still charted.
    expect(screen.getByText("Nose speed")).toBeInTheDocument();
    expect(
      screen.queryByText("Apex velocity · along x"),
    ).not.toBeInTheDocument();
  });

  it("gives both units, the axis' and SI", () => {
    expect(speedReadout(120.5)).toBe("121 µm/ms (0.120 m/s)");
    // A near-stationary flank must not read as a flat zero -- that contrast
    // against the nose is what the profile chart exists to show.
    expect(speedReadout(0.0031)).toBe("0.00310 µm/ms (0.00000310 m/s)");
    expect(speedReadout(0)).toBe("0 µm/ms (0 m/s)");
  });
});

import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  ApiError,
  api,
  type FrontProfile,
  type FrontVelocityReport,
} from "../src/lib/api";
import { kymographImage } from "../src/views/results/FrontProfilePanel";
import {
  FrontVelocityTab,
  speedReadout,
} from "../src/views/results/FrontVelocityTab";

/** Six bins: one per cap, two per body — the traversal in miniature. The nose
 * cap's measured values are null, as the suppression rule requires. */
const PROFILE: FrontProfile = {
  s: [1 / 12, 3 / 12, 5 / 12, 7 / 12, 9 / 12, 11 / 12],
  segments: [
    {
      name: "root_cap",
      bin_start: 0,
      bin_end: 1,
      s_start: 0,
      s_end: 1 / 6,
      measured: true,
    },
    {
      name: "upper_body",
      bin_start: 1,
      bin_end: 3,
      s_start: 1 / 6,
      s_end: 1 / 2,
      measured: true,
    },
    {
      name: "nose_cap",
      bin_start: 3,
      bin_end: 4,
      s_start: 1 / 2,
      s_end: 2 / 3,
      measured: false,
    },
    {
      name: "lower_body",
      bin_start: 4,
      bin_end: 6,
      s_start: 2 / 3,
      s_end: 1,
      measured: true,
    },
  ],
  times: [
    {
      t_ms: 0.0,
      frames: [1, 2],
      heldout: false,
      model: [0.1, 0.4, 3.2, 120.0, 3.0, 0.3],
      measured: [0.1, 0.5, 3.0, null, 2.8, 0.2],
    },
    {
      t_ms: 0.1,
      frames: [2, 3],
      heldout: true,
      model: [0.1, 0.5, 3.6, 131.0, 3.4, 0.4],
      measured: [0.2, 0.4, 3.3, null, 3.1, 0.3],
    },
  ],
  kymograph: {
    t_ms: [0, 0.1, 0.2, 0.3],
    v_um_per_ms: [
      [0.1, 0.4, 3.2, 120.0, 3.0, 0.3],
      [0.1, 0.5, 3.6, 131.0, 3.4, 0.4],
      [0.1, 0.5, 3.7, 133.0, 3.5, 0.4],
      [0.0, 0.3, 2.4, 90.0, 2.2, 0.2],
    ],
  },
};

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
  profile: PROFILE,
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
    // Every speed chart carries the same axis unit beside its heading: the
    // nose, both apex components, and the profile.
    expect(screen.getAllByText("µm/ms")).toHaveLength(4);
  });

  it("says the report has not been written rather than drawing an empty axis", async () => {
    serve(new ApiError("no front-velocity report", 404));
    render(<FrontVelocityTab runId="run-a" />);

    expect(
      await screen.findByText(/re-run the evaluate stage/i),
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

  it("walks the front's segments and names them on the axis", async () => {
    serve(REPORT);
    const { container } = render(<FrontVelocityTab runId="run-a" />);

    expect(
      await screen.findByText("Normal speed along the front"),
    ).toBeInTheDocument();
    const labels = [...container.querySelectorAll(".chart-band-label")].map(
      (node) => node.textContent,
    );
    expect(labels).toEqual([
      "root cap",
      "upper body",
      "nose cap",
      "lower body",
    ]);
  });

  it("marks the nose cap as a span the measurement deliberately omits", async () => {
    serve(REPORT);
    const { container } = render(<FrontVelocityTab runId="run-a" />);
    await screen.findByText("Normal speed along the front");

    // Exactly one band is muted, and it is the one the report flagged.
    expect(container.querySelectorAll(".chart-band.muted")).toHaveLength(1);
    expect(
      screen.getByText(/level-set estimate is first-order/),
    ).toBeInTheDocument();
  });

  it("breaks the measured line across the gap instead of drawing through it", async () => {
    serve(REPORT);
    const { container } = render(<FrontVelocityTab runId="run-a" />);
    await screen.findByText("Normal speed along the front");

    // Two sub-paths (an "M" each) for the measured series: one either side of
    // the suppressed nose cap. A line drawn straight across the gap would be
    // one, and would assert a value the data never claimed.
    const paths = [...container.querySelectorAll("path.chart-line")];
    const measured = paths.find((node) => node.classList.contains("series-1"))!;
    expect((measured.getAttribute("d")!.match(/M/g) ?? []).length).toBe(2);
  });

  it("scrubs between frame pairs", async () => {
    serve(REPORT);
    render(<FrontVelocityTab runId="run-a" />);

    expect(await screen.findByText("Frames 1\u20132")).toBeInTheDocument();
    const scrubber = screen.getByLabelText("Frame pair");
    fireEvent.change(scrubber, { target: { value: "1" } });

    expect(await screen.findByText("Frames 2\u20133")).toBeInTheDocument();
    expect(screen.getByText(/held out/)).toBeInTheDocument();
  });

  it("draws the kymograph as an image with a signed scale", async () => {
    serve(REPORT);
    render(<FrontVelocityTab runId="run-a" />);

    expect(await screen.findByText("Kymograph")).toBeInTheDocument();
    expect(
      screen.getByRole("img", { name: /^Kymograph:/ }),
    ).toBeInTheDocument();
    // Signed, so the scale runs either side of zero.
    expect(screen.getByText(/^−/)).toBeInTheDocument();
  });

  it("gives both units, the axis' and SI", () => {
    expect(speedReadout(120.5)).toBe("121 µm/ms (0.120 m/s)");
    // A near-stationary flank must not read as a flat zero -- that contrast
    // against the nose is what the profile chart exists to show.
    expect(speedReadout(0.0031)).toBe("0.00310 µm/ms (0.00000310 m/s)");
    expect(speedReadout(0)).toBe("0 µm/ms (0 m/s)");
  });
});

describe("kymographImage", () => {
  it("transposes the report's [time][position] into an image of time by position", () => {
    // Two instants, three positions along the front.
    const rows = [
      [1, 0, -1],
      [1, 1, 1],
    ];
    const image = kymographImage(rows, 1);

    expect(image.width).toBe(2); // time runs along x
    expect(image.height).toBe(3); // position runs down y
    expect(image.data).toHaveLength(2 * 3 * 4);

    // The +1 and -1 of the first instant are the extremes of the diverging
    // ramp, so they must differ; a transposition error would put the second
    // instant's uniform row where this contrast belongs.
    const pixel = (t: number, s: number) => {
      const at = (s * image.width + t) * 4;
      return [...image.data.slice(at, at + 3)].join(",");
    };
    expect(pixel(0, 0)).not.toBe(pixel(0, 2));
    expect(pixel(1, 0)).toBe(pixel(1, 2));
  });

  it("leaves an uncomputed cell transparent rather than painting it as zero", () => {
    // Zero is a perfectly good speed — "the front stood still" — so a missing
    // value must not borrow its colour.
    const image = kymographImage([[null, 0]], 1);

    expect(image.data[3]).toBe(0); // alpha of the null cell
    expect(image.data[7]).toBe(255); // alpha of the genuine zero
  });
});

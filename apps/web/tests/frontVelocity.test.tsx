import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, api, type FrontVelocityReport } from "../src/lib/api";
import { FrontVelocityTab } from "../src/views/results/FrontVelocityTab";

const REPORT: FrontVelocityReport = {
  front_geometry: true,
  nose_speed: {
    t_ms: [0, 0.1, 0.2, 0.3],
    v_um_per_ms: [120.5, 138.2, 141.9, 96.4],
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
    expect(screen.getByText("µm/ms")).toBeInTheDocument();
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
});

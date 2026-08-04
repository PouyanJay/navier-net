import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { PhysicsDiagnostics } from "../src/lib/api";
import { InterfacePhysicsPanel } from "../src/views/results/InterfacePhysicsPanel";

const DIAGNOSTICS: PhysicsDiagnostics = {
  laplace_error_nose: 0.06,
  laplace_error_front: 0.18,
  axial_capillary_gradient: 0.41,
  neck_depth_model: 0.44,
  neck_depth_measured: 0.47,
  neck_location_model: 0.5,
  neck_location_measured: 0.5,
  profile_stations: [0.1, 0.5, 0.9],
  per_frame: [
    {
      frame: 10,
      neck_depth_model: 0.3,
      neck_depth_measured: 0.32,
      neck_location_model: 0.5,
      neck_location_measured: 0.5,
      half_width_model: [0.2, 0.14, 0.38],
      half_width_measured: [0.21, 0.13, 0.39],
    },
    {
      frame: 11,
      neck_depth_model: 0.44,
      neck_depth_measured: 0.47,
      neck_location_model: 0.5,
      neck_location_measured: 0.5,
      half_width_model: [0.2, 0.12, 0.39],
      half_width_measured: [0.21, 0.11, 0.39],
    },
  ],
  residual_convergence: {
    darcy: { first: 6.8, last: 0.31, ratio: 0.046 },
    mom: { first: 6.78, last: 4.68, ratio: 0.69 },
  },
};

describe("InterfacePhysicsPanel", () => {
  it("shows the interface conditions and the neck against the measured masks", () => {
    render(<InterfacePhysicsPanel physics={DIAGNOSTICS} frontGeometry />);

    expect(screen.getByText("Young–Laplace · nose")).toBeInTheDocument();
    expect(screen.getByText("6.0")).toBeInTheDocument();
    expect(screen.getByText("Axial capillary gradient")).toBeInTheDocument();
    // Both necks are shown, model beside measured -- in the headline stat and
    // again per frame, so the same numbers legitimately appear twice.
    expect(screen.getAllByText("0.440").length).toBeGreaterThan(0);
    expect(screen.getAllByText("0.470").length).toBeGreaterThan(0);
    expect(screen.getByText(/measured 0.470 at u = 0.50/)).toBeInTheDocument();
  });

  it("separates a residual that converged from one that never moved", () => {
    render(<InterfacePhysicsPanel physics={DIAGNOSTICS} frontGeometry />);

    // The value of a residual says little; whether it FELL is the signal.
    expect(screen.getByText("converging")).toBeInTheDocument();
    expect(screen.getByText("not solved")).toBeInTheDocument();
  });

  it("explains why there is nothing to measure without an explicit front", () => {
    render(<InterfacePhysicsPanel physics={null} frontGeometry={false} />);
    expect(screen.getByText(/model.front_geometry/)).toBeInTheDocument();
    expect(screen.getByText(/Enable Front geometry/)).toBeInTheDocument();
  });

  it("distinguishes a front-geometry run whose diagnostics were never written", () => {
    render(<InterfacePhysicsPanel physics={null} frontGeometry />);
    expect(screen.getByText(/re-run the evaluate stage/)).toBeInTheDocument();
  });

  it("says so when no physics residual was active at all", () => {
    render(
      <InterfacePhysicsPanel
        physics={{ ...DIAGNOSTICS, residual_convergence: {} }}
        frontGeometry
      />,
    );
    expect(screen.getByText(/nothing to converge/)).toBeInTheDocument();
  });
});

/** Reconstruction viewport: real contour rendering, layers, scrub. */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ReconstructionViewport } from "../src/components/ReconstructionViewport";
import type { InterfaceData } from "../src/lib/api";

const DATA: InterfaceData = {
  run_id: "run-test",
  domain: { x_um: [0, 1700], y_um: [0, 360], x_pin_um: 120 },
  frames: [
    {
      t_ms: 0,
      contours: [
        [
          [100, 100],
          [300, 100],
          [300, 250],
          [100, 250],
        ],
      ],
    },
    {
      t_ms: 2.5,
      contours: [
        [
          [100, 100],
          [700, 100],
          [700, 250],
          [100, 250],
        ],
      ],
    },
  ],
  measured: [
    {
      t_ms: 0,
      contours: [
        [
          [110, 110],
          [290, 110],
          [290, 240],
        ],
      ],
    },
  ],
};

describe("ReconstructionViewport", () => {
  it("renders interface and measured contours with the HUD readout", () => {
    const { container } = render(<ReconstructionViewport data={DATA} />);
    expect(container.querySelectorAll("path.vp-interface")).toHaveLength(1);
    expect(container.querySelectorAll("path.vp-measured")).toHaveLength(1);
    // L = max contour x − x_pin = 300 − 120.
    expect(screen.getByText(/t = 0\.00 ms · L = 180 µm/)).toBeInTheDocument();
    expect(screen.getByText(/nucleation cavity · pinned/)).toBeInTheDocument();
  });

  it("toggles layers off and scrubs to a later instant", () => {
    const { container } = render(<ReconstructionViewport data={DATA} />);
    fireEvent.click(screen.getByRole("button", { name: "measured" }));
    expect(container.querySelectorAll("path.vp-measured")).toHaveLength(0);
    fireEvent.click(screen.getByRole("button", { name: "interface" }));
    expect(container.querySelectorAll("path.vp-interface")).toHaveLength(0);

    fireEvent.change(
      screen.getByRole("slider", { name: "Scrub reconstruction time" }),
      {
        target: { value: "1" },
      },
    );
    expect(screen.getByText(/t = 2\.50 ms · L = 580 µm/)).toBeInTheDocument();
    expect(screen.getByText("t 2.50 / 2.50 ms")).toBeInTheDocument();
  });

  it("play toggles to pause with an accessible name", () => {
    render(<ReconstructionViewport data={DATA} />);
    const play = screen.getByRole("button", { name: "Play reconstruction" });
    fireEvent.click(play);
    expect(
      screen.getByRole("button", { name: "Pause playback" }),
    ).toBeInTheDocument();
  });
});

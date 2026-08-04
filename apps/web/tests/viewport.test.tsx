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
      // The nose racing ahead, a flank barely moving, and a root that is
      // stationary enough to fall below the drawing floor.
      front: [
        [300, 175, 1, 0, 120],
        [200, 250, 0, 1, 12],
        [100, 175, -1, 0, 0.2],
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
      front: null,
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
      front: null,
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

  it("draws front-velocity arrows only when asked, and drops the specks", () => {
    const { container } = render(<ReconstructionViewport data={DATA} />);
    // Off by default: the arrows annotate the contour, they are not the subject.
    expect(container.querySelectorAll(".vp-velocity line")).toHaveLength(0);

    fireEvent.click(screen.getByRole("button", { name: /velocity/ }));

    // Two of the three: the near-stationary root is below the drawing floor,
    // where an arrow would read as noise on the contour rather than as data.
    expect(container.querySelectorAll(".vp-velocity line")).toHaveLength(2);
    // And the overlay states what the arrows are, visibly — an arrow that looks
    // like a velocity must say which velocity it is.
    expect(screen.getByText(/normal component/)).toBeInTheDocument();
    expect(screen.getByText(/longest 120 µm\/ms/)).toBeInTheDocument();
  });

  it("points an arrow the right way on the inverted y axis", () => {
    const { container } = render(<ReconstructionViewport data={DATA} />);
    fireEvent.click(screen.getByRole("button", { name: /velocity/ }));
    const lines = [...container.querySelectorAll(".vp-velocity line")];

    // The nose arrow points along +x, which the flip does not touch.
    const nose = lines[0];
    expect(Number(nose.getAttribute("x2"))).toBeGreaterThan(
      Number(nose.getAttribute("x1")),
    );
    // The flank's outward normal is +y in data space; on screen y is inverted,
    // so it must be drawn UPWARD — a smaller y2 than y1. Adding the raw
    // component instead would send it into the bubble.
    const flank = lines[1];
    expect(Number(flank.getAttribute("y2"))).toBeLessThan(
      Number(flank.getAttribute("y1")),
    );
  });

  it("says why a run without an explicit front offers no arrows", () => {
    const { container } = render(<ReconstructionViewport data={DATA} />);
    fireEvent.change(
      screen.getByRole("slider", { name: "Scrub reconstruction time" }),
      { target: { value: "1" } },
    );

    // aria-disabled rather than disabled: the button keeps its place in the tab
    // order, so a keyboard user can still reach the reason it is unavailable —
    // and that reason is the accessible name, not a title only a pointer finds.
    const toggle = screen.getByRole("button", { name: /^velocity/ });
    expect(toggle).toHaveAttribute("aria-disabled", "true");
    expect(toggle).toHaveAccessibleName(
      expect.stringContaining("front_geometry"),
    );
    // And it does nothing when pressed, rather than toggling on an empty layer.
    fireEvent.click(toggle);
    expect(container.querySelectorAll(".vp-velocity line")).toHaveLength(0);
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

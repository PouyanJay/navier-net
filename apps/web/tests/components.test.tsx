import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { Chip, Stat, StatusDot } from "../src/components";

describe("Stat", () => {
  it("renders label, value and unit", () => {
    render(<Stat label="Holdout IoU" value="0.968" unit="σ" />);
    expect(screen.getByText("Holdout IoU")).toBeInTheDocument();
    expect(screen.getByText("0.968")).toBeInTheDocument();
  });

  it("carries the tone as a data attribute for token-driven color", () => {
    const { container } = render(<Stat label="x" value={1} tone="amber" />);
    expect(container.querySelector(".stat")).toHaveAttribute("data-tone", "amber");
  });
});

describe("StatusDot", () => {
  it("conveys state via a text label, not color alone", () => {
    render(<StatusDot tone="green" label="trained" />);
    expect(screen.getByText("trained")).toBeInTheDocument();
  });
});

describe("Chip", () => {
  it("renders its content with a tone", () => {
    const { container } = render(<Chip tone="accent">highest_t</Chip>);
    expect(screen.getByText("highest_t")).toBeInTheDocument();
    expect(container.querySelector(".chip")).toHaveAttribute("data-tone", "accent");
  });
});

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import {
  Chip,
  type Column,
  DL,
  IouChart,
  Stat,
  StatusDot,
  Table,
} from "../src/components";

describe("Stat", () => {
  it("renders label, value and unit", () => {
    render(<Stat label="Holdout IoU" value="0.968" unit="σ" />);
    expect(screen.getByText("Holdout IoU")).toBeInTheDocument();
    expect(screen.getByText("0.968")).toBeInTheDocument();
  });

  it("carries the tone as a data attribute for token-driven color", () => {
    const { container } = render(<Stat label="x" value={1} tone="amber" />);
    expect(container.querySelector(".stat")).toHaveAttribute(
      "data-tone",
      "amber",
    );
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
    expect(container.querySelector(".chip")).toHaveAttribute(
      "data-tone",
      "accent",
    );
  });
});

describe("DL", () => {
  it("renders label/value pairs", () => {
    render(<DL items={[{ label: "Reynolds", value: "215.5" }]} />);
    expect(screen.getByText("Reynolds")).toBeInTheDocument();
    expect(screen.getByText("215.5")).toBeInTheDocument();
  });
});

interface Row {
  frame: number;
  iou: number;
}

describe("Table", () => {
  it("renders headers, rows, and applies a row tone", () => {
    const columns: Column<Row>[] = [
      { header: "Frame", cell: (r) => r.frame, num: true },
      { header: "IoU", cell: (r) => r.iou.toFixed(3), num: true },
    ];
    const rows: Row[] = [
      { frame: 1, iou: 0.973 },
      { frame: 6, iou: 0.968 },
    ];
    const { container } = render(
      <Table
        columns={columns}
        rows={rows}
        rowKey={(r) => String(r.frame)}
        rowTone={(r) => (r.frame === 6 ? "amber" : undefined)}
      />,
    );
    expect(screen.getByText("Frame")).toBeInTheDocument();
    expect(screen.getByText("0.973")).toBeInTheDocument();
    expect(
      container.querySelector('tr[data-tone="amber"]'),
    ).toBeInTheDocument();
  });
});

describe("IouChart", () => {
  it("draws one bar per frame and marks the holdout", () => {
    const { container } = render(
      <IouChart
        data={[
          { frame: 1, iou: 0.97, holdout: false },
          { frame: 6, iou: 0.96, holdout: true },
        ]}
      />,
    );
    expect(container.querySelectorAll(".chart-bar")).toHaveLength(2);
    expect(container.querySelectorAll(".chart-bar.holdout")).toHaveLength(1);
  });
});

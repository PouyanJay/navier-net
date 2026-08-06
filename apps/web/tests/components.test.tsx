import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import {
  Chip,
  type Column,
  ConfirmDeleteDialog,
  DL,
  Select,
  Stat,
  StatusDot,
  Table,
} from "../src/components";
import { IouDotChart } from "../src/components/charts/IouDotChart";
import { RunHeader } from "../src/views/results/RunHeader";
import type { RunSummary } from "../src/lib/api";

describe("ConfirmDeleteDialog", () => {
  const setup = (onConfirm = vi.fn().mockResolvedValue(undefined)) => {
    const onClose = vi.fn();
    render(
      <ConfirmDeleteDialog
        title="Delete run"
        confirmLabel="Delete run"
        onConfirm={onConfirm}
        onClose={onClose}
      >
        Delete <b>demo_run</b> and all its outputs?
      </ConfirmDeleteDialog>,
    );
    return { onConfirm, onClose };
  };

  it("gates the destructive action behind an explicit, irreversible confirmation", () => {
    setup();
    // A real alertdialog with the consequence and the irreversibility spelled out.
    expect(screen.getByRole("alertdialog")).toBeInTheDocument();
    expect(screen.getByText("demo_run")).toBeInTheDocument();
    expect(screen.getByText("This cannot be undone.")).toBeInTheDocument();
  });

  it("runs onConfirm only when the danger button is pressed", () => {
    const { onConfirm } = setup();
    fireEvent.click(screen.getByRole("button", { name: "Delete run" }));
    expect(onConfirm).toHaveBeenCalledTimes(1);
  });

  it("Cancel closes without deleting", () => {
    const { onConfirm, onClose } = setup();
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(onConfirm).not.toHaveBeenCalled();
  });

  it("Escape closes without deleting", () => {
    const { onConfirm, onClose } = setup();
    fireEvent.keyDown(window, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(onConfirm).not.toHaveBeenCalled();
  });

  it("keeps itself open and surfaces the reason when the delete fails", async () => {
    const { onClose } = setup(
      vi.fn().mockRejectedValue(new Error("run is training")),
    );
    fireEvent.click(screen.getByRole("button", { name: "Delete run" }));
    expect(await screen.findByText("run is training")).toBeInTheDocument();
    expect(onClose).not.toHaveBeenCalled(); // still open, so the user can retry/cancel
  });
});

describe("RunHeader delete action", () => {
  const makeRun = (status: RunSummary["status"]): RunSummary => ({
    id: "demo_run",
    dataset: "highest_t",
    datasets: ["highest_t"],
    heldout_datasets: [],
    status,
    date: null,
    steps: 1500,
    iou_holdout: null,
    val_iou_mean: null,
  });

  const header = (status: RunSummary["status"], onDelete: () => void) => (
    <RunHeader
      run={makeRun(status)}
      detail={null}
      datasetLabels={new Map()}
      validationFrames={[10, 11]}
      standing="rank 1 of 3 · best val IoU"
      viewDataset={null}
      onViewDataset={() => {}}
      onResume={() => {}}
      resuming={false}
      onDelete={onDelete}
    />
  );

  it("offers a Delete button that opens the confirm", () => {
    const onDelete = vi.fn();
    render(header("trained", onDelete));
    fireEvent.click(screen.getByRole("button", { name: "Delete run" }));
    expect(onDelete).toHaveBeenCalledTimes(1);
  });

  it("disables Delete while the run is training (can't delete it out from under it)", () => {
    const onDelete = vi.fn();
    render(header("running", onDelete));
    const button = screen.getByRole("button", { name: "Delete run" });
    expect(button).toBeDisabled();
    fireEvent.click(button);
    expect(onDelete).not.toHaveBeenCalled();
  });
});

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

describe("IouDotChart", () => {
  it("positions one dot per frame and labels the holdout in text", () => {
    const { container } = render(
      <IouDotChart
        frames={[
          { frame: 1, iou: 0.97, role: "supervised" },
          { frame: 6, iou: 0.96, role: "holdout" },
        ]}
        mean={0.965}
        ariaLabel="Per-frame IoU"
      />,
    );
    expect(container.querySelectorAll(".iou-dot")).toHaveLength(2);
    expect(container.querySelectorAll(".iou-dot.hold")).toHaveLength(1);
    expect(container.textContent).toContain("HOLDOUT");
    expect(container.textContent).toContain("mean 0.965");
  });
});

describe("Select", () => {
  const OPTIONS = [
    { value: "a", label: "Growth kinematics", hint: "L(t) and its fit" },
    { value: "b", label: "Interface evolution", hint: "silhouettes" },
  ];

  function open(onChange = vi.fn()) {
    render(
      <Select
        label="Preprocessing check"
        value="a"
        options={OPTIONS}
        onChange={onChange}
      />,
    );
    const trigger = screen.getByLabelText("Preprocessing check");
    fireEvent.click(trigger);
    return { trigger, onChange };
  }

  it("is the app's own control, not the platform's", () => {
    // A native <select> renders the operating system's menu, which arrives in
    // neither our font, our surfaces, nor our themes.
    const { trigger } = open();
    expect(trigger.tagName).toBe("BUTTON");
    expect(document.querySelector("select")).toBeNull();
    expect(trigger).toHaveAttribute("aria-haspopup", "listbox");
    expect(trigger).toHaveAttribute("aria-expanded", "true");
  });

  it("states the current choice and what each option draws", () => {
    const { trigger } = open();
    expect(trigger).toHaveTextContent("Growth kinematics");
    expect(
      screen.getByRole("option", { name: /Growth kinematics/ }),
    ).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText("silhouettes")).toBeInTheDocument();
  });

  it("reserves the check column so labels do not shift with the choice", () => {
    // visibility, not display: hiding the glyph outright would slide every
    // label sideways as the selection moves down the list.
    open();
    const ticks = document.querySelectorAll(".pick-tick");
    expect(ticks).toHaveLength(2);
    expect((ticks[0] as HTMLElement).style.visibility).toBe("visible");
    expect((ticks[1] as HTMLElement).style.visibility).toBe("hidden");
  });

  it("opens, moves and commits from the keyboard", () => {
    const onChange = vi.fn();
    render(
      <Select
        label="Preprocessing check"
        value="a"
        options={OPTIONS}
        onChange={onChange}
      />,
    );
    const trigger = screen.getByLabelText("Preprocessing check");

    fireEvent.keyDown(trigger, { key: "ArrowDown" });
    expect(trigger).toHaveAttribute("aria-expanded", "true");
    fireEvent.keyDown(trigger, { key: "ArrowDown" });
    // The active option is named on the trigger, since focus never leaves it.
    expect(trigger.getAttribute("aria-activedescendant")).toMatch(/-b$/);

    fireEvent.keyDown(trigger, { key: "Enter" });
    expect(onChange).toHaveBeenCalledWith("b");
    expect(trigger).toHaveAttribute("aria-expanded", "false");
  });

  it("closes on Escape without choosing, and returns focus", () => {
    const { trigger, onChange } = open();
    fireEvent.keyDown(trigger, { key: "Escape" });

    expect(onChange).not.toHaveBeenCalled();
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
    expect(trigger).toHaveFocus();
  });

  it("closes when a press lands outside it", () => {
    open();
    fireEvent.pointerDown(document.body);
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
  });
});

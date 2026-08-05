/** The shell's topbar chrome: the status chips it claims about a workspace. */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { AppShell, type PlatformStatus } from "../src/app/AppShell";
import { ToastProvider } from "../src/components/Toast";

const TRAINED: PlatformStatus = { projects: 1 };
const EMPTY: PlatformStatus = { projects: 3 };
const OPEN_RUN = { id: "fvb-fix-s0", name: "fvb-fix-s0" };

function shell(
  status: PlatformStatus,
  project: string | null = "demo",
  runCrumb: { id: string; name: string } | null = null,
) {
  return render(
    <ToastProvider>
      <AppShell
        active="datasets"
        onNavigate={vi.fn()}
        activeRun={null}
        status={status}
        runCrumb={runCrumb}
        project={project}
        onHome={vi.fn()}
      >
        <p>canvas</p>
      </AppShell>
    </ToastProvider>,
  );
}

describe("topbar status chips", () => {
  it("says nothing about stage state inside a project", () => {
    // Two chips lived here: "Stage B · not configured", a hardcoded string that
    // was wrong for any project that had configured it, and "Stage A · trained",
    // which was accurate but repeated the trained badge the series already
    // carries in the library. Neither earned topbar space.
    shell(TRAINED);

    expect(screen.queryByText(/Stage A/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Stage B/)).not.toBeInTheDocument();
    expect(screen.queryByText(/trained/)).not.toBeInTheDocument();
    expect(document.querySelectorAll(".topbar .chip")).toHaveLength(0);
  });

  it("counts the workspace on the projects home", () => {
    shell(EMPTY, null);

    expect(screen.getByText("3 projects")).toBeInTheDocument();
    expect(screen.getByText("0 active")).toBeInTheDocument();
  });

  it("singularises a workspace of one", () => {
    shell(TRAINED, null);
    expect(screen.getByText("1 project")).toBeInTheDocument();
  });
});

describe("topbar actions", () => {
  const ACTIONS = ["Source on GitHub", "Share", "Export report"];

  it("carries each action as a glyph with an accessible name", () => {
    shell(TRAINED);

    for (const label of ACTIONS) {
      const action = screen.getByLabelText(label);
      // Icon-only: the name is on the control, never rendered as a caption.
      expect(action.textContent).toBe("");
      expect(action.querySelector("svg")).not.toBeNull();
    }
    expect(screen.getByLabelText(/Switch to .* theme/)).toBeInTheDocument();
  });

  it("points the source action at the repository", () => {
    shell(TRAINED);
    expect(screen.getByLabelText("Source on GitHub")).toHaveAttribute(
      "href",
      "https://github.com/PouyanJay/naviernet",
    );
  });

  it("has no search box, since the palette answers on its own key", () => {
    // ⌘K is bound globally, so the box was chrome advertising a shortcut
    // rather than a control anyone needed to click.
    shell(TRAINED);
    expect(
      screen.queryByText(/Search or run a command/),
    ).not.toBeInTheDocument();
    expect(document.querySelector(".topbar .search")).toBeNull();
  });
});

describe("the pipeline rail", () => {
  it("puts System at its foot rather than an avatar in the topbar", () => {
    shell(TRAINED);

    const system = screen.getByRole("button", { name: /System/ });
    expect(system.closest(".sidebar")).not.toBeNull();
    expect(system.closest(".topbar")).toBeNull();
    expect(screen.queryByText("PJ")).not.toBeInTheDocument();
  });

  it("opens the System menu on click and names the workspace", async () => {
    shell(TRAINED);
    const system = screen.getByRole("button", { name: /System/ });
    expect(system).toHaveAttribute("aria-expanded", "false");

    fireEvent.click(system);

    expect(system).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText("local workspace")).toBeInTheDocument();
  });

  it("carries the stages and no run metadata", () => {
    // A "Run metadata" block sat at the foot of the rail with three rows.
    // Checkpoint was the literal "ckpt.pt", identical for every run ever
    // trained; Backend was the literal "PyTorch CPU", false under
    // training.device=cuda; Run duplicated the breadcrumb. All three described
    // the latest run rather than the one being viewed.
    shell(TRAINED);

    expect(
      screen.getByRole("button", { name: "Results & validation" }),
    ).toBeInTheDocument();
    expect(screen.queryByText("Run metadata")).not.toBeInTheDocument();
    expect(screen.queryByText("ckpt.pt")).not.toBeInTheDocument();
    expect(screen.queryByText(/PyTorch/)).not.toBeInTheDocument();
  });

  it("names the open run in the breadcrumb", () => {
    shell(TRAINED, "demo", OPEN_RUN);
    expect(document.querySelector(".crumb")!.textContent).toContain(
      OPEN_RUN.name,
    );
  });

  it("names no run on a stage that is not looking at one", () => {
    // Datasets, Physics and Solver have no run in view, so the breadcrumb has
    // nothing to name. It used to print whichever run was newest, on every
    // stage, and kept printing it while Results showed a different one.
    shell(TRAINED, "demo", null);
    const crumb = document.querySelector(".crumb")!;
    expect(crumb.textContent).not.toContain("fvb-fix-s0");
    expect(crumb.querySelector(".mono")).toBeNull();
  });
});

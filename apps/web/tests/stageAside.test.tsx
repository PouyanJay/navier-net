/** The shell's secondary rail: a stage claims it, fills it, and can fold it. */

import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AppShell } from "../src/app/AppShell";
import { StageAside } from "../src/app/StageAside";
import { ToastProvider } from "../src/components/Toast";

function shell(children: React.ReactNode) {
  return render(
    <ToastProvider>
      <AppShell
        active="datasets"
        onNavigate={vi.fn()}
        activeRun={null}
        status={{ latestRun: null, projects: 1 }}
        project="demo"
        onHome={vi.fn()}
      >
        {children}
      </AppShell>
    </ToastProvider>,
  );
}

const RAIL = () => document.querySelector(".shell")!;

beforeEach(() => localStorage.clear());
afterEach(() => vi.restoreAllMocks());

describe("StageAside", () => {
  it("adds no rail for a stage that does not ask for one", () => {
    shell(<p>just the canvas</p>);

    expect(screen.getByText("just the canvas")).toBeInTheDocument();
    expect(RAIL()).not.toHaveAttribute("data-aside");
    expect(document.querySelector(".rail2")).toBeNull();
  });

  it("gives a claiming stage a titled rail and renders its body there", async () => {
    shell(
      <StageAside title="Series library" subtitle="per-series conditions">
        <button type="button">+ Upload new series</button>
      </StageAside>,
    );

    const rail = await screen.findByRole("complementary", {
      name: "Series library",
    });
    expect(rail).toHaveTextContent("per-series conditions");
    // The body is INSIDE the rail, not left in the canvas.
    expect(rail.querySelector(".rail2-body")!.textContent).toContain(
      "+ Upload new series",
    );
    expect(RAIL()).toHaveAttribute("data-aside", "open");
  });

  it("folds away and back, and says which it will do", async () => {
    shell(
      <StageAside title="Series library">
        <p>library body</p>
      </StageAside>,
    );
    await screen.findByRole("complementary", { name: "Series library" });

    const collapse = screen.getByRole("button", {
      name: "Collapse Series library",
    });
    expect(collapse).toHaveAttribute("aria-expanded", "true");
    fireEvent.click(collapse);

    expect(RAIL()).toHaveAttribute("data-aside", "collapsed");
    const expand = screen.getByRole("button", {
      name: "Expand Series library",
    });
    expect(expand).toHaveAttribute("aria-expanded", "false");

    // Folded, not unmounted: whatever the stage has open in there survives.
    expect(screen.getByText("library body")).toBeInTheDocument();

    fireEvent.click(expand);
    expect(RAIL()).toHaveAttribute("data-aside", "open");
  });

  it("remembers the fold across visits", async () => {
    const first = shell(
      <StageAside title="Series library">
        <p>body</p>
      </StageAside>,
    );
    await screen.findByRole("complementary", { name: "Series library" });
    fireEvent.click(
      screen.getByRole("button", { name: "Collapse Series library" }),
    );
    first.unmount();

    shell(
      <StageAside title="Series library">
        <p>body</p>
      </StageAside>,
    );
    await screen.findByRole("complementary", { name: "Series library" });
    expect(RAIL()).toHaveAttribute("data-aside", "collapsed");
  });

  it("gives the rail up when the stage leaves it", async () => {
    const view = shell(
      <StageAside title="Series library">
        <p>body</p>
      </StageAside>,
    );
    await screen.findByRole("complementary", { name: "Series library" });

    view.rerender(
      <ToastProvider>
        <AppShell
          active="physics"
          onNavigate={vi.fn()}
          activeRun={null}
          status={{ latestRun: null, projects: 1 }}
          project="demo"
          onHome={vi.fn()}
        >
          <p>another stage</p>
        </AppShell>
      </ToastProvider>,
    );

    expect(RAIL()).not.toHaveAttribute("data-aside");
    expect(document.querySelector(".rail2")).toBeNull();
  });
});

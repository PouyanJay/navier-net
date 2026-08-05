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
        status={{ projects: 1 }}
        runCrumb={null}
        project="demo"
        onHome={vi.fn()}
      >
        {children}
      </AppShell>
    </ToastProvider>,
  );
}

const SHELL = () => document.querySelector<HTMLElement>(".shell")!;

beforeEach(() => localStorage.clear());
afterEach(() => vi.restoreAllMocks());

describe("StageAside", () => {
  it("adds no rail for a stage that does not ask for one", () => {
    shell(<p>just the canvas</p>);

    expect(screen.getByText("just the canvas")).toBeInTheDocument();
    expect(SHELL()).not.toHaveAttribute("data-aside");
    expect(screen.queryByRole("complementary")).not.toBeInTheDocument();
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
    expect(rail.querySelector(".stage-aside-body")!.textContent).toContain(
      "+ Upload new series",
    );
    expect(SHELL()).toHaveAttribute("data-aside", "open");
  });

  it("sizes the rail to the width the stage asks for", async () => {
    // A stage whose rows are denser than the default (Physics, Solver) states
    // its own width. Dropping it silently is invisible until the stage's own
    // content starts wrapping or overflowing.
    shell(
      <StageAside title="Physics & model" width={384}>
        <p>equations</p>
      </StageAside>,
    );

    await screen.findByRole("complementary", { name: "Physics & model" });
    expect(SHELL().style.getPropertyValue("--stage-aside-w")).toBe("384px");
  });

  it("leaves the width to the shell when a stage does not ask", async () => {
    shell(
      <StageAside title="Series library">
        <p>series</p>
      </StageAside>,
    );

    await screen.findByRole("complementary", { name: "Series library" });
    expect(SHELL().style.getPropertyValue("--stage-aside-w")).toBe("");
  });

  it("folds away and back, and says which it will do", async () => {
    shell(
      <StageAside title="Series library">
        <label>
          draft
          <input defaultValue="" />
        </label>
      </StageAside>,
    );
    await screen.findByRole("complementary", { name: "Series library" });

    // Half-finished work in the rail: a static child could not tell "stayed
    // mounted" apart from "unmounted and remounted with identical markup",
    // which is exactly the difference that matters here.
    const draft = screen.getByLabelText("draft");
    fireEvent.change(draft, { target: { value: "half typed" } });

    const collapse = screen.getByRole("button", {
      name: "Collapse Series library",
    });
    expect(collapse).toHaveAttribute("aria-expanded", "true");
    fireEvent.click(collapse);

    expect(SHELL()).toHaveAttribute("data-aside", "collapsed");
    const expand = screen.getByRole("button", {
      name: "Expand Series library",
    });
    expect(expand).toHaveAttribute("aria-expanded", "false");
    // Folded by CSS, so the state flag and the actual fold cannot drift apart.
    expect(
      getComputedStyle(document.querySelector(".stage-aside-body")!).display,
    ).toBe("none");
    // Focus went to the toggle rather than falling to <body> when the element
    // it was inside was hidden.
    expect(document.activeElement).toBe(expand);

    fireEvent.click(expand);
    expect(SHELL()).toHaveAttribute("data-aside", "open");
    expect(screen.getByLabelText("draft")).toHaveValue("half typed");
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
    expect(SHELL()).toHaveAttribute("data-aside", "collapsed");
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
          status={{ projects: 1 }}
          runCrumb={null}
          project="demo"
          onHome={vi.fn()}
        >
          <p>another stage</p>
        </AppShell>
      </ToastProvider>,
    );

    expect(SHELL()).not.toHaveAttribute("data-aside");
    expect(screen.queryByRole("complementary")).not.toBeInTheDocument();
  });
});

/** The shell's topbar chrome: the status chips it claims about a workspace. */

import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { AppShell, type PlatformStatus } from "../src/app/AppShell";
import { ToastProvider } from "../src/components/Toast";

const TRAINED: PlatformStatus = {
  latestRun: { id: "fvb-fix-s0", name: "fvb-fix-s0", steps: 3000 },
  projects: 1,
};
const EMPTY: PlatformStatus = { latestRun: null, projects: 3 };

function shell(status: PlatformStatus, project: string | null = "demo") {
  return render(
    <ToastProvider>
      <AppShell
        active="datasets"
        onNavigate={vi.fn()}
        activeRun={null}
        status={status}
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

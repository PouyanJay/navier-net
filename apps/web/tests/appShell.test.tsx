/** The shell's topbar chrome: the status chips it claims about a project. */

import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { AppShell, type PlatformStatus } from "../src/app/AppShell";
import { ToastProvider } from "../src/components/Toast";

const TRAINED: PlatformStatus = {
  latestRun: { id: "fvb-fix-s0", name: "fvb-fix-s0", steps: 3000 },
  projects: 1,
};
const UNTRAINED: PlatformStatus = { latestRun: null, projects: 1 };

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
  it("reports Stage A trained when the project has a trained run", () => {
    shell(TRAINED);
    expect(screen.getByText(/Stage A · trained/)).toBeInTheDocument();
  });

  it("reports Stage A untrained when it does not", () => {
    shell(UNTRAINED);
    expect(screen.getByText(/Stage A · untrained/)).toBeInTheDocument();
  });

  it("claims nothing about Stage B", () => {
    // A chip here used to read "Stage B · not configured" unconditionally --
    // a hardcoded string, wrong for every project that had configured it.
    // Nothing the shell receives carries a run's stage, so the honest move is
    // to say nothing rather than to guess.
    shell(TRAINED);
    expect(screen.queryByText(/Stage B/)).not.toBeInTheDocument();
  });

  it("counts the workspace instead of a pipeline on the projects home", () => {
    shell(UNTRAINED, null);
    expect(screen.getByText("1 project")).toBeInTheDocument();
    expect(screen.queryByText(/Stage A/)).not.toBeInTheDocument();
  });
});

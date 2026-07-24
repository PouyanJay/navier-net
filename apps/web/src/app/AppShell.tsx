import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";

import { CommandPalette, type PaletteAction } from "../components/CommandPalette";
import { useToast } from "../components/Toast";
import type { RunJobStatus } from "../lib/api";
import { applyTheme, initialTheme, type Theme } from "../theme";
import "./appshell.css";

export interface NavItem {
  id: string;
  label: string;
  sub: string;
  /** Pipeline stage number; Projects is the workspace home, not a stage. */
  stage?: number;
}

/** The platform's top-level navigation, in the mockup's order. */
export const NAV_ITEMS: NavItem[] = [
  { id: "projects", label: "Projects", sub: "workspace home" },
  { id: "datasets", label: "Datasets & conditions", sub: "frames · calibration", stage: 1 },
  { id: "physics", label: "Physics & model", sub: "equations · architecture", stage: 2 },
  { id: "solver", label: "Solver", sub: "configure & run", stage: 3 },
  { id: "results", label: "Results & validation", sub: "fields · figures · video", stage: 4 },
];

/** What the shell knows about the platform's real state (drives chrome). */
export interface PlatformStatus {
  /** Stage ids that are complete (e.g. datasets processed, model trained). */
  done: Record<string, boolean>;
  /** Latest trained run, for the sidebar's run metadata. */
  latestRun: { id: string; steps: number | null } | null;
  /** Number of projects in the workspace (home-mode chip). */
  projects: number;
}

interface AppShellProps {
  active: string;
  onNavigate: (id: string) => void;
  activeRun: RunJobStatus | null;
  status: PlatformStatus;
  /** Open project id, or null on the workspace home. Drives the rail mode. */
  project: string | null;
  onHome: () => void;
  children: ReactNode;
}

function ThemeIcon({ theme }: { theme: Theme }) {
  return theme === "dark" ? (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <path d="M13 9.5A5.5 5.5 0 0 1 6.5 3 5.5 5.5 0 1 0 13 9.5Z" fill="currentColor" />
    </svg>
  ) : (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <circle cx="8" cy="8" r="3.2" fill="currentColor" />
      {Array.from({ length: 8 }, (_, i) => {
        const a = (i * Math.PI) / 4;
        return (
          <line
            key={i}
            x1={8 + Math.cos(a) * 5}
            y1={8 + Math.sin(a) * 5}
            x2={8 + Math.cos(a) * 6.6}
            y2={8 + Math.sin(a) * 6.6}
            stroke="currentColor"
            strokeWidth="1.4"
            strokeLinecap="round"
          />
        );
      })}
    </svg>
  );
}

/** Avatar trigger + dismissible account panel (outside click / Escape). */
function AccountMenu() {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!open) return;
    const onClick = (event: MouseEvent) => {
      if (!(event.target as Element).closest(".uwrap")) setOpen(false);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div className="uwrap">
      <button
        type="button"
        className="avatar"
        aria-label="Account menu"
        aria-expanded={open}
        aria-controls="account-menu"
        onClick={() => setOpen((current) => !current)}
      >
        PJ
      </button>
      {open && (
        <div className="umenu" id="account-menu">
          <div className="uh">
            <b>Pouyan Jahangiri</b>
            <span>local workspace</span>
          </div>
          <a
            className="uitem"
            href="https://github.com/PouyanJay/naviernet"
            target="_blank"
            rel="noreferrer"
            onClick={() => setOpen(false)}
          >
            Documentation
          </a>
        </div>
      )}
    </div>
  );
}

/** Topbar status chips: workspace counts at home, stage status in a project. */
function StatusChips({
  project,
  trained,
  projectCount,
  running,
}: {
  project: string | null;
  trained: boolean;
  projectCount: number;
  running: boolean;
}) {
  if (project) {
    return (
      <>
        <span className="chip" data-tone={trained ? "green" : undefined}>
          <span className="cdot" aria-hidden="true" />
          Stage A · {trained ? "trained" : "untrained"}
        </span>
        <span className="chip" data-tone="amber">
          <span className="cdot" aria-hidden="true" />
          Stage B · not configured
        </span>
      </>
    );
  }
  return (
    <>
      <span className="chip">
        <span className="cdot" aria-hidden="true" />
        {projectCount} project{projectCount === 1 ? "" : "s"}
      </span>
      <span className="chip">
        <span className="cdot" aria-hidden="true" />
        {running ? "1 active" : "0 active"}
      </span>
    </>
  );
}

/** Dark rail: "← All projects" always; pipeline stages only inside a project. */
function Sidebar({
  project,
  active,
  status,
  onNavigate,
  onHome,
}: {
  project: string | null;
  active: string;
  status: PlatformStatus;
  onNavigate: (id: string) => void;
  onHome: () => void;
}) {
  return (
    <nav className="sidebar" aria-label="Primary">
      <button type="button" className="backhome" onClick={onHome} aria-label="All projects">
        <span aria-hidden="true">←</span>
        <span className="backhome-label">All projects</span>
      </button>
      {project ? (
        <>
          <div className="raillbl">Reconstruction pipeline</div>
          <div className="nav">
            {NAV_ITEMS.filter((item) => item.stage).map((item) => (
              <button
                key={item.id}
                type="button"
                className="stage"
                aria-label={item.label}
                aria-current={item.id === active ? "page" : undefined}
                onClick={() => onNavigate(item.id)}
              >
                <span className="node" data-done={status.done[item.id] || undefined}>
                  {status.done[item.id] ? "✓" : item.stage}
                </span>
                <span className="txt">
                  <b>{item.label}</b>
                  <span>{item.sub}</span>
                </span>
              </button>
            ))}
          </div>
          <div className="spacer" />
          <div className="railfoot">
            <div className="raillbl">Run metadata</div>
            <div className="kv">
              <span>Checkpoint</span>
              <span className="mono">{status.latestRun ? "ckpt.pt" : "—"}</span>
            </div>
            <div className="kv">
              <span>Run</span>
              <span className="mono">{status.latestRun?.id ?? "—"}</span>
            </div>
            <div className="kv">
              <span>Backend</span>
              <span className="mono">PyTorch CPU</span>
            </div>
          </div>
        </>
      ) : (
        <div className="raillbl">Workspace</div>
      )}
    </nav>
  );
}

/** Fixed dark chrome (brand + rail) and top bar around the paper workspace. */
export function AppShell({
  active,
  onNavigate,
  activeRun,
  status,
  project,
  onHome,
  children,
}: AppShellProps) {
  const [theme, setTheme] = useState<Theme>(() => initialTheme());
  const [paletteOpen, setPaletteOpen] = useState(false);
  const toast = useToast();

  const toggleTheme = useCallback(() => {
    setTheme((current) => {
      const next: Theme = current === "dark" ? "light" : "dark";
      applyTheme(next);
      return next;
    });
  }, []);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setPaletteOpen((open) => !open);
      }
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "j") {
        event.preventDefault();
        toggleTheme();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggleTheme]);

  const paletteActions = useMemo<PaletteAction[]>(
    () => [
      { group: "Navigate", label: "All projects", shortcut: "0", run: onHome },
      ...NAV_ITEMS.filter((item) => item.stage).map((item) => ({
        group: "Navigate",
        label: item.label,
        shortcut: String(item.stage),
        run: () =>
          project
            ? onNavigate(item.id)
            : toast("Open a project first", "pipeline stages are project-scoped"),
      })),
      { group: "Appearance", label: "Toggle dark mode", shortcut: "⌘J", run: toggleTheme },
    ],
    [onNavigate, toggleTheme, project, onHome, toast],
  );

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brandblock">
          <img className="mark" src="/navnet.png" alt="" />
          <div>
            NavierNet
            <small>PINN Solver Platform</small>
          </div>
        </div>
        <nav className="crumb" aria-label="Breadcrumb">
          <span>Two-phase flows workspace</span>
          <span className="crumb-sep" aria-hidden="true">
            /
          </span>
          <b>{project ?? "Projects"}</b>
          {project && status.latestRun && status.latestRun.id !== project && (
            <>
              <span className="crumb-sep" aria-hidden="true">
                /
              </span>
              <span className="mono">{status.latestRun.id}</span>
            </>
          )}
        </nav>
        <div className="topbar-spacer" />
        <button type="button" className="search" onClick={() => setPaletteOpen(true)}>
          <span>Search or run a command…</span>
          <span className="kbds" aria-hidden="true">
            <kbd>⌘</kbd>
            <kbd>K</kbd>
          </span>
        </button>
        {activeRun?.state === "running" && (
          <button type="button" className="runpill" onClick={() => onNavigate("solver")}>
            <span className="pdot" aria-hidden="true" />
            Training · <span className="mono">{activeRun.run_id}</span>
          </button>
        )}
        <StatusChips
          project={project}
          trained={Boolean(status.done.solver)}
          projectCount={status.projects}
          running={activeRun?.state === "running"}
        />
        <button
          type="button"
          className="thbtn"
          onClick={toggleTheme}
          aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
        >
          <ThemeIcon theme={theme} />
        </button>
        <button
          type="button"
          className="btn"
          onClick={() =>
            toast("Sharing is not available yet", "this workspace is local to your machine")
          }
        >
          Share
        </button>
        <button
          type="button"
          className="btn primary"
          onClick={() =>
            toast("Report export is not available yet", "planned: PDF with config + figures")
          }
        >
          Export report
        </button>
        <AccountMenu />
      </header>

      <Sidebar
        project={project}
        active={active}
        status={status}
        onNavigate={onNavigate}
        onHome={onHome}
      />

      <main className="workspace">
        <div className="page">{children}</div>
      </main>

      <CommandPalette
        open={paletteOpen}
        onClose={() => setPaletteOpen(false)}
        actions={paletteActions}
      />
    </div>
  );
}

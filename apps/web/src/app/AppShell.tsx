import { useEffect, useMemo, useState, type ReactNode } from "react";

import { CommandPalette, type PaletteAction } from "../components/CommandPalette";
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
}

interface AppShellProps {
  active: string;
  onNavigate: (id: string) => void;
  activeRun: RunJobStatus | null;
  status: PlatformStatus;
  children: ReactNode;
}

function BrandMark() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <path
        d="M2 10 C4 6 6 6 8 9 C10 12 12 12 14 7"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
      />
    </svg>
  );
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

/** Fixed dark chrome (brand + rail) and top bar around the paper workspace. */
export function AppShell({ active, onNavigate, activeRun, status, children }: AppShellProps) {
  const [theme, setTheme] = useState<Theme>(() => initialTheme());
  const [paletteOpen, setPaletteOpen] = useState(false);

  function toggleTheme() {
    const next: Theme = theme === "dark" ? "light" : "dark";
    setTheme(next);
    applyTheme(next);
  }

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
  });

  const paletteActions = useMemo<PaletteAction[]>(
    () => [
      ...NAV_ITEMS.map((item, i) => ({
        group: "Navigate",
        label: `Open ${item.label}`,
        shortcut: String(i),
        run: () => onNavigate(item.id),
      })),
      { group: "Appearance", label: "Toggle dark mode", shortcut: "⌘J", run: toggleTheme },
    ],
    // eslint-disable-next-line react-hooks/exhaustive-deps -- toggleTheme reads current theme
    [onNavigate, theme],
  );

  const activeLabel = NAV_ITEMS.find((item) => item.id === active)?.label ?? "";
  const trained = status.done.solver;

  return (
    <div className="shell">
      <div className="brandblock">
        <span className="mark" aria-hidden="true">
          <BrandMark />
        </span>
        <div>
          NavierNet
          <small>PINN Solver Platform</small>
        </div>
      </div>

      <header className="topbar">
        <nav className="crumb" aria-label="Breadcrumb">
          <span>Two-phase flows workspace</span>
          <span className="crumb-sep" aria-hidden="true">
            /
          </span>
          <b>{activeLabel}</b>
        </nav>
        <button type="button" className="search" onClick={() => setPaletteOpen(true)}>
          <span>Search or run a command…</span>
          <span className="kbds" aria-hidden="true">
            <kbd>⌘</kbd>
            <kbd>K</kbd>
          </span>
        </button>
        <div className="topbar-spacer" />
        {activeRun?.state === "running" && (
          <button type="button" className="runpill" onClick={() => onNavigate("solver")}>
            <span className="pdot" aria-hidden="true" />
            Training · <span className="mono">{activeRun.run_id}</span>
          </button>
        )}
        <span className="chip" data-tone={trained ? "green" : undefined}>
          Stage A · {trained ? "trained" : "untrained"}
        </span>
        <span className="chip" data-tone="amber">
          Stage B · not configured
        </span>
        <button
          type="button"
          className="thbtn"
          onClick={toggleTheme}
          aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
        >
          <ThemeIcon theme={theme} />
        </button>
      </header>

      <nav className="sidebar" aria-label="Primary">
        <div className="nav">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              type="button"
              className={item.stage ? "stage" : "homelink"}
              aria-label={item.label}
              aria-current={item.id === active ? "page" : undefined}
              onClick={() => onNavigate(item.id)}
            >
              {item.stage && (
                <span className="node" data-done={status.done[item.id] || undefined}>
                  {status.done[item.id] ? "✓" : item.stage}
                </span>
              )}
              <span className="txt">
                <b>{item.label}</b>
                <span>{item.sub}</span>
              </span>
            </button>
          ))}
        </div>
        <div className="spacer" />
        <div className="railfoot">
          <div className="lbl">Run metadata</div>
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
      </nav>

      <main className="workspace">{children}</main>

      <CommandPalette
        open={paletteOpen}
        onClose={() => setPaletteOpen(false)}
        actions={paletteActions}
      />
    </div>
  );
}

import { useState, type ReactNode } from "react";

import { applyTheme, initialTheme, type Theme } from "../theme";
import "./appshell.css";

export interface NavItem {
  id: string;
  label: string;
}

/** The platform's top-level navigation, in the mockup's order. */
export const NAV_ITEMS: NavItem[] = [
  { id: "projects", label: "Projects" },
  { id: "datasets", label: "Datasets & conditions" },
  { id: "physics", label: "Physics & model" },
  { id: "solver", label: "Solver" },
  { id: "results", label: "Results & validation" },
];

interface AppShellProps {
  active: string;
  onNavigate: (id: string) => void;
  children: ReactNode;
}

/** Fixed dark sidebar + scrolling paper workspace (enterprise-ui §2). */
export function AppShell({ active, onNavigate, children }: AppShellProps) {
  const [theme, setTheme] = useState<Theme>(() => initialTheme());

  function toggleTheme() {
    const next: Theme = theme === "dark" ? "light" : "dark";
    setTheme(next);
    applyTheme(next);
  }

  return (
    <div className="shell">
      <nav className="sidebar" aria-label="Primary">
        <div className="brand">
          <span className="mark" aria-hidden="true" />
          NavierNet
        </div>
        <div className="nav">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              type="button"
              aria-current={item.id === active ? "page" : undefined}
              onClick={() => onNavigate(item.id)}
            >
              {item.label}
            </button>
          ))}
        </div>
        <div className="spacer" />
        <button
          type="button"
          className="theme-toggle"
          onClick={toggleTheme}
          aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
        >
          <span>Theme</span>
          <span className="mono">{theme === "dark" ? "dark" : "light"}</span>
        </button>
      </nav>
      <main className="workspace">{children}</main>
    </div>
  );
}

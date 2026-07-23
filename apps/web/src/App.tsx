import { useState } from "react";

import { AppShell, NAV_ITEMS } from "./app/AppShell";
import { Placeholder } from "./views/Placeholder";
import { ResultsView } from "./views/ResultsView";

const PAGE_TITLE: Record<string, string> = Object.fromEntries(
  NAV_ITEMS.map((item) => [item.id, item.label]),
);

export function App() {
  // The skeleton lands on Results & validation, the first surface with real data.
  const [active, setActive] = useState("results");

  return (
    <AppShell active={active} onNavigate={setActive}>
      <header className="pagehead">
        <h1>{PAGE_TITLE[active]}</h1>
        {active === "results" && (
          <p>
            Solver runs and their validation against the measured bubble. Every number
            is read live from the pipeline&apos;s own artifacts.
          </p>
        )}
      </header>
      <div className="stack">
        {active === "results" ? <ResultsView /> : <Placeholder title={PAGE_TITLE[active]} />}
      </div>
    </AppShell>
  );
}

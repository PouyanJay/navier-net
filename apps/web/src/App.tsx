import { useState } from "react";

import { AppShell, NAV_ITEMS } from "./app/AppShell";
import { DatasetsView } from "./views/DatasetsView";
import { Placeholder } from "./views/Placeholder";
import { ProjectsView } from "./views/ProjectsView";
import { ResultsView } from "./views/ResultsView";

const PAGE_TITLE: Record<string, string> = Object.fromEntries(
  NAV_ITEMS.map((item) => [item.id, item.label]),
);

const PAGE_INTRO: Record<string, string> = {
  projects:
    "Each imaged dataset is a project. Open one to inspect its conditions, upload frames, and preprocess.",
  datasets:
    "Operating conditions, derived dimensionless groups, the raw image sequence, and calibration/segmentation.",
  results:
    "Solver runs and their validation against the measured bubble. Every number is read live from the pipeline's own artifacts.",
};

export function App() {
  const [active, setActive] = useState("results");
  const [datasetId, setDatasetId] = useState<string | null>(null);

  function openDataset(id: string) {
    setDatasetId(id);
    setActive("datasets");
  }

  return (
    <AppShell active={active} onNavigate={setActive}>
      <header className="pagehead">
        <h1>{PAGE_TITLE[active]}</h1>
        {PAGE_INTRO[active] && <p>{PAGE_INTRO[active]}</p>}
      </header>
      <div className="stack">
        {active === "results" && <ResultsView />}
        {active === "projects" && <ProjectsView onOpen={openDataset} />}
        {active === "datasets" && <DatasetsView datasetId={datasetId} />}
        {active !== "results" && active !== "projects" && active !== "datasets" && (
          <Placeholder title={PAGE_TITLE[active]} />
        )}
      </div>
    </AppShell>
  );
}

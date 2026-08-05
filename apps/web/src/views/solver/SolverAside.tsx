import type { DatasetSummary } from "../../lib/api";
import { seriesNameOf } from "../../lib/series";
import type { SolverFormState } from "./form";
import { treatmentMeta, treatmentOf } from "./formulation";
import { LaunchDock } from "./LaunchDock";
import { ObjectiveBand } from "./ObjectiveBand";
import { PhysicsBand } from "./PhysicsBand";
import { RunBand, type LaunchMode } from "./RunBand";
import { TrainingBand } from "./TrainingBand";
import type { RunTargets } from "./useRunTargets";
import type { SeriesCapability } from "./useSeriesCapability";

interface SolverAsideProps {
  form: SolverFormState;
  onForm: (patch: Partial<SolverFormState>) => void;
  mode: LaunchMode;
  onMode: (mode: LaunchMode) => void;
  targets: RunTargets;
  capability: SeriesCapability;
  seedsText: string;
  onSeedsText: (text: string) => void;
  /** The parsed seed list, or null when the text does not parse. */
  seeds: number[] | null;
  running: boolean;
  onReset: () => void;
  onRun: () => void;
}

/**
 * The Solver's rail, as four bands and a launch dock.
 *
 * The bands answer four different questions, in the order a run is decided:
 * what it trains on, what physics it is given, what it is scored on, and how
 * the optimizer walks that score. Everything that used to sit in one 30-control
 * card is now in the band whose question it answers, and nothing that does not
 * apply to the current choices is rendered at all.
 */
export function SolverAside({
  form,
  onForm,
  mode,
  onMode,
  targets,
  capability,
  seedsText,
  onSeedsText,
  seeds,
  running,
  onReset,
  onRun,
}: SolverAsideProps) {
  const available = targets.available ?? [];
  // On resume the checkpoint's own config owns every field but Steps.
  const fixed = running || mode === "resume";

  return (
    <>
      <RunBand
        form={form}
        onForm={onForm}
        mode={mode}
        onMode={onMode}
        available={available}
        selected={targets.selected}
        onToggle={targets.toggleDataset}
        onSelectAll={targets.selectAll}
        heldout={targets.heldout}
        onToggleHeldout={targets.toggleHeldout}
        resumableRuns={targets.resumableRuns}
        resumeRunId={targets.resumeRunId}
        onResumeRunId={targets.setResumeRunId}
        seedsText={seedsText}
        onSeedsText={onSeedsText}
        seedsValid={seeds !== null}
        locked={running}
      />
      <PhysicsBand
        form={form}
        onForm={onForm}
        capability={capability}
        locked={fixed}
      />
      <ObjectiveBand form={form} onForm={onForm} locked={fixed} />
      <TrainingBand
        form={form}
        onForm={onForm}
        locked={running}
        fixed={fixed}
      />
      <LaunchDock
        summary={runSummary(form, mode, targets, seeds, available)}
        blocked={launchBlocked(form, mode, targets, seeds, running)}
        running={running}
        onReset={onReset}
        onRun={onRun}
      />
    </>
  );
}

/** What this configuration amounts to, for the line above Run. */
export function runSummary(
  form: SolverFormState,
  mode: LaunchMode,
  targets: RunTargets,
  seeds: number[] | null,
  available: DatasetSummary[],
): string[] {
  const parts: string[] = [];
  if (mode === "resume") {
    parts.push(
      targets.resumeRunId ? `resume ${targets.resumeRunId}` : "resume",
    );
  } else if (mode === "sweep") {
    parts.push(`${seeds?.length ?? 0} seeds`);
  } else if (targets.selected.length === 1) {
    parts.push(seriesNameOf(available, targets.selected[0]));
  } else {
    parts.push(`${targets.selected.length} series`);
  }
  if (mode === "new" && targets.heldout) parts.push("1 held out");
  parts.push(`${form.steps} steps`);
  parts.push(treatmentMeta(treatmentOf(form)).label.toLowerCase());
  return parts;
}

/**
 * Why Run cannot be pressed, in the words of the thing that is missing.
 *
 * The formulation's own invalid combinations are unreachable by construction
 * (the ladder clears what it does not admit), so what is left here is a target
 * that is not chosen yet, and the one combination a user can still type: a
 * measurement supervised at weight zero, which the API rejects for measuring
 * the front and then ignoring it.
 */
export function launchBlocked(
  form: SolverFormState,
  mode: LaunchMode,
  targets: RunTargets,
  seeds: number[] | null,
  running: boolean,
): string | null {
  if (running) return "A run is already in progress.";
  if (mode === "resume") {
    return targets.resumeRunId ? null : "Choose a run to resume.";
  }
  if (targets.selected.length === 0) {
    return mode === "sweep"
      ? "Select the series the sweep trains on."
      : "Select at least one series to train on.";
  }
  if (mode === "sweep" && seeds === null) {
    return "Give 1-6 unique, non-negative seeds.";
  }
  if (
    form.front_velocity &&
    form.fv_weight === 0 &&
    form.fv_apex_weight === 0
  ) {
    return "Measured front velocity needs a weight on the profile or the apex.";
  }
  return null;
}

import { useMemo, useState, type KeyboardEvent } from "react";

import { Button, Panel } from "../../components";
import type { RunSummary } from "../../lib/api";
import { isTrainedRun, MAX_COMPARED } from "../../lib/runs";
import {
  formatRunDate,
  RANK_METRICS,
  rankMetric,
  runDisplayName,
  runHeadline,
  runProvenance,
  type RankMetricId,
} from "./format";
import { buildFamilies, type RunFamily } from "./runFamilies";

/** How a run's state reads without relying on its dot's colour. */
const STATE_WORD: Record<RunSummary["status"], string | null> = {
  running: "training",
  trained: null,
  failed: "failed",
  empty: "no metrics",
};

interface RunRailProps {
  runs: RunSummary[];
  datasetLabels: Map<string, string>;
  selectedId: string | null;
  onOpen: (id: string) => void;
  /** Open the Compare tab on these runs. */
  onCompare: (ids: string[]) => void;
}

/**
 * The project's runs as a ranked list of experiments.
 *
 * Three things it does that a folder listing cannot: it leads every row with the
 * best metric that run actually recorded (and names which one), it folds seed
 * replicates into the experiment they belong to so their spread is visible, and
 * it lets a comparison be assembled here rather than in the tab that draws it.
 */
export function RunRail({
  runs,
  datasetLabels,
  selectedId,
  onOpen,
  onCompare,
}: RunRailProps) {
  const [metric, setMetric] = useState<RankMetricId>("val");
  const [query, setQuery] = useState("");
  const [opened, setOpened] = useState<Set<string>>(new Set());
  const [picked, setPicked] = useState<string[]>([]);

  const families = useMemo(
    () => buildFamilies(runs, { metric, query }),
    [runs, metric, query],
  );

  const togglePick = (id: string) =>
    setPicked((current) =>
      current.includes(id)
        ? current.filter((other) => other !== id)
        : current.length >= MAX_COMPARED
          ? current
          : [...current, id],
    );

  const shown = families.reduce((n, family) => n + family.runs.length, 0);
  const subtitle =
    shown === runs.length
      ? `${families.length} experiments · ${runs.length} runs`
      : `${shown} of ${runs.length} runs`;

  return (
    <Panel title="Runs" subtitle={subtitle}>
      <div className="rail-tools">
        <input
          className="rail-search"
          type="search"
          value={query}
          placeholder="Filter by name or recipe"
          aria-label="Filter runs"
          onChange={(event) => setQuery(event.target.value)}
        />
        {/* The metric a row leads with is also the one it is ranked by, so this
            is one control doing both jobs — and the rows themselves label the
            number, which is where that label belongs. */}
        <div
          className="seg compact rail-lead"
          role="group"
          aria-label="Lead and sort runs by"
        >
          {RANK_METRICS.map((option) => (
            <button
              key={option.id}
              type="button"
              className={option.id === metric ? "segb on" : "segb"}
              aria-pressed={option.id === metric}
              onClick={() => setMetric(option.id)}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>

      {families.length === 0 ? (
        <p className="res-quiet">
          No run matches “{query}”. Recipes are searchable too — try “causal”.
        </p>
      ) : (
        <div
          className="runlist"
          role="listbox"
          aria-label="Runs of this project"
          onKeyDown={moveRunFocus}
        >
          {families.map((family, index) => (
            <FamilyRow
              key={family.name + (family.recipe ?? []).join()}
              family={family}
              metric={metric}
              datasetLabels={datasetLabels}
              selectedId={selectedId}
              picked={picked}
              open={opened.has(family.name)}
              // Under a date sort the list is a diary, so each day is ruled off
              // once instead of every row repeating the date.
              daySeparator={
                metric === "date" &&
                dayOf(family) !== dayOf(families[index - 1])
                  ? dayOf(family)
                  : null
              }
              onOpen={onOpen}
              onToggle={() =>
                setOpened((current) => {
                  const next = new Set(current);
                  if (!next.delete(family.name)) next.add(family.name);
                  return next;
                })
              }
              onPick={togglePick}
            />
          ))}
        </div>
      )}

      <div className="rail-dock">
        <p>
          {picked.length === 0
            ? "Tick runs to compare"
            : picked.length === 1
              ? "1 picked · pick one more"
              : `${picked.length} of ${MAX_COMPARED} picked`}
        </p>
        <Button
          size="sm"
          variant="primary"
          disabled={picked.length < 2}
          onClick={() => onCompare(picked)}
        >
          Compare
        </Button>
      </div>
    </Panel>
  );
}

const dayOf = (family: RunFamily | undefined) =>
  family ? (formatRunDate(family.newest)?.split(" · ")[0] ?? "") : "";

/** How far a training run has got, for its row's bar. */
const progressPct = (run: RunSummary) =>
  run.steps_total ? ((run.steps_done ?? 0) / run.steps_total) * 100 : 0;

interface FamilyRowProps {
  family: RunFamily;
  metric: RankMetricId;
  datasetLabels: Map<string, string>;
  selectedId: string | null;
  picked: string[];
  open: boolean;
  daySeparator: string | null;
  onOpen: (id: string) => void;
  onToggle: () => void;
  onPick: (id: string) => void;
}

/** One experiment: its rank, its recipe, its score and the gap to the leader.
 * A multi-seed experiment opens to its members. */
function FamilyRow({
  family,
  metric,
  datasetLabels,
  selectedId,
  picked,
  open,
  daySeparator,
  onOpen,
  onToggle,
  onPick,
}: FamilyRowProps) {
  const lead = family.runs[0];
  const many = family.runs.length > 1;
  const holds = family.runs.some((run) => run.id === selectedId);
  const live = lead.status === "running" ? runHeadline(lead) : null;
  const label =
    metric === "date"
      ? (runHeadline(lead)?.label ?? "")
      : rankMetric(metric).label;
  // Nothing is ranked under a date sort, so no row is "best" and none is behind
  // another; the number is simply labelled with the metric it is.
  const ranked = metric !== "date";
  const leads =
    ranked &&
    !live &&
    family.score != null &&
    !family.incomparable &&
    family.behind === null;
  // An experiment is compared through the run that leads it — comparing four
  // seeds of the same recipe fills the chart with one line four times. A
  // specific seed can still be swapped in from the Compare tab's own picker.
  const representative = family.runs.find(isTrainedRun) ?? null;
  const isPicked = representative != null && picked.includes(representative.id);

  return (
    <>
      {daySeparator && <p className="rail-day">{daySeparator}</p>}
      <div className={holds ? "famrow sel" : "famrow"}>
        {/* The picker is its own control: ticking a run for comparison must not
            also navigate away from the one being read. */}
        <button
          type="button"
          role="checkbox"
          aria-checked={isPicked}
          aria-label={`Compare ${family.name}`}
          className={isPicked ? "famrow-pick on" : "famrow-pick"}
          disabled={representative === null}
          onClick={() => representative && onPick(representative.id)}
        />
        <button
          type="button"
          role="option"
          aria-selected={holds}
          className="famrow-open"
          onClick={() => {
            onOpen(lead.id);
            if (many) onToggle();
          }}
        >
          {family.rank != null && (
            <span className="famrow-rank mono">{family.rank}</span>
          )}
          <span className="famrow-nm">
            <b>{runDisplayName(lead, datasetLabels)}</b>
            <span className="famrow-chips">
              {many && (
                <span className="rchip seeds">{family.runs.length} seeds</span>
              )}
              {family.recipe === null ? (
                <span className="rchip">config not recorded</span>
              ) : family.recipe.length === 0 ? (
                <span className="rchip">default recipe</span>
              ) : (
                family.recipe.map((chip) => (
                  <span className="rchip on" key={chip}>
                    {chip}
                  </span>
                ))
              )}
              {runProvenance(lead).map((chip) => (
                <span className="rchip" key={chip}>
                  {chip}
                </span>
              ))}
              {family.incomparable && (
                <span className="rchip warn">{family.incomparable}</span>
              )}
              {STATE_WORD[lead.status] && (
                <span
                  className={
                    lead.status === "failed" ? "rchip bad" : "rchip live"
                  }
                >
                  {STATE_WORD[lead.status]}
                </span>
              )}
              {metric !== "date" && (
                <span className="rchip quiet">
                  {formatRunDate(family.newest)?.split(" · ")[0]}
                </span>
              )}
            </span>
          </span>
          {/* A run still training has no score to rank, and its own number is
              how far along it is — so the value column becomes its progress and
              the bar becomes its progress bar. */}
          <span className="famrow-val mono">
            <b>
              {live
                ? live.value
                : family.score != null
                  ? family.score.toFixed(3)
                  : "—"}
              {!live && family.spread != null && (
                <small> ± {family.spread.toFixed(3)}</small>
              )}
            </b>
            <small className={leads ? "best" : ""}>
              {live
                ? live.label
                : family.score == null
                  ? label || "not evaluated"
                  : leads
                    ? `best ${label}`
                    : ranked && family.behind != null
                      ? `−${family.behind.toFixed(3)}`
                      : label}
            </small>
            {(live || family.score != null) && (
              <span className="famrow-bar" aria-hidden="true">
                <i
                  className={leads ? "best" : ""}
                  style={{
                    width: live
                      ? `${progressPct(lead)}%`
                      : `${family.bar * 100}%`,
                  }}
                />
              </span>
            )}
          </span>
        </button>
      </div>

      {open && many && (
        <ul className="famrow-kids">
          {family.runs.map((run) => {
            const own = runHeadline(run);
            return (
              <li key={run.id}>
                <button
                  type="button"
                  className={run.id === selectedId ? "kid sel" : "kid"}
                  onClick={() => onOpen(run.id)}
                >
                  <span className="kid-nm mono">{run.id}</span>
                  {run.seed != null && (
                    <span className="kid-seed mono">seed {run.seed}</span>
                  )}
                  <span className="kid-val mono">{own?.value ?? "—"}</span>
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </>
  );
}

/** Arrow keys walk the run list, matching the command palette's listbox. */
function moveRunFocus(event: KeyboardEvent<HTMLDivElement>) {
  if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
  const options = [
    ...event.currentTarget.querySelectorAll<HTMLButtonElement>(
      "[role='option']",
    ),
  ];
  const index = options.indexOf(document.activeElement as HTMLButtonElement);
  if (index === -1) return;
  event.preventDefault();
  const next =
    options[
      (index + (event.key === "ArrowDown" ? 1 : options.length - 1)) %
        options.length
    ];
  next.focus();
}

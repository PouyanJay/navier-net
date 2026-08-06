/** Grouping the run list into experiments, and ranking them.
 *
 * A seed sweep exists to produce a spread, and the list showed its replicates as
 * unrelated rows — two numbers where the interesting quantity is the distance
 * between them. This module folds runs that ran the SAME recipe into one row
 * carrying that spread, ranks the rows by the chosen metric, and marks the ones
 * that cannot honestly be ranked against the rest.
 *
 * Grouping keys on the recorded recipe rather than the id, because the id lies:
 * only sweep children are named by the machine (`<sweep_id>-s<seed>`), and a
 * hand-typed `-s2` suffix can disagree with the seed the run itself recorded.
 */

import type { RunSummary } from "../../lib/api";
import { rankMetric, type RankMetricId } from "./format";

export interface RunFamily {
  /** Display name: the shared id stem, or the single run's id. */
  name: string;
  /** The runs in it, best-first under the current metric. */
  runs: RunSummary[];
  /** Mean of the metric across the scored runs, or null when none scored. */
  score: number | null;
  /** Half the spread across seeds, or null for a single run. */
  spread: number | null;
  /** Newest member's timestamp, for the date ordering. */
  newest: string;
  /** Rank under the current metric (1-based); null when it has no score. */
  rank: number | null;
  /** Gap to the leader, or null for the leader itself and for unranked rows. */
  behind: number | null;
  /** The metric bar's width, 0-1, scaled to the list's own range. */
  bar: number;
  /** Why this row cannot be ranked against the others, or null. */
  incomparable: string | null;
  /** The recipe chips, shared by construction; null = no config recorded. */
  recipe: string[] | null;
}

/** The id stem a family is named after: `fvb-fix-s0` → `fvb-fix`. Only used for
 * the LABEL — the grouping itself is by recipe, never by name. */
export function familyStem(id: string): string {
  return id.replace(/-s\d+$/, "");
}

/** The frame count the majority of scored runs were evaluated over. A run
 * measured over a different number of frames is a different measurement. */
function commonFrameCount(runs: RunSummary[]): number | null {
  const counts = new Map<number, number>();
  for (const run of runs) {
    if (run.n_frames == null) continue;
    counts.set(run.n_frames, (counts.get(run.n_frames) ?? 0) + 1);
  }
  let best: number | null = null;
  let seen = 0;
  for (const [frames, n] of counts) {
    if (n > seen) {
      best = frames;
      seen = n;
    }
  }
  return best;
}

/** The grouping key: same recipe and same id stem is the same experiment. The
 * stem keeps two unrelated runs that happen to share a recipe apart, which is
 * the common case in a project where every run is the recommended one. */
function key(run: RunSummary): string {
  const recipe =
    run.recipe === null ? "unrecorded" : (run.recipe ?? []).join("+");
  return `${familyStem(run.id)}::${recipe}`;
}

export interface FamilyOptions {
  metric: RankMetricId;
  /** Free-text filter over the id and the recipe chips. */
  query?: string;
}

/** Fold runs into experiments, ranked under the chosen metric. */
export function buildFamilies(
  runs: RunSummary[],
  { metric, query = "" }: FamilyOptions,
): RunFamily[] {
  const scoreOf = rankMetric(metric).of;
  const basis = commonFrameCount(runs);

  const groups = new Map<string, RunSummary[]>();
  for (const run of runs) {
    const bucket = groups.get(key(run));
    if (bucket) bucket.push(run);
    else groups.set(key(run), [run]);
  }

  let families: RunFamily[] = [...groups.values()].map((members) => {
    const scored = members
      .map((run) => scoreOf(run))
      .filter((value): value is number => value != null);
    const score = scored.length
      ? scored.reduce((sum, value) => sum + value, 0) / scored.length
      : null;
    const odd = members.find(
      (run) => basis != null && run.n_frames != null && run.n_frames !== basis,
    );
    return {
      name: familyStem(members[0].id),
      runs: [...members].sort(
        (a, b) => (scoreOf(b) ?? -1) - (scoreOf(a) ?? -1),
      ),
      score,
      spread:
        scored.length > 1
          ? (Math.max(...scored) - Math.min(...scored)) / 2
          : null,
      newest: members.reduce(
        (latest, run) => (run.date && run.date > latest ? run.date : latest),
        "",
      ),
      rank: null,
      behind: null,
      bar: 0,
      incomparable: odd ? `${odd.n_frames} frames · not comparable` : null,
      recipe: members[0].recipe ?? null,
    };
  });

  const needle = query.trim().toLowerCase();
  if (needle) {
    families = families.filter(
      (family) =>
        family.runs.some((run) => run.id.toLowerCase().includes(needle)) ||
        (family.recipe ?? []).some((chip) =>
          chip.toLowerCase().includes(needle),
        ),
    );
  }

  families.sort((a, b) => {
    if (metric === "date") return b.newest.localeCompare(a.newest);
    if (a.score == null) return b.score == null ? 0 : 1;
    if (b.score == null) return -1;
    // A run measured on a different basis is not part of this ranking, so it
    // sorts below it rather than taking the top slot on a number nobody else
    // was scored against. Under a date sort nothing is ranked and it stays put.
    if (Boolean(a.incomparable) !== Boolean(b.incomparable)) {
      return a.incomparable ? 1 : -1;
    }
    return b.score - a.score;
  });

  return rank(families, metric);
}

/**
 * Rank, gap-to-leader, and the bar.
 *
 * The leader is drawn from the comparable rows only: a run measured over a
 * different frame count may still be listed and scored, but it never becomes
 * the datum every other row is reported as short of.
 */
function rank(families: RunFamily[], metric: RankMetricId): RunFamily[] {
  const comparable = families.filter(
    (family) => family.score != null && !family.incomparable,
  );
  const best = comparable.length
    ? Math.max(...comparable.map((family) => family.score as number))
    : null;
  const scores = families
    .map((family) => family.score)
    .filter((value): value is number => value != null);
  const floor = scores.length ? Math.min(...scores) : 0;
  const span = best != null && best > floor ? best - floor : 1;

  let position = 0;
  return families.map((family) => {
    if (family.score == null) return family;
    // Only comparable rows are numbered. An incomparable one keeps its place in
    // score order — that is honest — but claiming a rank would assert exactly
    // the comparison its own chip says cannot be made.
    if (!family.incomparable) position += 1;
    return {
      ...family,
      rank: metric === "date" || family.incomparable ? null : position,
      // Under a date sort nothing is ranked, so nothing is "behind" anything:
      // the row states which metric it is showing instead.
      behind:
        metric === "date" ||
        best == null ||
        family.incomparable ||
        family.score >= best
          ? null
          : best - family.score,
      // A floor of `floor` would render the worst run as an invisible sliver, so
      // the scale starts a little below it: the bar compares, it does not measure.
      bar: Math.max(0.06, (family.score - floor + span * 0.08) / (span * 1.08)),
    };
  });
}

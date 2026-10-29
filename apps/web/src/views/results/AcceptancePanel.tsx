import { Panel } from "../../components";
import type { AuditVerdict, RunMetrics } from "../../lib/api";
import { CheckRow, UnrunRow } from "./CheckRow";

export type { AuditVerdict } from "../../lib/api";

const VERDICT_IDS = [
  "outline",
  "outline_drop",
  "nose",
  "jump_shape",
  "jump_pressure",
  "mirror",
  "residuals",
  "residuals_in_time",
  "flux",
  "flux_in_time",
  "heat",
] as const;
type VerdictId = (typeof VERDICT_IDS)[number];

/** What a run's metrics say about each acceptance test: the verdict, or why
 * there is none ("absent" when the run wrote no audit, "unreadable" when it
 * wrote one in a shape this page cannot read). */
export type AuditVerdicts = Record<
  VerdictId,
  AuditVerdict | "absent" | "unreadable"
>;

/** A verdict as written: the worst value against the threshold it must stay
 * below (`max`) or above (`min`); one of the two must be there. */
function readVerdict(entry: unknown): AuditVerdict | "unreadable" {
  if (!entry || typeof entry !== "object") return "unreadable";
  const { worst, max, min, pass } = entry as Record<string, unknown>;
  if (typeof worst !== "number") return "unreadable";
  if (typeof max === "number") return { worst, max, pass: pass === true };
  if (typeof min === "number") return { worst, min, pass: pass === true };
  return "unreadable";
}

/** The acceptance verdicts a run recorded, read defensively: the metrics are
 * a file the evaluate stage wrote, and an older or partial audit must read as
 * missing tests, never as passes. Null for a run that wrote no audit at all. */
export function auditVerdicts(
  metrics: RunMetrics | null | undefined,
): AuditVerdicts | null {
  const audit = metrics?.audit;
  if (!audit || typeof audit !== "object") return null;
  const entries = VERDICT_IDS.map((id) => [
    id,
    id in audit ? readVerdict(audit[id]) : "absent",
  ]);
  return Object.fromEntries(entries) as AuditVerdicts;
}

/** What each verdict measured, in the interface's own voice. */
const ROWS: { id: VerdictId; name: string; detail: string }[] = [
  {
    id: "outline",
    name: "Outline against the masks",
    detail:
      "lowest IoU between the model's bubble and the segmented frame, over every event frame",
  },
  {
    id: "outline_drop",
    name: "Outline held from the warm-up",
    detail:
      "largest drop of a frame's IoU below the fit the warm-up reached before the physics engaged",
  },
  {
    id: "nose",
    name: "Nose position",
    detail:
      "largest distance between the model's nose and the measured nose, in channel widths",
  },
  {
    id: "jump_shape",
    name: "Shape the jump condition asks for",
    detail:
      "Young-Laplace mismatch on the front as the shape change that would remove it (90th percentile, pixels)",
  },
  {
    id: "jump_pressure",
    name: "Pressure the jump condition misses by",
    detail:
      "Young-Laplace mismatch on the front as a pressure (RMS over the front, worst frame, pascals)",
  },
  {
    id: "mirror",
    name: "Mirror symmetry",
    detail:
      "symmetry-breaking fraction of the worst hidden field about the centreline, over every event frame",
  },
  {
    id: "residuals",
    name: "Residuals on the audit grid",
    detail: "RMS of the worst collocation residual, equation by equation",
  },
  {
    id: "residuals_in_time",
    name: "Residuals between the frames",
    detail:
      "the same, scanned at instants across the whole event, camera frames or not",
  },
  {
    id: "flux",
    name: "Cross-section flux balance",
    detail:
      "largest defect of volume flux against the inlet flux plus the vapour created upstream, at any station",
  },
  {
    id: "flux_in_time",
    name: "Flux balance between the frames",
    detail:
      "the largest flux defect at instants across the whole event, camera frames or not",
  },
  {
    id: "heat",
    name: "Heat budget",
    detail:
      "spread between the latent power the growth needs, the source integrates to, and the film and the interface flux carry",
  },
];

/** A verdict's number in one register: three significant figures. */
const fmt = (value: number) => value.toPrecision(3);

/**
 * The acceptance audit: the tests a hidden field must pass before it is
 * trusted, each against the threshold the run's own config set
 * (`evaluation.acceptance.*`). The outline rows come first: the images are
 * the measurement, and a run that fails them is rejected whatever its
 * residuals say. Seed agreement is the one test a single run cannot make; it
 * belongs to a seed sweep.
 */
export function AcceptancePanel({ audit }: { audit: AuditVerdicts | null }) {
  const measured = audit
    ? ROWS.filter((row) => typeof audit[row.id] === "object")
    : [];
  const passed = measured.filter(
    (row) => (audit?.[row.id] as AuditVerdict).pass,
  ).length;
  return (
    <Panel
      title="Acceptance audit"
      subtitle={audit ? `${passed} of ${measured.length} pass` : "not run"}
    >
      {!audit && (
        <p className="state-note">
          This run recorded no acceptance audit. Re-run the evaluate stage with
          the audit on to measure it.
        </p>
      )}
      {audit &&
        ROWS.map((row) => {
          const verdict = audit[row.id];
          if (typeof verdict !== "object") {
            return (
              <UnrunRow
                key={row.id}
                check={{
                  id: row.id,
                  name: row.name,
                  reason:
                    verdict === "absent"
                      ? "not in this run's audit; re-run the evaluate stage"
                      : "recorded in a shape this page cannot read",
                }}
              />
            );
          }
          return (
            <CheckRow
              key={row.id}
              check={{
                id: row.id,
                name: row.name,
                detail: row.detail,
                value: fmt(verdict.worst),
                against:
                  verdict.min != null
                    ? `≥ ${fmt(verdict.min)}`
                    : `≤ ${fmt(verdict.max ?? NaN)}`,
                ok: verdict.pass,
              }}
            />
          );
        })}
      <UnrunRow
        check={{
          id: "seeds",
          name: "Seed agreement",
          reason:
            "compared across the runs of a seed sweep, not inside one run; a field is reported only when the seeds agree",
        }}
      />
    </Panel>
  );
}

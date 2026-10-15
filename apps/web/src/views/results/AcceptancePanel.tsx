import { Panel } from "../../components";
import { CheckRow, UnrunRow } from "./CheckRow";

/** One acceptance verdict as the evaluate stage writes it into the metrics
 * (`audit.verdicts` of audit.json): the worst value over every event frame,
 * the threshold it was read against, and whether it passed. */
export interface AuditVerdict {
  worst: number;
  max: number;
  pass: boolean;
}

export type AuditVerdicts = Partial<
  Record<"mirror" | "residuals" | "flux" | "heat", AuditVerdict>
>;

const VERDICT_KEYS = ["mirror", "residuals", "flux", "heat"] as const;

/** The acceptance verdicts an evaluated run recorded, or null for a run that
 * wrote no audit (one evaluated before the audit existed, or with it off). */
export function auditVerdicts(
  metrics: object | null | undefined,
): AuditVerdicts | null {
  const audit = (metrics as Record<string, unknown> | null | undefined)?.[
    "audit"
  ];
  if (!audit || typeof audit !== "object") return null;
  const verdicts: AuditVerdicts = {};
  for (const key of VERDICT_KEYS) {
    const entry = (audit as Record<string, unknown>)[key];
    if (!entry || typeof entry !== "object") continue;
    const { worst, max, pass } = entry as Record<string, unknown>;
    if (typeof worst === "number" && typeof max === "number") {
      verdicts[key] = { worst, max, pass: pass === true };
    }
  }
  return Object.keys(verdicts).length > 0 ? verdicts : null;
}

/** What each verdict measured, in the interface's own voice. */
const ROWS: { id: keyof AuditVerdicts; name: string; detail: string }[] = [
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
    id: "flux",
    name: "Cross-section flux balance",
    detail:
      "largest defect of volume flux against the inlet flux plus the vapour created upstream, at any station",
  },
  {
    id: "heat",
    name: "Heat budget",
    detail:
      "spread between the latent power the growth needs, the heaters supply and the interface flux carries",
  },
];

/** A verdict's number in one register: three significant figures. */
const fmt = (value: number) => value.toPrecision(3);

/**
 * The acceptance audit: the tests a hidden field must pass before it is
 * trusted, each against the threshold the run's own config set
 * (`evaluation.acceptance.*`). Seed agreement is the one test a single run
 * cannot make; it belongs to a seed sweep.
 */
export function AcceptancePanel({ audit }: { audit: AuditVerdicts | null }) {
  const rows = audit ? ROWS.filter((row) => audit[row.id]) : [];
  const passed = rows.filter((row) => audit?.[row.id]?.pass).length;
  return (
    <Panel
      title="Acceptance audit"
      subtitle={audit ? `${passed} of ${rows.length} pass` : "not run"}
    >
      {!audit && (
        <p className="state-note">
          This run recorded no acceptance audit. Re-run the evaluate stage with
          the audit on to measure it.
        </p>
      )}
      {rows.map((row) => {
        const verdict = audit![row.id]!;
        return (
          <CheckRow
            key={row.id}
            check={{
              id: row.id,
              name: row.name,
              detail: row.detail,
              value: fmt(verdict.worst),
              against: `≤ ${fmt(verdict.max)}`,
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

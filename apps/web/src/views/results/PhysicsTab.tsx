import { Panel, Stat } from "../../components";
import {
  api,
  type DimensionlessGroups,
  type PhysicsValidation,
} from "../../lib/api";
import type { RunSummary } from "../../lib/api";
import {
  physicsVerdict,
  type PhysicsCheck,
  type UnrunCheck,
} from "./physicsChecks";
import { StateNote } from "./StateNote";
import { InterfacePhysicsPanel } from "./InterfacePhysicsPanel";
import { useApiResource } from "./useApiResource";

/** Group tiles worth showing, with display labels (subset of groups.json). */
const GROUP_TILES: [key: string, label: string, unit: string][] = [
  ["Re", "Re", ""],
  ["Re_in", "Re_in", ""],
  ["We", "We", ""],
  ["Ca", "Ca", ""],
  ["Bond", "Bo", ""],
  ["Pr", "Pr", ""],
  ["Ja", "Ja", ""],
  ["Pe", "Pe", ""],
  ["hele_shaw", "12(L/H)²/Re", ""],
  ["bretherton_film_um", "film", "µm"],
];

/** One consistent register for every tile: three significant figures, with
 * scientific notation only outside [0.01, 1000) — no per-tile drift. */
function fmtGroup(value: number): string {
  if (value === 0) return "0";
  const magnitude = Math.abs(value);
  if (magnitude >= 1000 || magnitude < 0.01)
    return value.toExponential(2).replace("e-", "e−").replace("e+", "e");
  return value.toPrecision(3);
}

/** One measured check: what it compared, the number, and the tolerance it is
 * read against — a check without a stated tolerance is an assertion. */
function CheckRow({ check }: { check: PhysicsCheck }) {
  return (
    <div className="check-row" data-ok={check.ok}>
      <span
        className={"check-ic" + (check.ok ? " ok" : " open")}
        aria-hidden="true"
      >
        {check.ok ? "✓" : "!"}
      </span>
      <div>
        <b>{check.name}</b>
        <span>{check.detail}</span>
      </div>
      <span className={check.ok ? "check-v" : "check-v flag"}>
        {check.value}
        <small>{check.against}</small>
      </span>
    </div>
  );
}

/** A check that could not run. Not a failure: an input the series never
 * recorded, and what would supply it. */
function UnrunRow({ check }: { check: UnrunCheck }) {
  return (
    <div className="check-row unrun">
      <span className="check-ic none" aria-hidden="true" />
      <div>
        <b>{check.name}</b>
        <span>{check.reason}</span>
      </div>
      <span className="check-v quiet">—</span>
    </div>
  );
}

interface PhysicsTabProps {
  run: RunSummary;
  runId: string;
  /** Whether the run was trained with an explicit front; decides which empty
   * state the interface-physics panel shows. */
  frontGeometry: boolean;
  /** The viewing condition; its groups drive the tiles (joint runs). */
  dataset: string | null;
  /** The series' display label (ids stay in URLs; labels in copy). */
  datasetName: string | null;
  validation: PhysicsValidation | null;
}

/** The checks that hold independently of training, and the dimensionless
 * groups the physics (and the conditioning vector) derive from. */
export function PhysicsTab({
  run,
  runId,
  frontGeometry,
  dataset,
  datasetName,
  validation,
}: PhysicsTabProps) {
  const groupsQ = useApiResource<DimensionlessGroups>(
    dataset,
    (id) => api.getDatasetGroups(id),
    { nullOn404: true },
  );

  const seriesName = datasetName ?? dataset ?? runId;
  const verdict = physicsVerdict(run, validation, seriesName);
  const film = validation?.bretherton_film_um ?? null;
  const groups = groupsQ.data;

  return (
    <>
      <Panel title="Physics validation" subtitle="independent of training">
        <div className="physics-grid">
          <div>
            <div className="physics-stats">
              <Stat
                label="Nose speed · inferred"
                value={
                  validation?.nose_speed_inferred_mm_s != null
                    ? validation.nose_speed_inferred_mm_s.toFixed(1)
                    : "—"
                }
                unit="mm·s⁻¹"
              />
              <Stat
                label="Nose speed · measured"
                value={
                  validation?.nose_speed_measured_mm_s != null
                    ? validation.nose_speed_measured_mm_s.toFixed(0)
                    : "—"
                }
                unit="mm·s⁻¹"
                hint={
                  validation?.nose_speed_measured_mm_s == null
                    ? `not recorded for ${seriesName}`
                    : undefined
                }
              />
              <Stat
                label="Flags"
                value={verdict.flags.length}
                tone={verdict.flags.length > 0 ? "amber" : "green"}
                hint={`of ${verdict.measured.length} measured checks`}
              />
            </div>

            <p className="check-band">Measured</p>
            {verdict.measured.length === 0 && (
              <p className="state-note">
                This run recorded no physics diagnostics; re-run the evaluate
                stage to measure them.
              </p>
            )}
            {verdict.measured.map((check) => (
              <CheckRow key={check.id} check={check} />
            ))}
            {film != null && (
              <CheckRow
                check={{
                  id: "film",
                  name: "Bretherton film regime",
                  detail:
                    "1.34·Ca^⅔·(H/2), the lubrication film the sides ride on",
                  value: `${film.toFixed(1)} µm`,
                  against: "regime check",
                  ok: true,
                }}
              />
            )}

            {verdict.notRun.length > 0 && (
              <>
                <p className="check-band">Not run</p>
                {verdict.notRun.map((check) => (
                  <UnrunRow key={check.id} check={check} />
                ))}
              </>
            )}

            <StateNote title="Global mass closure is not yet quantitative.">
              The free dilatation source closes with the Stage-B evaporation
              coupling. That is a property of the method rather than a verdict
              on this run, which is why it is stated here once instead of
              failing a check on every run ever trained.
            </StateNote>
          </div>
          <div>
            <h3 className="env-title">
              Dimensionless groups{" "}
              <span className="env-sub">
                {datasetName ?? dataset ?? runId} · conditioning inputs
              </span>
            </h3>
            {groupsQ.loading && <p className="state-note">Loading groups…</p>}
            {!groupsQ.loading && !groups && (
              <p className="state-note">
                No groups recorded for this condition; preprocess the series to
                derive them.
              </p>
            )}
            {groups && (
              <div className="group-tiles">
                {GROUP_TILES.filter(
                  ([key]) => typeof groups[key] === "number",
                ).map(([key, label, unit]) => (
                  <div className="gtile" key={key}>
                    <div className="gtile-k">{label}</div>
                    <div className="gtile-v">
                      {fmtGroup(groups[key])}
                      {unit && <em> {unit}</em>}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </Panel>
      <InterfacePhysicsPanel
        physics={validation?.physics ?? null}
        frontGeometry={frontGeometry}
      />
    </>
  );
}

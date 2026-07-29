import { Panel, Stat } from "../../components";
import {
  api,
  type DimensionlessGroups,
  type PhysicsValidation,
} from "../../lib/api";
import { fmtIou, noseSpeedTone } from "./format";
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

function fmtGroup(value: number): string {
  if (Math.abs(value) >= 1000 || (value !== 0 && Math.abs(value) < 0.01))
    return value.toExponential(1);
  if (Math.abs(value) >= 100) return value.toFixed(0);
  return value.toPrecision(3);
}

interface CheckProps {
  ok: boolean;
  title: string;
  detail: string;
}

function CheckRow({ ok, title, detail }: CheckProps) {
  return (
    <div className="check-row">
      <span className={"check-ic" + (ok ? " ok" : " open")} aria-hidden="true">
        {ok ? "✓" : "!"}
      </span>
      <div>
        <b>{title}</b>
        <span>{detail}</span>
      </div>
    </div>
  );
}

interface PhysicsTabProps {
  runId: string;
  /** The viewing condition; its groups drive the tiles (joint runs). */
  dataset: string | null;
  /** The series' display label (ids stay in URLs; labels in copy). */
  datasetName: string | null;
  validation: PhysicsValidation | null;
}

/** The checks that hold independently of training, and the dimensionless
 * groups the physics (and the conditioning vector) derive from. */
export function PhysicsTab({
  runId,
  dataset,
  datasetName,
  validation,
}: PhysicsTabProps) {
  const groupsQ = useApiResource<DimensionlessGroups>(
    dataset,
    (id) => api.getDatasetGroups(id),
    { nullOn404: true },
  );

  const noseErr = validation?.nose_speed_error_pct ?? null;
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
                    ? "no measured reference for these conditions"
                    : undefined
                }
              />
              <Stat
                label="Error"
                value={noseErr != null ? noseErr.toFixed(1) : "—"}
                unit={noseErr != null ? "%" : undefined}
                tone={noseSpeedTone(noseErr)}
              />
            </div>
            {noseErr != null ? (
              <CheckRow
                ok={noseSpeedTone(noseErr) === "green"}
                title="Nose-speed agreement"
                detail={`inferred vs measured within ${noseErr.toFixed(1)} % — neither quantity was ever given to the model`}
              />
            ) : (
              <CheckRow
                ok={false}
                title="Nose-speed agreement — no reference"
                detail="record a measured nose speed for this condition to close the check"
              />
            )}
            {film != null && (
              <CheckRow
                ok
                title="Bretherton film regime"
                detail={`1.34·Ca^⅔·(H/2) → ${film.toFixed(1)} µm lubrication film`}
              />
            )}
            <CheckRow
              ok={false}
              title="Global mass closure — open"
              detail="the free dilatation source is not yet quantitative; closes with the Stage-B evaporation coupling"
            />
            <CheckRow
              ok={
                validation?.iou_holdout != null ||
                validation?.val_iou_mean != null
              }
              title="Unsupervised agreement exists"
              detail={`holdout/validation IoU ${fmtIou(validation?.iou_holdout ?? validation?.val_iou_mean)} — the generalization evidence`}
            />
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
    </>
  );
}

import { useEffect, useState } from "react";

import { Panel } from "../../components";
import {
  api,
  type ConditionsResponse,
  type ConditionsUpdate,
  type OperatingConditions,
} from "../../lib/api";
import { errorMessage } from "../../lib/errors";

type NumericField = keyof ConditionsUpdate;

interface FieldSpec {
  field: NumericField;
  label: string;
  hint?: string;
  unit: string;
  step: number;
  value: (c: OperatingConditions) => number | null;
}

// Each editable field maps to a real config value on the API side.
const FIELDS: FieldSpec[] = [
  {
    field: "dt_frame_ms",
    label: "Frame interval",
    hint: "Δt",
    unit: "ms",
    step: 0.1,
    value: (c) => c.dt_frame_ms,
  },
  {
    field: "channel_width_um",
    label: "Channel width",
    hint: "imaged",
    unit: "µm",
    step: 10,
    value: (c) => c.channel_width_um,
  },
  {
    field: "channel_height_um",
    label: "Channel height",
    hint: "heated wall below",
    unit: "µm",
    step: 10,
    value: (c) => c.channel_height_um,
  },
  {
    field: "flow_rate_mL_hr",
    label: "Flow rate",
    unit: "mL·hr⁻¹",
    step: 0.5,
    value: (c) => c.flow_rate_mL_hr,
  },
  {
    field: "q_wall_W_cm2",
    label: "Wall heat flux",
    hint: "baseline",
    unit: "W·cm⁻²",
    step: 0.1,
    value: (c) => c.q_wall_W_cm2,
  },
  {
    field: "U_ref",
    label: "Reference velocity",
    hint: "U_ref",
    unit: "m·s⁻¹",
    step: 0.01,
    value: (c) => c.U_ref_m_s,
  },
  {
    field: "T_sat_C",
    label: "Saturation temperature",
    hint: "T_sat",
    unit: "°C",
    step: 0.1,
    value: (c) => c.T_sat_C,
  },
];

interface ConditionsPanelProps {
  datasetId: string;
  conditions: OperatingConditions;
  /** Saved edit round-trip: updated conditions + recomputed groups. */
  onSaved: (response: ConditionsResponse) => void;
}

/** Editable per-series operating conditions; groups recompute on save. */
export function ConditionsPanel({ datasetId, conditions, onSaved }: ConditionsPanelProps) {
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  // A new series selection discards any unsaved drafts of the previous one.
  useEffect(() => {
    setDrafts({});
    setError(null);
  }, [datasetId]);

  async function commit(spec: FieldSpec) {
    const raw = drafts[spec.field];
    if (raw === undefined) return; // untouched
    const parsed = Number(raw);
    if (raw === "" || !Number.isFinite(parsed)) {
      // Not silently dropped: say why the edit didn't save.
      setError(`${spec.label}: enter a number to save this condition.`);
      return;
    }
    if (parsed === spec.value(conditions)) return; // unchanged — nothing to save
    setSaving(true);
    setError(null);
    try {
      onSaved(await api.updateConditions(datasetId, { [spec.field]: parsed }));
      setDrafts((current) => {
        const next = { ...current };
        delete next[spec.field];
        return next;
      });
    } catch (err) {
      setError(`${spec.label}: ${errorMessage(err)}`);
    } finally {
      setSaving(false);
    }
  }

  return (
    <Panel
      title={`Operating conditions — ${datasetId}`}
      subtitle="saved per series · groups recompute live"
    >
      <div className="frm">
        <div className="fld">
          <label htmlFor="cond-fluid">Working fluid</label>
          <div className="ug">
            <select id="cond-fluid" value={conditions.fluid} disabled>
              <option value={conditions.fluid}>
                {conditions.fluid} (sat. {conditions.T_sat_C.toFixed(1)} °C)
              </option>
            </select>
          </div>
        </div>
        {FIELDS.map((spec) => (
          <div className="fld" key={spec.field}>
            <label htmlFor={`cond-${spec.field}`}>
              {spec.label}
              {spec.hint && <span className="hint">{spec.hint}</span>}
            </label>
            <div className="ug">
              <input
                id={`cond-${spec.field}`}
                type="number"
                step={spec.step}
                min={0}
                value={drafts[spec.field] ?? spec.value(conditions) ?? ""}
                disabled={saving}
                onChange={(e) =>
                  setDrafts((current) => ({ ...current, [spec.field]: e.target.value }))
                }
                onBlur={() => void commit(spec)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") (e.target as HTMLInputElement).blur();
                }}
              />
              <span className="sfx">{spec.unit}</span>
            </div>
          </div>
        ))}
      </div>
      {error && (
        <p className="state-note error" role="alert">
          {error}
        </p>
      )}
    </Panel>
  );
}

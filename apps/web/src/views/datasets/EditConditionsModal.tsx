import { useEffect, useMemo, useState } from "react";

import { Button, Callout } from "../../components";
import { type ConditionsUpdate, type DatasetDetail } from "../../lib/api";
import { errorMessage } from "../../lib/errors";
import {
  ALL_CONDITIONS,
  BAKED_KEYS,
  type ConditionInputs,
  type ConditionKey,
  DEFAULT_FLUID_ID,
  FluidSelect,
  MeasurementField,
  OPERATING_CONDITIONS,
  parseCondition,
  REQUIRED_CONDITIONS,
  useFluids,
} from "./conditions";

interface EditConditionsModalProps {
  detail: DatasetDetail;
  onClose: () => void;
  onSave: (updates: ConditionsUpdate) => Promise<void>;
  onSaveLabel: (label: string) => Promise<void>;
}

// Keep in sync with LabelUpdate.label (max_length) in apps/api .../models.py.
const MAX_LABEL_LEN = 80;

/** The current conditions as form strings. U_ref is reported as `U_ref_m_s`. */
function initialInputs(detail: DatasetDetail): ConditionInputs {
  const c = detail.conditions;
  return {
    dt_frame_ms: String(c.dt_frame_ms),
    channel_width_um: String(c.channel_width_um),
    channel_height_um: String(c.channel_height_um),
    q_wall_W_cm2: String(c.q_wall_W_cm2),
    flow_rate_mL_hr: String(c.flow_rate_mL_hr),
    U_ref: c.U_ref_m_s == null ? "" : String(c.U_ref_m_s),
  };
}

/** Edit an existing series' operating conditions. Cheap fields apply on save;
 * editing a tensor-baked field on a processed series warns that a re-preprocess
 * is required (and would mark a trained run stale). T_sat stays derived from the
 * chosen fluid. */
export function EditConditionsModal({
  detail,
  onClose,
  onSave,
  onSaveLabel,
}: EditConditionsModalProps) {
  const fluids = useFluids();
  const [fluidId, setFluidId] = useState<string | null>(null);
  const initial = useMemo(() => initialInputs(detail), [detail]);
  const [conditions, setConditions] = useState<ConditionInputs>(initial);
  // The editable display name; the id (detail.id) is the immutable filesystem key.
  const [label, setLabel] = useState(detail.label ?? "");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  /** A field's saved (original) value, or null when it was unset/out of range. */
  const originalValue = (spec: (typeof ALL_CONDITIONS)[number]) =>
    parseCondition(initial[spec.key], spec.min, spec.max);

  // The detail carries the fluid's display name; map it back to the group id the
  // API expects, once the catalogue has loaded.
  const fluidList = useMemo(
    () => (fluids.status === "ready" ? fluids.list : []),
    [fluids],
  );
  useEffect(() => {
    if (fluidId !== null || fluidList.length === 0) return;
    const match = fluidList.find((f) => f.name === detail.conditions.fluid);
    setFluidId(match?.id ?? fluidList[0]?.id ?? DEFAULT_FLUID_ID);
  }, [fluidList, fluidId, detail.conditions.fluid]);

  const parsed = ALL_CONDITIONS.map((spec) => ({
    spec,
    value: parseCondition(conditions[spec.key], spec.min, spec.max),
  }));
  const conditionsValid = parsed.every((entry) => entry.value !== null);
  const fluidReady = fluidList.length > 0 && fluidId !== null;

  /** A baked field the user has changed on already-processed tensors. */
  function bakedNote(key: ConditionKey): string | undefined {
    if (!detail.processed || !BAKED_KEYS.has(key)) return undefined;
    const spec = ALL_CONDITIONS.find((s) => s.key === key)!;
    const current = parseCondition(conditions[key], spec.min, spec.max);
    if (current === null || current === originalValue(spec)) return undefined;
    return "Baked into the tensors — saving requires a re-preprocess.";
  }

  /** Only the fields the user actually changed, so an edit to one condition
   * never re-sends (and re-bakes) the others. */
  function changedUpdates(): ConditionsUpdate {
    const updates: ConditionsUpdate = {};
    for (const { spec, value } of parsed) {
      if (value !== null && value !== originalValue(spec)) {
        updates[spec.key] = value;
      }
    }
    const originalFluidId = fluidList.find(
      (f) => f.name === detail.conditions.fluid,
    )?.id;
    if (fluidId !== null && fluidId !== originalFluidId)
      updates.fluid = fluidId;
    return updates;
  }

  // Trim + collapse whitespace to match how the API stores the label, so a
  // no-op edit (e.g. trailing spaces) doesn't count as a change.
  const cleanedLabel = label.trim().replace(/\s+/g, " ");
  const labelChanged = cleanedLabel !== (detail.label ?? "");

  // Whether the user has touched any condition field or the fluid (regardless of
  // validity), so a label-only rename isn't blocked by a condition it never
  // edited -- but an in-progress, invalid conditions edit still is.
  const originalFluidId = fluidList.find(
    (f) => f.name === detail.conditions.fluid,
  )?.id;
  const conditionsDirty =
    ALL_CONDITIONS.some((s) => conditions[s.key] !== initial[s.key]) ||
    (fluidId !== null && fluidId !== originalFluidId);
  const conditionsBlocked =
    conditionsDirty && (!conditionsValid || !fluidReady || fluidId === null);

  async function save() {
    if (conditionsBlocked) return;
    const updates = changedUpdates();
    const hasConditionUpdates = Object.keys(updates).length > 0;
    if (!hasConditionUpdates && !labelChanged) {
      onClose(); // nothing changed
      return;
    }
    setSaving(true);
    setError(null);
    try {
      if (labelChanged) await onSaveLabel(cleanedLabel);
      if (hasConditionUpdates) await onSave(updates);
      onClose();
    } catch (err) {
      setError(`Could not save the changes: ${errorMessage(err)}`);
      setSaving(false);
    }
  }

  const field = (key: ConditionKey) => {
    const spec = ALL_CONDITIONS.find((s) => s.key === key)!;
    return (
      <MeasurementField
        key={key}
        spec={spec}
        value={conditions[key]}
        disabled={saving}
        bakedNote={bakedNote(key)}
        onChange={(next) => setConditions((c) => ({ ...c, [key]: next }))}
      />
    );
  };

  return (
    <div
      className="modal-ov"
      role="presentation"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget && !saving) onClose();
      }}
    >
      <div
        className="modal series-modal"
        role="dialog"
        aria-modal="true"
        aria-label={`Edit ${detail.label ?? detail.id} conditions`}
      >
        <div className="hd">
          <h2>Edit {detail.label ?? detail.id} conditions</h2>
          <span className="sub">
            frame interval, width & U_ref re-bake the tensors
          </span>
        </div>
        <div className="body">
          <div className="modal-section">
            <h3 className="modal-section-hd">
              Display name <span>what the UI shows · the id stays fixed</span>
            </h3>
            <div className="ds-label-field">
              <input
                type="text"
                value={label}
                maxLength={MAX_LABEL_LEN}
                placeholder="Display name"
                aria-label="Display name"
                disabled={saving}
                onChange={(e) => setLabel(e.target.value)}
              />
              <span className="mono ds-id" aria-label={`id: ${detail.id}`}>
                {detail.id}
              </span>
            </div>
          </div>

          <div className="modal-section">
            <h3 className="modal-section-hd">
              Measurements{" "}
              <span>frame interval & width bake into the tensors</span>
            </h3>
            <div className="frm">
              {REQUIRED_CONDITIONS.map((s) => field(s.key))}
            </div>
          </div>

          <div className="modal-section">
            <h3 className="modal-section-hd">
              Operating conditions <span>edit to override</span>
            </h3>
            <div className="frm">
              <FluidSelect
                fluids={fluids}
                fluidId={fluidId ?? DEFAULT_FLUID_ID}
                disabled={saving}
                onChange={setFluidId}
              />
              {OPERATING_CONDITIONS.map((s) => field(s.key))}
            </div>
          </div>

          {error && <Callout tone="error">{error}</Callout>}

          <div className="pform-actions">
            <Button
              variant="primary"
              onClick={() => void save()}
              disabled={saving || conditionsBlocked}
            >
              {saving ? "Saving…" : "Save changes"}
            </Button>
            <Button onClick={onClose} disabled={saving}>
              Cancel
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

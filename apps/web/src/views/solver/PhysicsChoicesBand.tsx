import { Band, SelectField } from "../../components";
import { OptionalPhysicsIcon } from "../../components/icons";
import { fieldByName, fieldOptions, optionLabel } from "../../lib/fields";
import type { SolverFormState } from "./form";
import { treatmentOf } from "./formulation";
import { TRUSTED_CAUSAL_GATE, TRUSTED_TIME_BINS } from "./recipes";
import { Caution, SubLabel, SwitchRow, ValueRow } from "./Rows";
import type { SeriesCapability } from "./useSeriesCapability";

interface GroupProps {
  form: SolverFormState;
  onForm: (patch: Partial<SolverFormState>) => void;
  locked: boolean;
}

interface PhysicsChoicesBandProps extends GroupProps {
  /** What the primary series trains, so an unsupported ask is refused here. */
  capability: SeriesCapability;
}

/** The form keys that are a choice among the field table's options. */
type ChoiceKey = {
  [K in keyof SolverFormState]: SolverFormState[K] extends string ? K : never;
}[keyof SolverFormState];

/** The table's label for a form field. */
const labelOf = (name: keyof SolverFormState) => fieldByName(name).label;

/** The table's bounds for a numeric form field. */
function boundsOf(name: keyof SolverFormState): { min: number; max: number } {
  const spec = fieldByName(name);
  return { min: spec.lo ?? -Infinity, max: spec.hi ?? Infinity };
}

/**
 * The trusted-fields choices: where the walls are and what holds at them, how
 * the pressure and the temperature are represented, and the balances the run
 * is held to.
 *
 * Every control renders from the field table (`lib/fields.ts`): its label,
 * its explanation, its options and its bounds are the pipeline's own
 * declaration, so what the rail offers is what the schema accepts. What the
 * band adds is the dependency tree the trainer enforces, so a choice the rung
 * or the series cannot carry is explained here rather than refused at Run.
 */
export function PhysicsChoicesBand({
  form,
  onForm,
  capability,
  locked,
}: PhysicsChoicesBandProps) {
  const summary = [
    form.walls_at === "physical" ? "physical walls" : "crop walls",
    optionLabel(form.momentum_closure),
    form.mass_balance ? "mass balance" : null,
  ]
    .filter(Boolean)
    .join(" · ");
  return (
    <Band
      icon={OptionalPhysicsIcon}
      label="Physics choices"
      hint={summary}
      collapsible
      defaultOpen={false}
    >
      <WallsGroup form={form} onForm={onForm} locked={locked} />
      <RepresentationGroup
        form={form}
        onForm={onForm}
        locked={locked}
        capability={capability}
      />
      <BalancesGroup form={form} onForm={onForm} locked={locked} />
    </Band>
  );
}

/** Where the liquid stops, what holds there, and what closes momentum. */
function WallsGroup({ form, onForm, locked }: GroupProps) {
  const sharp = treatmentOf(form) === "sharp";
  const overDetermined =
    sharp &&
    form.momentum_closure === "darcy" &&
    form.wall_condition === "no_slip";
  return (
    <>
      <SubLabel note="where the liquid stops">Walls and outlet</SubLabel>
      <Choice name="walls_at" form={form} onForm={onForm} locked={locked} />
      <Choice
        name="wall_condition"
        form={form}
        onForm={onForm}
        locked={locked}
      />
      <Choice
        name="momentum_closure"
        form={form}
        onForm={onForm}
        locked={locked}
      />
      {overDetermined && (
        <Caution>
          Darcy drag with no slip is over-determined: a first-order law admits
          only no penetration at a wall. Choose Brinkman, or no penetration.
        </Caution>
      )}
      <Choice
        name="outlet_condition"
        form={form}
        onForm={onForm}
        locked={locked}
      />
    </>
  );
}

/** How the pressure and the temperature are represented: the choices that
 * read the sharp front, and the series' fields. */
function RepresentationGroup({
  form,
  onForm,
  locked,
  capability,
}: PhysicsChoicesBandProps) {
  const treatment = treatmentOf(form);
  const sharp = treatment === "sharp";
  const twoPhaseBlocked =
    !sharp || (capability.known && !capability.hasPressure);
  const sharpThermalBlocked =
    !sharp || (capability.known && !capability.hasTemperature);
  const sharpThermal = form.thermal_model === "sharp";
  return (
    <>
      <SubLabel note="needs the sharp front">Pressure and temperature</SubLabel>
      <Choice
        name="pressure_representation"
        form={form}
        onForm={onForm}
        locked={locked || twoPhaseBlocked}
        reason={
          twoPhaseBlocked
            ? "The two-phase pressure blends the vapour pressure across the Young-Laplace jump, so it needs the sharp front and its pressure field."
            : undefined
        }
      />
      <Choice
        name="thermal_model"
        form={form}
        onForm={(patch) =>
          // The thermal source exists only under the sharp model.
          onForm(
            patch.thermal_model === "sharp"
              ? patch
              : { ...patch, source_mode: "free" },
          )
        }
        locked={locked || sharpThermalBlocked}
        reason={
          sharpThermalBlocked
            ? "The sharp thermal model reads the evaporation flux on the explicit front and needs the temperature field: the sharp treatment, with Energy on Physics & model."
            : undefined
        }
      />
      <Choice
        name="source_mode"
        form={form}
        onForm={(patch) =>
          // The thermal source rebuilds the front once per distinct collocation
          // instant, so the batch must be binned.
          onForm(
            patch.source_mode === "thermal"
              ? {
                  ...patch,
                  collocation_time_bins:
                    form.collocation_time_bins || TRUSTED_TIME_BINS,
                }
              : patch,
          )
        }
        locked={locked || !sharpThermal}
        reason={
          sharpThermal
            ? undefined
            : "The vapour source can be taken from the evaporation flux only under the sharp thermal model."
        }
      />
      <ValueRow
        label={labelOf("collocation_time_bins")}
        hint="distinct instants per batch"
        value={form.collocation_time_bins}
        onChange={(collocation_time_bins) => onForm({ collocation_time_bins })}
        min={
          form.source_mode === "thermal"
            ? 1
            : boundsOf("collocation_time_bins").min
        }
        max={boundsOf("collocation_time_bins").max}
        step={1}
        disabled={locked}
      />
      <SwitchRow
        label={labelOf("mirror_symmetry")}
        hint="every field even about the centreline, v odd"
        verdict={{ tone: "unbenched" }}
        checked={form.mirror_symmetry}
        onChange={(mirror_symmetry) => onForm({ mirror_symmetry })}
        disabled={locked || treatment === "diffuse"}
        reason={
          treatment === "diffuse"
            ? "The mirror axis is the centre of the wall span the front geometry carries; choose an explicit front."
            : undefined
        }
      />
    </>
  );
}

/** The balances the run is held to, and what it records about them. */
function BalancesGroup({ form, onForm, locked }: GroupProps) {
  const gated = form.causal_gate_min_weight != null;
  return (
    <>
      <SubLabel note="what the run is held to">Balances and records</SubLabel>
      <SwitchRow
        label={labelOf("mass_balance")}
        hint="flux through every cross-section · inlet plus vapour created upstream"
        verdict={{ tone: "unbenched" }}
        checked={form.mass_balance}
        onChange={(mass_balance) => onForm({ mass_balance })}
        disabled={locked}
      >
        <ValueRow
          label={labelOf("mass_balance_weight")}
          hint="outside the rebalancer"
          value={form.mass_balance_weight}
          onChange={(mass_balance_weight) => onForm({ mass_balance_weight })}
          min={boundsOf("mass_balance_weight").min}
          max={boundsOf("mass_balance_weight").max}
          step={0.1}
          disabled={locked}
        />
      </SwitchRow>
      <SwitchRow
        label={labelOf("causal_gate_min_weight")}
        hint="eps tightens only once every time chunk clears the gate"
        verdict={{ tone: "positive" }}
        checked={gated}
        onChange={(on) =>
          onForm({ causal_gate_min_weight: on ? TRUSTED_CAUSAL_GATE : null })
        }
        disabled={locked || !form.causal_weighting}
        reason={
          form.causal_weighting
            ? undefined
            : "Gates the causal schedule, so it needs causal weighting (Training band)."
        }
      >
        <ValueRow
          label="Gate threshold"
          hint="minimum chunk weight"
          value={form.causal_gate_min_weight ?? TRUSTED_CAUSAL_GATE}
          onChange={(causal_gate_min_weight) =>
            onForm({ causal_gate_min_weight })
          }
          min={boundsOf("causal_gate_min_weight").min}
          max={boundsOf("causal_gate_min_weight").max}
          step={0.01}
          disabled={locked}
        />
      </SwitchRow>
      <SwitchRow
        label={labelOf("log_chunk_residuals")}
        hint="each equation's residual per time chunk, in the run's history"
        verdict={{ tone: "recipe" }}
        checked={form.log_chunk_residuals}
        onChange={(log_chunk_residuals) => onForm({ log_chunk_residuals })}
        disabled={locked}
      />
    </>
  );
}

interface ChoiceProps<K extends ChoiceKey> {
  name: K;
  form: SolverFormState;
  onForm: (patch: Partial<SolverFormState>) => void;
  locked: boolean;
  /** Why it cannot be changed here, and what would let it. */
  reason?: string;
}

/** One choice among the table's options, labelled and explained by the table.
 * The explanation sits under the control, not in its label, so the control's
 * accessible name stays the label alone. */
function Choice<K extends ChoiceKey>({
  name,
  form,
  onForm,
  locked,
  reason,
}: ChoiceProps<K>) {
  const spec = fieldByName(name);
  return (
    <div className="srow">
      <SelectField
        label={spec.label}
        value={form[name]}
        // The options are the table's, so a chosen value is of the key's type;
        // this is the one place the string widens back to it.
        onChange={(value) =>
          onForm({
            [name]: value as SolverFormState[K],
          } as Partial<SolverFormState>)
        }
        options={fieldOptions(spec)}
        disabled={locked}
      />
      <p className="choice-help">{spec.help}</p>
      {reason && <p className="srow-why">{reason}</p>}
    </div>
  );
}

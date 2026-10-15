import { Band, SelectField } from "../../components";
import {
  HugeiconsIcon,
  OptionalPhysicsIcon,
  WarningIcon,
} from "../../components/icons";
import { fieldByName, fieldOptions, optionLabel } from "../../lib/fields";
import { FORM_BOUNDS, type SolverFormState } from "./form";
import { treatmentOf } from "./formulation";
import { TRUSTED_CAUSAL_GATE, TRUSTED_TIME_BINS } from "./recipes";
import { SubLabel, SwitchRow, ValueRow } from "./Rows";
import type { SeriesCapability } from "./useSeriesCapability";

interface PhysicsChoicesBandProps {
  form: SolverFormState;
  onForm: (patch: Partial<SolverFormState>) => void;
  /** What the primary series trains, so an unsupported ask is refused here. */
  capability: SeriesCapability;
  locked: boolean;
}

type ChoiceKey =
  | "walls_at"
  | "wall_condition"
  | "momentum_closure"
  | "outlet_condition"
  | "pressure_representation"
  | "thermal_model"
  | "source_mode";

/** The table entry behind one choice, with its label, help and options. */
const choice = (key: ChoiceKey) => fieldByName(key);

/**
 * The trusted-fields choices: where the walls are and what holds at them, how
 * the pressure and the temperature are represented, and the balances the
 * run is held to.
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
  const treatment = treatmentOf(form);
  const sharp = treatment === "sharp";
  const noPressure = capability.known && !capability.hasPressure;
  const noTemperature = capability.known && !capability.hasTemperature;
  const twoPhaseBlocked = !sharp || noPressure;
  const sharpThermalBlocked = !sharp || noTemperature;
  const sharpThermal = form.thermal_model === "sharp";
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
      <SubLabel note="where the liquid stops">Walls and outlet</SubLabel>
      <Choice
        spec="walls_at"
        value={form.walls_at}
        onChange={(walls_at) =>
          onForm({ walls_at: walls_at as SolverFormState["walls_at"] })
        }
        disabled={locked}
      />
      <Choice
        spec="wall_condition"
        value={form.wall_condition}
        onChange={(wall_condition) =>
          onForm({
            wall_condition: wall_condition as SolverFormState["wall_condition"],
          })
        }
        disabled={locked}
      />
      <Choice
        spec="momentum_closure"
        value={form.momentum_closure}
        onChange={(momentum_closure) =>
          onForm({
            momentum_closure:
              momentum_closure as SolverFormState["momentum_closure"],
          })
        }
        disabled={locked}
      />
      {form.momentum_closure === "darcy" &&
        form.wall_condition === "no_slip" &&
        sharp && (
          <Caution>
            Darcy drag with no slip is over-determined: a first-order law admits
            only no penetration at a wall. Choose Brinkman, or no penetration.
          </Caution>
        )}
      <Choice
        spec="outlet_condition"
        value={form.outlet_condition}
        onChange={(outlet_condition) =>
          onForm({
            outlet_condition:
              outlet_condition as SolverFormState["outlet_condition"],
          })
        }
        disabled={locked}
      />

      <SubLabel note="needs the sharp front">Pressure and temperature</SubLabel>
      <Choice
        spec="pressure_representation"
        value={form.pressure_representation}
        onChange={(value) =>
          onForm({
            pressure_representation:
              value as SolverFormState["pressure_representation"],
          })
        }
        disabled={locked || twoPhaseBlocked}
        reason={
          twoPhaseBlocked
            ? "The two-phase pressure blends the vapour pressure across the Young-Laplace jump, so it needs the sharp front and its pressure field."
            : undefined
        }
      />
      <Choice
        spec="thermal_model"
        value={form.thermal_model}
        onChange={(value) =>
          onForm(
            value === "sharp"
              ? { thermal_model: "sharp" }
              : { thermal_model: "legacy", source_mode: "free" },
          )
        }
        disabled={locked || sharpThermalBlocked}
        reason={
          sharpThermalBlocked
            ? "The sharp thermal model reads the evaporation flux on the explicit front and needs the temperature field: the sharp treatment, with Energy on Physics & model."
            : undefined
        }
      />
      <Choice
        spec="source_mode"
        value={form.source_mode}
        onChange={(value) =>
          onForm(
            value === "thermal"
              ? {
                  source_mode: "thermal",
                  // The thermal source rebuilds the front once per distinct
                  // collocation instant, so the batch must be binned.
                  collocation_time_bins:
                    form.collocation_time_bins || TRUSTED_TIME_BINS,
                }
              : { source_mode: "free" },
          )
        }
        disabled={locked || !sharpThermal}
        reason={
          sharpThermal
            ? undefined
            : "The vapour source can be taken from the evaporation flux only under the sharp thermal model."
        }
      />
      <ValueRow
        label={choice_label("collocation_time_bins")}
        hint={fieldByName("collocation_time_bins").help}
        value={form.collocation_time_bins}
        onChange={(collocation_time_bins) => onForm({ collocation_time_bins })}
        min={FORM_BOUNDS.collocation_time_bins.min}
        max={FORM_BOUNDS.collocation_time_bins.max}
        step={1}
        disabled={locked}
      />
      <SwitchRow
        label={choice_label("mirror_symmetry")}
        hint={fieldByName("mirror_symmetry").help}
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

      <SubLabel note="what the run is held to">Balances and records</SubLabel>
      <SwitchRow
        label={choice_label("mass_balance")}
        hint={fieldByName("mass_balance").help}
        verdict={{ tone: "unbenched" }}
        checked={form.mass_balance}
        onChange={(mass_balance) => onForm({ mass_balance })}
        disabled={locked}
      >
        <ValueRow
          label={choice_label("mass_balance_weight")}
          hint={fieldByName("mass_balance_weight").help}
          value={form.mass_balance_weight}
          onChange={(mass_balance_weight) => onForm({ mass_balance_weight })}
          min={FORM_BOUNDS.weight.min}
          max={FORM_BOUNDS.weight.max}
          step={0.1}
          disabled={locked}
        />
      </SwitchRow>
      <SwitchRow
        label={choice_label("causal_gate_min_weight")}
        hint={fieldByName("causal_gate_min_weight").help}
        verdict={{ tone: "positive" }}
        checked={form.causal_gate_min_weight != null}
        onChange={(on) =>
          onForm({
            causal_gate_min_weight: on ? TRUSTED_CAUSAL_GATE : null,
          })
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
          hint="every time chunk's weight must clear this before eps tightens"
          value={form.causal_gate_min_weight ?? TRUSTED_CAUSAL_GATE}
          onChange={(causal_gate_min_weight) =>
            onForm({ causal_gate_min_weight })
          }
          min={FORM_BOUNDS.causal_gate_min_weight.min}
          max={FORM_BOUNDS.causal_gate_min_weight.max}
          step={0.01}
          disabled={locked}
        />
      </SwitchRow>
      <SwitchRow
        label={choice_label("log_chunk_residuals")}
        hint={fieldByName("log_chunk_residuals").help}
        verdict={{ tone: "recipe" }}
        checked={form.log_chunk_residuals}
        onChange={(log_chunk_residuals) => onForm({ log_chunk_residuals })}
        disabled={locked}
      />
    </Band>
  );
}

const choice_label = (name: string) => fieldByName(name).label;

interface ChoiceProps {
  spec: ChoiceKey;
  value: string;
  onChange: (value: string) => void;
  disabled: boolean;
  /** Why it cannot be changed here, and what would let it. */
  reason?: string;
}

/** One choice among the table's options, with its help as the hint. */
function Choice({ spec, value, onChange, disabled, reason }: ChoiceProps) {
  const entry = choice(spec);
  return (
    <div className="srow">
      <SelectField
        label={entry.label}
        hint={entry.help}
        value={value}
        onChange={onChange}
        options={fieldOptions(entry)}
        disabled={disabled}
      />
      {reason && <p className="srow-why">{reason}</p>}
    </div>
  );
}

function Caution({ children }: { children: React.ReactNode }) {
  return (
    <p className="rail-caution" role="status">
      <HugeiconsIcon icon={WarningIcon} size={13} aria-hidden="true" />
      <span>{children}</span>
    </p>
  );
}

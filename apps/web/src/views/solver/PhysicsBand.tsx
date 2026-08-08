import { Band } from "../../components";
import {
  HugeiconsIcon,
  InterfaceIcon,
  WarningIcon,
} from "../../components/icons";
import { FORM_BOUNDS, type SolverFormState } from "./form";
import {
  modifierPatch,
  modifiersFor,
  TREATMENTS,
  treatmentMeta,
  treatmentOf,
  treatmentPatch,
} from "./formulation";
import { SubLabel, SwitchRow, ValueRow } from "./Rows";
import type { SeriesCapability } from "./useSeriesCapability";

interface PhysicsBandProps {
  form: SolverFormState;
  onForm: (patch: Partial<SolverFormState>) => void;
  /** What the primary series trains, so an unsupported ask is refused here. */
  capability: SeriesCapability;
  locked: boolean;
}

/** The default target the sharpening switch anneals toward when it is turned on
 * with no target set — the form default, so the switch is never a dead end. */
const DEFAULT_ALPHA_EPS_FINAL = 0.02;
const DEFAULT_ANNEAL_STEPS = 600;

/**
 * The interface formulation, as one choice instead of four coupled switches.
 *
 * The trainer's `front_geometry` / `sharp_interface` / `film_pressure` /
 * `hard_pin` are not independent: they form a ladder, and most of their
 * combinations are rejected by the API. Offering them as a radio group over the
 * three coherent treatments makes every reachable state valid by construction,
 * and lets the rail say the thing that actually changes between them — which
 * equation closes momentum.
 */
export function PhysicsBand({
  form,
  onForm,
  capability,
  locked,
}: PhysicsBandProps) {
  const treatment = treatmentOf(form);
  const meta = treatmentMeta(treatment);
  // Only a KNOWN answer blocks a rung. While the fields are in flight, or when
  // the check itself failed, the choice stays open rather than silently
  // narrowing on a series that may well support it.
  const noPressure = capability.known && !capability.hasPressure;
  const sharpening = form.alpha_eps_anneal_steps > 0;

  return (
    <Band
      icon={InterfaceIcon}
      label="Interface & physics"
      hint={meta.label.toLowerCase()}
      collapsible
    >
      <div
        className="treatments"
        role="radiogroup"
        aria-label="Interface treatment"
      >
        {TREATMENTS.map((option) => {
          const on = option.name === treatment;
          const unavailable = option.needs === "p" && noPressure;
          return (
            <button
              key={option.name}
              type="button"
              role="radio"
              aria-checked={on}
              // aria-disabled, not disabled: the reason below the group is the
              // only explanation there is, and a disabled control leaves the
              // tab order taking its explanation with it.
              aria-disabled={unavailable || locked || undefined}
              className={on ? "treat on" : "treat"}
              onClick={() => {
                if (unavailable || locked) return;
                onForm(treatmentPatch(option.name));
              }}
            >
              <span className="treat-nm">{option.label}</span>
              <span className="treat-note mono">{option.note}</span>
            </button>
          );
        })}
      </div>

      <p className="treat-readout">
        <span className="treat-tag mono">momentum</span>
        {meta.momentum}
      </p>
      <p className="treat-detail">{meta.detail}</p>

      {noPressure && (
        <p className="rail-caution" role="status">
          <HugeiconsIcon icon={WarningIcon} size={13} aria-hidden="true" />
          <span>
            The sharp front reads a pressure field, which{" "}
            <b>{capability.fields ? "this series" : "the series"}</b> does not
            train. Enable Momentum on Physics &amp; model to reach it.
          </span>
        </p>
      )}
      {capability.loadError && (
        <p className="rail-caution" role="status">
          <HugeiconsIcon icon={WarningIcon} size={13} aria-hidden="true" />
          <span>
            Could not read what this series trains ({capability.loadError}), so
            the treatments are all offered. A launch the series cannot support
            will be refused with the reason.
          </span>
        </p>
      )}

      {modifiersFor(treatment).map((modifier) => (
        <SwitchRow
          key={modifier.key}
          label={modifier.label}
          hint={modifier.hint}
          checked={form[modifier.key]}
          onChange={(on) => onForm(modifierPatch(modifier, on))}
          disabled={locked}
        >
          {modifier.key === "hard_pin" && (
            <ValueRow
              label="Pin gate scale"
              hint="d_ref · x* · far field free beyond ~2×"
              value={form.pin_d_ref}
              onChange={(pin_d_ref) => onForm({ pin_d_ref })}
              min={FORM_BOUNDS.pin_d_ref.min}
              max={FORM_BOUNDS.pin_d_ref.max}
              step={0.01}
              disabled={locked}
            />
          )}
        </SwitchRow>
      ))}

      <SubLabel note="α = σ(φ/ε)">Interface sharpening</SubLabel>
      <SwitchRow
        label="Sharpen the interface"
        hint="narrow the blur during training, so a neck is not the same width as it"
        checked={sharpening}
        onChange={(on) =>
          onForm(
            on
              ? {
                  alpha_eps_anneal_steps: DEFAULT_ANNEAL_STEPS,
                  alpha_eps_final:
                    form.alpha_eps_final || DEFAULT_ALPHA_EPS_FINAL,
                }
              : { alpha_eps_anneal_steps: 0 },
          )
        }
        disabled={locked}
      >
        <ValueRow
          label="Anneal over"
          value={form.alpha_eps_anneal_steps}
          onChange={(alpha_eps_anneal_steps) =>
            onForm({ alpha_eps_anneal_steps })
          }
          min={FORM_BOUNDS.alpha_eps_anneal_steps.min}
          max={FORM_BOUNDS.alpha_eps_anneal_steps.max}
          step={100}
          suffix="steps"
          disabled={locked}
        />
        <ValueRow
          label="Final αε"
          hint={`target half-thickness · ${FORM_BOUNDS.alpha_eps_final.min} – ${FORM_BOUNDS.alpha_eps_final.max} y*`}
          value={form.alpha_eps_final}
          onChange={(alpha_eps_final) => onForm({ alpha_eps_final })}
          min={FORM_BOUNDS.alpha_eps_final.min}
          max={FORM_BOUNDS.alpha_eps_final.max}
          step={0.005}
          disabled={locked}
        />
      </SwitchRow>

      <SubLabel note="needs the T field">Evaporation closure</SubLabel>
      <SwitchRow
        label="Depletable superheat"
        hint="the liquid may cool below the inlet, throttling evaporation"
        checked={form.depletable_superheat}
        onChange={(depletable_superheat) => onForm({ depletable_superheat })}
        disabled={locked || (capability.known && !capability.hasTemperature)}
        reason={
          capability.known && !capability.hasTemperature
            ? "This series trains no temperature field; enable Energy on Physics & model."
            : undefined
        }
      />
      <SwitchRow
        label="Two-way evaporation"
        hint="the closure may raise the temperature, not only lower the source"
        checked={form.evap_closure_two_way}
        onChange={(evap_closure_two_way) => onForm({ evap_closure_two_way })}
        disabled={locked}
      />
    </Band>
  );
}

import { Band, TextField } from "../../components";
import { OverridesIcon } from "../../components/icons";
import { parseOverrides, type SolverFormState } from "./form";

interface AdvancedBandProps {
  form: SolverFormState;
  onForm: (patch: Partial<SolverFormState>) => void;
  locked: boolean;
}

/**
 * The escape hatch to the whole schema.
 *
 * The fine dials (rba_gamma, kin_grid, fv_smooth_px, root_delta, ...) are
 * deliberately not form controls -- the rail offers the decision, not every
 * dial -- but before this band they were reachable only from the CLI, so a UI
 * run and a CLI run were different config universes. One mono row accepts the
 * CLI's own vocabulary; the API validates every token against the schema's
 * strict allowlist and answers a 422 naming the offender, and accepted values
 * land in the run's config snapshot like any other override.
 */
export function AdvancedBand({ form, onForm, locked }: AdvancedBandProps) {
  const tokens = parseOverrides(form.overridesText);
  const count = tokens?.length ?? 0;
  return (
    <Band
      icon={OverridesIcon}
      label="Advanced overrides"
      hint={count > 0 ? `${count} dial${count === 1 ? "" : "s"}` : "full schema"}
      collapsible
      defaultOpen={false}
    >
      <div className="adv-overrides">
        <TextField
          label="Overrides"
          hint="key=value · space-separated"
          value={form.overridesText}
          onChange={(overridesText) => onForm({ overridesText })}
          placeholder="training.kin_grid=64"
          invalid={tokens === null}
          disabled={locked}
        />
      </div>
      <p className="rail-note">
        Any top-level <span className="mono">training.*</span> /{" "}
        <span className="mono">model.*</span> schema dial the form does not
        already own. Validated at launch with the reason on refusal; accepted
        values are recorded in the run&apos;s config snapshot.
      </p>
    </Band>
  );
}

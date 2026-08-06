import { useEffect } from "react";

import type { SolverFormState } from "./form";
import { treatmentPatch } from "./formulation";
import type { SeriesCapability } from "./useSeriesCapability";

/**
 * Keep the form on a configuration the primary series can actually run.
 *
 * The form's defaults are the recommended Stage-B recipe — the sharp front and
 * a depletable superheat — and the launcher refuses an EXPLICIT ask for either
 * on a series that trains no pressure or no temperature. Without this, a
 * Stage-A series met a form it could never launch, and only found out at Run.
 *
 * The correction is stated, never silent: the treatment falls back to the
 * explicit front (which needs no Stage-B field and keeps the capsule
 * construction), and the band it happened in says why and where to change it.
 */
export function useCapabilityGuard(
  form: SolverFormState,
  onForm: (patch: Partial<SolverFormState>) => void,
  capability: SeriesCapability,
): void {
  const { known, hasPressure, hasTemperature } = capability;
  const sharp = form.sharp_interface;
  const depletable = form.depletable_superheat;

  useEffect(() => {
    if (!known) return;
    const patch: Partial<SolverFormState> = {};
    if (sharp && !hasPressure) Object.assign(patch, treatmentPatch("front"));
    if (depletable && !hasTemperature) patch.depletable_superheat = false;
    if (Object.keys(patch).length > 0) onForm(patch);
  }, [known, hasPressure, hasTemperature, sharp, depletable, onForm]);
}

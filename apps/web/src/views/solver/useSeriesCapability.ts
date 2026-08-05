import { useEffect, useState } from "react";

import { api } from "../../lib/api";
import { errorMessage } from "../../lib/errors";

/**
 * What the primary series can actually support, read from the fields it trains.
 *
 * The sharp-interface family reads fields a Stage-A series does not have, and
 * the launcher rejects an EXPLICIT ask it cannot honour (422) rather than
 * downgrading it silently. The form's own default is the Stage-B recipe, so
 * without this the Solver would offer a configuration that the API is certain
 * to refuse, and say so only after the user pressed Run.
 *
 * The launcher resolves the family against the PRIMARY series (the first of a
 * joint run), so that is what this reads — the same series, the same answer.
 */
export interface SeriesCapability {
  /** null while loading, or when there is no series to read. */
  fields: string[] | null;
  /** Momentum is enabled, so there is a pressure field to read at the front. */
  hasPressure: boolean;
  /** Energy is enabled, so there is a temperature field to deplete. */
  hasTemperature: boolean;
  /** Whether the answer is known yet; false while the fields are in flight. */
  known: boolean;
  loadError: string | null;
}

const UNKNOWN: SeriesCapability = {
  fields: null,
  hasPressure: false,
  hasTemperature: false,
  known: false,
  loadError: null,
};

export function useSeriesCapability(primary: string | null): SeriesCapability {
  const [state, setState] = useState<SeriesCapability>(UNKNOWN);

  useEffect(() => {
    if (!primary) {
      setState(UNKNOWN);
      return;
    }
    let alive = true;
    setState(UNKNOWN);
    api
      .getPhysics(primary)
      .then((physics) => {
        if (!alive) return;
        setState({
          fields: physics.fields,
          hasPressure: physics.fields.includes("p"),
          hasTemperature: physics.fields.includes("T"),
          known: true,
          loadError: null,
        });
      })
      // Unknown is not "unsupported": leave the options as they are and say why
      // the check could not run, rather than quietly locking the form down.
      .catch(
        (error) =>
          alive && setState({ ...UNKNOWN, loadError: errorMessage(error) }),
      );
    return () => {
      alive = false;
    };
  }, [primary]);

  return state;
}

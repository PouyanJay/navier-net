import { useEffect, useState } from "react";

import { Callout } from "../../components";
import { ReconstructionViewport } from "../../components/ReconstructionViewport";
import { api, ApiError, type InterfaceData } from "../../lib/api";
import { errorMessage } from "../../lib/errors";

type Load =
  | { status: "loading" }
  | { status: "unavailable" }
  | { status: "error"; message: string }
  | { status: "ready"; data: InterfaceData };

/** The Results hero: the continuous reconstruction, served as real contours. */
export function ReconstructionPanel({ runId }: { runId: string }) {
  const [load, setLoad] = useState<Load>({ status: "loading" });

  useEffect(() => {
    let alive = true;
    setLoad({ status: "loading" });
    api
      .getInterface(runId)
      .then((data) => alive && setLoad({ status: "ready", data }))
      .catch((err) => {
        if (!alive) return;
        // A 404 means "nothing trained to reconstruct"; an empty state, not a failure.
        setLoad(
          err instanceof ApiError && err.status === 404
            ? { status: "unavailable" }
            : { status: "error", message: errorMessage(err) },
        );
      });
    return () => {
      alive = false;
    };
  }, [runId]);

  if (load.status === "loading") {
    return (
      <p className="state-note" role="status">
        Reconstructing the interface…
      </p>
    );
  }
  if (load.status === "unavailable") {
    return (
      <p className="state-note">
        No trained model to reconstruct from; train this run in the Solver
        first.
      </p>
    );
  }
  if (load.status === "error") {
    return (
      <Callout tone="error" title="Could not load the reconstruction">
        {load.message}
      </Callout>
    );
  }

  const { data } = load;
  const cameraInstants = data.measured.length;
  const ratio =
    cameraInstants > 1
      ? Math.round(data.frames.length / (cameraInstants - 1))
      : null;
  return (
    <div>
      <ReconstructionViewport data={data} />
      <p className="figcap">
        <b>Figure 1.</b> Continuous PINN interface reconstruction;{" "}
        {data.frames.length} instants
        {ratio != null && ratio > 1 && <> · ~{ratio}× finer than the camera</>}.
        Dashed contours are the measured interface at camera instants.
      </p>
    </div>
  );
}

/** SSE consumer for a run's live event stream (`/api/runs/{id}/stream`).
 *
 * The server replays the run's whole event buffer on connect and closes the
 * stream once the run is over, so a subscriber — including one that joins late
 * or re-attaches after navigating away — always reconstructs the full console
 * and loss history. The stream is closed client-side on a terminal status so
 * EventSource's automatic reconnect cannot replay events twice.
 */

import type { ConsoleLine, LossRecord, RunJobStatus } from "./api";

export interface RunStreamHandlers {
  onHist?: (record: LossRecord) => void;
  onLog?: (line: ConsoleLine) => void;
  onStatus?: (status: RunJobStatus) => void;
  /** The stream dropped before a terminal status (network, server restart). */
  onInterrupted?: () => void;
}

/** Subscribe to a run's events. Returns a cleanup that closes the stream. */
export function openRunStream(runId: string, handlers: RunStreamHandlers): () => void {
  const source = new EventSource(`/api/runs/${encodeURIComponent(runId)}/stream`);
  let finished = false;

  source.addEventListener("hist", (event) => {
    handlers.onHist?.(JSON.parse((event as MessageEvent).data) as LossRecord);
  });
  source.addEventListener("log", (event) => {
    handlers.onLog?.(JSON.parse((event as MessageEvent).data) as ConsoleLine);
  });
  source.addEventListener("status", (event) => {
    const status = JSON.parse((event as MessageEvent).data) as RunJobStatus;
    if (status.state !== "running") {
      finished = true;
      source.close();
    }
    handlers.onStatus?.(status);
  });
  source.onerror = () => {
    if (finished) return;
    source.close();
    handlers.onInterrupted?.();
  };

  return () => source.close();
}

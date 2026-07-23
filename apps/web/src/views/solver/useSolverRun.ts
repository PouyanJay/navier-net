import { useCallback, useEffect, useRef, useState } from "react";

import {
  api,
  type ConsoleLine,
  type LossRecord,
  type RunJobStatus,
  type RunLaunchRequest,
} from "../../lib/api";
import { errorMessage } from "../../lib/errors";
import { openRunStream } from "../../lib/runStream";

const IDLE_LINES: ConsoleLine[] = [
  { line: "[naviernet] solver idle — configure a run and press Run", tone: "dim" },
];

export interface SolverRun {
  status: RunJobStatus | null;
  running: boolean;
  hist: LossRecord[];
  lines: ConsoleLine[];
  holdoutIou: number | null;
  error: string | null;
  start: (request: RunLaunchRequest) => Promise<void>;
  reset: () => void;
}

/**
 * Owns one training run's lifecycle: launch, live SSE consumption, terminal
 * follow-up (holdout IoU), and re-attachment to a run that is still going when
 * the view remounts (the server replays the full event buffer on connect, so
 * re-attaching reconstructs the console and loss history).
 */
export function useSolverRun(
  onStatusChange?: (status: RunJobStatus | null) => void,
  onFinished?: () => void,
): SolverRun {
  const [status, setStatus] = useState<RunJobStatus | null>(null);
  const [hist, setHist] = useState<LossRecord[]>([]);
  const [lines, setLines] = useState<ConsoleLine[]>(IDLE_LINES);
  const [holdoutIou, setHoldoutIou] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const closeStream = useRef<(() => void) | null>(null);

  const handleStatus = useCallback(
    (next: RunJobStatus) => {
      setStatus(next);
      onStatusChange?.(next);
      if (next.state === "done") {
        api
          .getRun(next.run_id)
          .then((detail) => setHoldoutIou(detail.metrics?.iou_holdout ?? null))
          .catch(() => setHoldoutIou(null)); // metrics stay "—" if unreadable
      }
      if (next.state !== "running") onFinished?.();
    },
    [onStatusChange, onFinished],
  );

  const attach = useCallback(
    (runId: string) => {
      closeStream.current?.();
      closeStream.current = openRunStream(runId, {
        onHist: (record) => setHist((prev) => [...prev, record]),
        onLog: (line) => setLines((prev) => [...prev, line]),
        onStatus: handleStatus,
        onInterrupted: () => {
          setLines((prev) => [
            ...prev,
            { line: "[naviernet] live stream interrupted — status may lag", tone: "err" },
          ]);
          // Best-effort refresh; the interrupted banner above already told the user.
          api.getRunStatus(runId).then(handleStatus).catch(() => {});
        },
      });
    },
    [handleStatus],
  );

  // Re-attach to a run that is still training when the view (re)mounts.
  useEffect(() => {
    api
      .getActiveRun()
      .then((active) => {
        if (active?.state === "running") {
          setStatus(active);
          setHist([]);
          setLines([]);
          attach(active.run_id);
        }
      })
      .catch(() => {}); // no server-side run to re-attach to
    return () => closeStream.current?.();
  }, [attach]);

  const start = useCallback(
    async (request: RunLaunchRequest) => {
      setError(null);
      setHist([]);
      setLines([]);
      setHoldoutIou(null);
      try {
        const launched = await api.startRun(request);
        setStatus(launched);
        onStatusChange?.(launched);
        attach(launched.run_id);
      } catch (err) {
        setError(errorMessage(err));
        setLines(IDLE_LINES);
      }
    },
    [attach, onStatusChange],
  );

  const reset = useCallback(() => {
    if (status?.state === "running") return; // never discard a live console
    closeStream.current?.();
    setStatus(null);
    setHist([]);
    setLines(IDLE_LINES);
    setHoldoutIou(null);
    setError(null);
    onStatusChange?.(null);
  }, [status?.state, onStatusChange]);

  return {
    status,
    running: status?.state === "running",
    hist,
    lines,
    holdoutIou,
    error,
    start,
    reset,
  };
}

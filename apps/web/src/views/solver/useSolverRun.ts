import { useCallback, useEffect, useRef, useState } from "react";

import {
  api,
  type ConsoleLine,
  type LossRecord,
  type RunJobStatus,
  type RunLaunchRequest,
  type SweepLaunchRequest,
  type SweepStatus,
} from "../../lib/api";
import { errorMessage } from "../../lib/errors";
import { openRunStream } from "../../lib/runStream";

const IDLE_LINES: ConsoleLine[] = [
  { line: "[naviernet] solver idle — configure a run and press Run", tone: "dim" },
];

const SWEEP_POLL_MS = 1000;

export interface SolverRun {
  status: RunJobStatus | null;
  sweep: SweepStatus | null;
  running: boolean;
  hist: LossRecord[];
  lines: ConsoleLine[];
  holdoutIou: number | null;
  error: string | null;
  start: (request: RunLaunchRequest) => Promise<void>;
  startSweep: (request: SweepLaunchRequest) => Promise<void>;
  reset: () => void;
}

/**
 * Owns one training run's — or one sweep's — lifecycle: launch, live SSE
 * consumption, terminal follow-up (holdout IoU), and re-attachment to work
 * still going when the view remounts (the server replays the full event
 * buffer on connect, so re-attaching reconstructs console and loss history).
 *
 * During a sweep the monitor follows the child currently training: the sweep
 * status is polled, and the stream re-attaches (with a console divider) each
 * time the running child changes. The loss chart is per child.
 */
export function useSolverRun(
  onStatusChange?: (status: RunJobStatus | null) => void,
  onFinished?: () => void,
): SolverRun {
  const [status, setStatus] = useState<RunJobStatus | null>(null);
  const [sweep, setSweep] = useState<SweepStatus | null>(null);
  const [hist, setHist] = useState<LossRecord[]>([]);
  const [lines, setLines] = useState<ConsoleLine[]>(IDLE_LINES);
  const [holdoutIou, setHoldoutIou] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const closeStream = useRef<(() => void) | null>(null);
  const attachedChild = useRef<string | null>(null);

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
      attachedChild.current = runId;
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

  /** Point the monitor at the sweep child now training (fresh chart per child). */
  const followSweepChild = useCallback(
    (next: SweepStatus) => {
      const current = next.children.find((child) => child.state === "running");
      if (current && current.run_id !== attachedChild.current) {
        setHist([]);
        setLines((prev) => [
          ...prev,
          { line: `[sweep] ── child ${current.run_id} ──`, tone: "em" },
        ]);
        attach(current.run_id);
      }
    },
    [attach],
  );

  // Poll a running sweep; the stream only covers the current child.
  useEffect(() => {
    if (sweep?.state !== "running") return;
    const id = window.setInterval(async () => {
      try {
        const next = await api.getSweep(sweep.sweep_id);
        setSweep(next);
        followSweepChild(next);
        if (next.state !== "running") {
          window.clearInterval(id);
          if (next.state === "error" && next.message) setError(next.message);
        }
      } catch (err) {
        window.clearInterval(id);
        setError(errorMessage(err));
      }
    }, SWEEP_POLL_MS);
    return () => window.clearInterval(id);
  }, [sweep?.state, sweep?.sweep_id, followSweepChild]);

  // Re-attach to a run or sweep that is still going when the view (re)mounts.
  useEffect(() => {
    api
      .getActiveSweep()
      .then((activeSweep) => {
        if (activeSweep?.state === "running") {
          setSweep(activeSweep);
          setHist([]);
          setLines([]);
          followSweepChild(activeSweep);
          return null;
        }
        return api.getActiveRun();
      })
      .then((active) => {
        if (active?.state === "running") {
          setStatus(active);
          setHist([]);
          setLines([]);
          attach(active.run_id);
        }
      })
      .catch(() => {}); // nothing server-side to re-attach to
    return () => closeStream.current?.();
  }, [attach, followSweepChild]);

  const clearSession = useCallback(() => {
    setError(null);
    setHist([]);
    setLines([]);
    setHoldoutIou(null);
    setSweep(null);
    attachedChild.current = null;
  }, []);

  const start = useCallback(
    async (request: RunLaunchRequest) => {
      clearSession();
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
    [attach, clearSession, onStatusChange],
  );

  const startSweep = useCallback(
    async (request: SweepLaunchRequest) => {
      clearSession();
      try {
        const launched = await api.startSweep(request);
        setSweep(launched);
        const first = launched.children[0] ?? null;
        setStatus(first);
        if (first) onStatusChange?.(first);
        setLines([
          {
            line: `[sweep] ${launched.sweep_id} — seeds ${launched.seeds.join(", ")}`,
            tone: "dim",
          },
        ]);
        followSweepChild(launched);
      } catch (err) {
        setError(errorMessage(err));
        setLines(IDLE_LINES);
      }
    },
    [clearSession, followSweepChild, onStatusChange],
  );

  const sweepRunning = sweep?.state === "running";
  const runRunning = status?.state === "running" || status?.state === "queued";

  const reset = useCallback(() => {
    if (sweepRunning || status?.state === "running") return; // never discard a live console
    closeStream.current?.();
    setStatus(null);
    clearSession();
    setLines(IDLE_LINES);
    onStatusChange?.(null);
  }, [sweepRunning, status?.state, clearSession, onStatusChange]);

  return {
    status,
    sweep,
    running: sweepRunning || runRunning,
    hist,
    lines,
    holdoutIou,
    error,
    start,
    startSweep,
    reset,
  };
}

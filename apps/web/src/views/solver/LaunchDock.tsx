import { Button } from "../../components";

interface LaunchDockProps {
  /** The run this configuration amounts to, in a few words. */
  summary: string[];
  /** Why Run is unavailable, or null when it can be pressed. */
  blocked: string | null;
  running: boolean;
  onReset: () => void;
  onRun: () => void;
}

/**
 * The launch bar: what is about to run, and the two buttons that do it.
 *
 * It carries the summary because the configuration above it is four folded
 * bands long — the last thing before Run should state what Run will do. And a
 * disabled Run says why it is disabled, rather than leaving the user to hunt
 * the rail for the field that is empty.
 */
export function LaunchDock({
  summary,
  blocked,
  running,
  onReset,
  onRun,
}: LaunchDockProps) {
  return (
    <div className="aside-actions solver-dock">
      <p className="dock-summary mono">{summary.join(" · ")}</p>
      {blocked && (
        <p className="dock-blocked" role="status">
          {blocked}
        </p>
      )}
      <div className="dock-buttons">
        <Button onClick={onReset} disabled={running}>
          Reset
        </Button>
        <Button
          variant="primary"
          onClick={onRun}
          disabled={running || blocked !== null}
        >
          Run
        </Button>
      </div>
    </div>
  );
}

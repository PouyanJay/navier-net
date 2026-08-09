import { Button } from "../../components";

interface LaunchDockProps {
  /** The run this configuration amounts to, in a few words. */
  summary: string[];
  /** The recipe baseline and each departure from it, as chips. Null hides the
   * row entirely (a resume takes its config from the checkpoint, so there is
   * no baseline to depart from). */
  recipe: { label: string; diff: string[] } | null;
  /** Why Run is unavailable, or null when it can be pressed. */
  blocked: string | null;
  running: boolean;
  onReset: () => void;
  onRun: () => void;
}

/**
 * The launch bar: what is about to run, and the two buttons that do it.
 *
 * It carries the summary because the configuration above it is five folded
 * bands long — the last thing before Run should state what Run will do. The
 * chips are the reproducibility statement: the chosen recipe, then everything
 * this run changes about it. And a disabled Run says why it is disabled,
 * rather than leaving the user to hunt the rail for the field that is empty.
 */
export function LaunchDock({
  summary,
  recipe,
  blocked,
  running,
  onReset,
  onRun,
}: LaunchDockProps) {
  return (
    <div className="aside-actions solver-dock">
      <p className="dock-summary mono">{summary.join(" · ")}</p>
      {recipe && (
        <ul className="dock-chips mono" aria-label="Departures from the recipe">
          <li className="dock-chip base">{recipe.label}</li>
          {recipe.diff.map((chip) => (
            <li key={chip} className="dock-chip">
              {chip}
            </li>
          ))}
        </ul>
      )}
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

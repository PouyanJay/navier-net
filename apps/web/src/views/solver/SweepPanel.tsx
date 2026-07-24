import { Chip, Panel, StatusDot } from "../../components";
import type { SweepStatus } from "../../lib/api";

const CHILD_TONE = {
  queued: "default",
  running: "accent",
  done: "green",
  error: "red",
} as const;

const seedOf = (runId: string) => runId.split("-s").pop() ?? runId;

/** Live sweep overview: one chip per child, in execution order. */
export function SweepPanel({ sweep }: { sweep: SweepStatus }) {
  const tone =
    sweep.state === "running" ? "accent" : sweep.state === "done" ? "green" : "red";
  return (
    <Panel title="Seed sweep" subtitle={sweep.sweep_id}>
      <div className="sweep-row">
        <StatusDot tone={tone} label={`sweep · ${sweep.state}`} />
        <div className="sweep-chips">
          {sweep.children.map((child) => (
            <Chip key={child.run_id} tone={CHILD_TONE[child.state]}>
              seed {seedOf(child.run_id)} · {child.state}
            </Chip>
          ))}
        </div>
      </div>
      {sweep.state === "done" && (
        <p className="state-note">
          Sweep complete — select its children under Results &amp; validation to compare
          seeds side by side.
        </p>
      )}
    </Panel>
  );
}

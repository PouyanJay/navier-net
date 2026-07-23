import { Panel } from "../components";

/** Stand-in for views arriving in later phases, so the shell nav is complete. */
export function Placeholder({ title }: { title: string }) {
  return (
    <Panel title={title} subtitle="Coming in a later phase">
      <p className="state-note">
        This surface is part of the platform roadmap and will be built in an upcoming
        phase.
      </p>
    </Panel>
  );
}

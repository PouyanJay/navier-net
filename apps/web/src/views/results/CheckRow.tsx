import type { PhysicsCheck, UnrunCheck } from "./physicsChecks";

/** One measured check: what it compared, the number, and the tolerance it is
 * read against — a check without a stated tolerance is an assertion. */
export function CheckRow({ check }: { check: PhysicsCheck }) {
  return (
    <div className="check-row" data-ok={check.ok}>
      <span
        className={"check-ic" + (check.ok ? " ok" : " open")}
        aria-hidden="true"
      >
        {check.ok ? "✓" : "!"}
      </span>
      <div>
        <b>{check.name}</b>
        <span>{check.detail}</span>
      </div>
      <span className={check.ok ? "check-v" : "check-v flag"}>
        {check.value}
        <small>{check.against}</small>
      </span>
    </div>
  );
}

/** A check that could not run. Not a failure: an input the series never
 * recorded, and what would supply it. */
export function UnrunRow({ check }: { check: UnrunCheck }) {
  return (
    <div className="check-row unrun">
      <span className="check-ic none" aria-hidden="true" />
      <div>
        <b>{check.name}</b>
        <span>{check.reason}</span>
      </div>
      <span className="check-v quiet">—</span>
    </div>
  );
}

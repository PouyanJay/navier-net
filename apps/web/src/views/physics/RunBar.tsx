import { useState } from "react";

import { Button, InfoPopover } from "../../components";
import { FIELDS, fmtCount } from "./model";
import type { PhysicsModel } from "./usePhysicsModel";

/** What the current configuration trains, plus the exact reproducible command. */
export function RunBar({ model }: { model: PhysicsModel }) {
  const [copied, setCopied] = useState(false);
  const base = model.perField.phi;
  const facts = model.activeFields.map((f) => FIELDS[f].label).join(" · ");

  const copy = () => {
    void navigator.clipboard?.writeText(model.hydraCommand).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <section className="card runbar" aria-label="Resulting training run">
      <div className="body">
        <div>
          <span className="runlead">This configuration trains</span>
          <div className="runfacts">
            {facts} <span className="dim">·</span> {base.width}×{base.depth}{" "}
            <span className="dim">·</span> Fourier {model.globals.ff}{" "}
            <span className="dim">·</span> {fmtCount(model.totalParams)} params
            {model.overrideCount > 0 && (
              <span className="dim">
                {" "}
                · {model.overrideCount} override
                {model.overrideCount === 1 ? "" : "s"}
              </span>
            )}
          </div>
        </div>
        <div className="runacts">
          <Button onClick={copy}>
            {copied ? "Copied" : "Copy train command"}
          </Button>
          <InfoPopover label="The exact command" width={560}>
            <div className="cmdpop">
              <div className="cmdlead">exact command · reproducible</div>
              <code>{model.hydraCommand}</code>
            </div>
          </InfoPopover>
        </div>
      </div>
    </section>
  );
}

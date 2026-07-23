import { EquationBlock, Panel } from "../../components";
import "./physics.css";

/** The Stage-A governing physics, rendered as math with explanatory prose. */
export function GoverningEquations() {
  return (
    <Panel title="Governing equations" subtitle="Stage A — VOF transport + continuity">
      <p className="prose">
        The network is constrained to satisfy the conservation laws, not merely to
        interpolate the pixels. The volume fraction is advected by the local velocity,
        and continuity carries an inferred dilatation source for phase change.
      </p>

      <p className="prose">Volume-of-fluid transport (the interface moves with the flow):</p>
      <EquationBlock tex="r_{\text{vof}} = \alpha_t + u\,\alpha_x + v\,\alpha_y = 0" />

      <p className="prose">Continuity with an inferred dilatation source <em>s</em>:</p>
      <EquationBlock tex="r_{\text{div}} = u_x + v_y - s = 0" />

      <p className="prose">
        The volume fraction is a bounded function of a level-set field, making the
        interface half-thickness <em>ε</em> an explicit, annealable parameter:
      </p>
      <EquationBlock tex="\alpha = \sigma\!\left(\phi / \varepsilon\right)" />

      <p className="prose">
        Boundary conditions: plug inflow at the inlet and no-slip side walls.
      </p>
      <EquationBlock tex="u\big|_{\text{inlet}} = u_{\text{in}}, \qquad \mathbf{u}\big|_{\text{wall}} = \mathbf{0}" />
    </Panel>
  );
}

import { useEffect, useState } from "react";

import { type FieldArch, type FieldName, FIELDS, fmtCount } from "./model";
import type { PhysicsModel } from "./usePhysicsModel";

/* Which residual / supervision nodes train each field. The sharp trio is here
   alongside momentum: a launch takes one treatment or the other, and the
   diagram draws whichever is actually admitted, so it never disagrees with
   the objective above it. */
const LINKS: Record<FieldName, string[]> = {
  phi: ["data", "vof", "div", "mom", "darcy", "laplace"],
  u: ["vof", "div", "bc", "mom", "darcy", "kinematic"],
  v: ["vof", "div", "bc", "mom", "darcy", "kinematic"],
  s: ["div", "energy"],
  p: ["mom", "darcy", "laplace"],
  T: ["energy"],
};

const RES_LABEL: Record<string, string> = {
  data: "L_data · α frames",
  vof: "r_vof",
  div: "r_div + src",
  bc: "L_bc · inlet + walls",
  mom: "r_mom",
  darcy: "r_darcy",
  kinematic: "r_kin · v_n = u·n",
  laplace: "r_YL · curvature jump",
  energy: "r_energy",
};

/* Hub order, matching the registry so the diagram and the objective list the
   same physics in the same sequence. */
const RES_ORDER = [
  "data",
  "vof",
  "div",
  "bc",
  "mom",
  "darcy",
  "kinematic",
  "laplace",
  "energy",
];

const X = { in: 58, ff: 205, net: 385, hub: 950 };
const LANE_H = 62;
const W = 1160;

function bez(x0: number, y0: number, x1: number, y1: number): string {
  const mx = (x0 + x1) / 2;
  return `M ${x0} ${y0} C ${mx} ${y0}, ${mx} ${y1}, ${x1} ${y1}`;
}

function layerBars(
  arch: FieldArch,
  ff: number,
): { label: string; p: number }[] {
  const { width: w, depth: d } = arch;
  const bars = [{ label: "γ→h₁", p: 2 * ff * w + w }];
  for (let i = 1; i < d; i++)
    bars.push({ label: `h${i}→h${i + 1}`, p: w * w + w });
  bars.push({ label: "head", p: w + 1 });
  return bars;
}

function ColumnCaptions() {
  return (
    <g
      fontFamily="var(--mono)"
      fontSize="9.5"
      fill="var(--viewtxt)"
      textAnchor="middle"
      letterSpacing="1"
    >
      <text x={X.in} y={22}>
        INPUT
      </text>
      <text x={X.ff} y={22}>
        ENCODING
      </text>
      <text x={(X.net + X.hub - 60) / 2} y={22}>
        FIELD NETWORKS · width × depth
      </text>
      <text x={X.hub + 50} y={22}>
        RESIDUALS &amp; SUPERVISION
      </text>
    </g>
  );
}

function InputHub({ cy }: { cy: number }) {
  return (
    <>
      {["x*", "y*", "t*"].map((n, i) => {
        const y = cy + (i - 1) * 40;
        return (
          <g key={n}>
            <circle
              cx={X.in}
              cy={y}
              r={13}
              fill="var(--view2)"
              stroke="var(--viewline)"
            />
            <text
              x={X.in}
              y={y + 3.5}
              textAnchor="middle"
              fontFamily="var(--mono)"
              fontSize="10"
              fill="var(--viewtxt)"
            >
              {n}
            </text>
            <path
              d={bez(X.in + 14, y, X.ff - 52, cy)}
              fill="none"
              stroke="var(--viewline)"
              strokeWidth={1.2}
            />
          </g>
        );
      })}
    </>
  );
}

function FourierBlock({
  cy,
  ff,
  ffScale,
}: {
  cy: number;
  ff: number;
  ffScale: number;
}) {
  return (
    <>
      <rect
        x={X.ff - 52}
        y={cy - 34}
        width={104}
        height={68}
        rx={8}
        fill="var(--view2)"
        stroke="var(--canvas-accent)"
      />
      <text
        x={X.ff}
        y={cy - 6}
        textAnchor="middle"
        fontFamily="var(--mono)"
        fontSize="10.5"
        fill="var(--canvas-accent)"
      >
        γ(x,y,t)
      </text>
      <text
        x={X.ff}
        y={cy + 10}
        textAnchor="middle"
        fontFamily="var(--mono)"
        fontSize="9.5"
        fill="var(--viewtxt)"
      >
        {ff} pairs · σ_B {ffScale}
      </text>
    </>
  );
}

function ResidualHub({
  nodes,
  resY,
}: {
  nodes: string[];
  resY: (i: number) => number;
}) {
  return (
    <>
      {nodes.map((id, i) => {
        const y = resY(i);
        return (
          <g key={id}>
            <rect
              x={X.hub - 10}
              y={y - 12}
              width={150}
              height={24}
              rx={6}
              fill="var(--view2)"
              stroke="var(--viewline)"
            />
            <text
              x={X.hub}
              y={y + 3.5}
              fontFamily="var(--mono)"
              fontSize="9.5"
              fill="var(--viewtxt)"
            >
              {RES_LABEL[id]}
            </text>
          </g>
        );
      })}
    </>
  );
}

interface LaneProps {
  field: FieldName;
  arch: FieldArch;
  y: number;
  cy: number;
  selected: boolean;
  params: number;
  resNodes: string[];
  resY: (i: number) => number;
  onSelect: () => void;
}

function FieldLane({
  field,
  arch,
  y,
  cy,
  selected,
  params,
  resNodes,
  resY,
  onSelect,
}: LaneProps) {
  const meta = FIELDS[field];
  const bh = Math.max(12, Math.round(Math.sqrt(arch.width) * 3.4));
  const bw = 13;
  const gap = 4;
  const netW = arch.depth * (bw + gap);
  return (
    <g
      className={selected ? "lane sel" : "lane"}
      tabIndex={0}
      role="button"
      aria-label={`${meta.label} network, ${arch.width} wide by ${arch.depth} deep, ${fmtCount(params)} parameters`}
      onClick={onSelect}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onSelect();
        }
      }}
    >
      <rect
        className="lanehit"
        x={X.net - 46}
        y={y - LANE_H / 2 + 4}
        width={X.hub - X.net + 26}
        height={LANE_H - 8}
        rx={8}
        fill={selected ? "var(--view2)" : "transparent"}
        stroke={selected ? "var(--canvas-accent)" : "transparent"}
      />
      <path
        className="ribbon"
        d={bez(X.ff + 52, cy, X.net - 4, y)}
        fill="none"
        stroke={meta.hue}
        strokeWidth={2}
        opacity={0.45}
      />
      {Array.from({ length: arch.depth }, (_, l) => (
        <rect
          key={l}
          x={X.net + l * (bw + gap)}
          y={y - bh / 2}
          width={bw}
          height={bh}
          rx={2.5}
          fill={meta.hue}
          opacity={selected ? 0.95 : 0.7}
        />
      ))}
      <text
        x={X.net + netW + 14}
        y={y - 4}
        fontFamily="var(--mono)"
        fontSize="10.5"
        fontWeight={600}
        fill={meta.hue}
      >
        {meta.label} · {arch.width}×{arch.depth}
      </text>
      <text
        x={X.net + netW + 14}
        y={y + 10}
        fontFamily="var(--mono)"
        fontSize="8.5"
        fill="var(--viewtxt)"
      >
        {meta.transform} · {fmtCount(params)}
      </text>
      {LINKS[field].map((rid) => {
        const ri = resNodes.indexOf(rid);
        if (ri < 0) return null;
        return (
          <path
            key={rid}
            className="ribbon"
            d={bez(X.net + netW + 110, y, X.hub - 12, resY(ri))}
            fill="none"
            stroke={meta.hue}
            strokeWidth={1.4}
            opacity={selected ? 0.85 : 0.22}
          />
        );
      })}
    </g>
  );
}

function Inspector({
  field,
  arch,
  activation,
  ff,
  params,
}: {
  field: FieldName;
  arch: FieldArch;
  activation: boolean;
  ff: number;
  params: number;
}) {
  const meta = FIELDS[field];
  const bars = layerBars(arch, ff);
  const maxP = Math.max(...bars.map((b) => b.p));
  return (
    <div className="inspector" aria-live="polite">
      <span className="fname" style={{ color: meta.hue }}>
        {meta.label}
      </span>
      <span>
        {arch.width} × {arch.depth} · {activation ? "adaptive-tanh" : "tanh"}
      </span>
      <span className="sep">|</span>
      <span>{meta.transform}</span>
      <span className="sep">|</span>
      <span>{fmtCount(params)} params</span>
      <span
        role="img"
        aria-label={`Per-layer parameter distribution for ${meta.label}`}
        style={{
          display: "inline-flex",
          alignItems: "flex-end",
          gap: "3px",
          height: "26px",
        }}
      >
        {bars.map((b, i) => (
          <span
            key={i}
            title={`${b.label} · ${fmtCount(b.p)}`}
            style={{
              display: "block",
              width: "9px",
              height: `${Math.max(3, (b.p / maxP) * 26)}px`,
              borderRadius: "2px 2px 0 0",
              background: "var(--canvas-accent)",
              opacity: 0.75,
            }}
          />
        ))}
      </span>
    </div>
  );
}

function Legend({ active }: { active: FieldName[] }) {
  return (
    <div className="vlegend">
      {active.map((f) => (
        <span key={f}>
          <span className="sw2" style={{ background: FIELDS[f].hue }} />
          {FIELDS[f].label}
        </span>
      ))}
      <span style={{ marginLeft: "auto" }}>
        block height ∝ √width · blocks = layers · click a lane
      </span>
    </div>
  );
}

/** The live network ensemble: shared input → Fourier → per-field lanes →
 * residual/supervision hub. Click a lane to inspect it. */
interface Reco {
  value: string;
  why: string;
  modified: boolean;
}

function recommendations(model: PhysicsModel): Reco[] {
  const g = model.globals;
  const pf = model.perField;
  const pOn = model.fieldOn("p") || model.fieldOn("T");
  return [
    {
      value: `σ_B = ${g.ffScale} · ε = ${g.alphaEps}`,
      why: `Spectral scale sized so ${g.ff} Fourier pairs resolve the interface half-width.`,
      modified:
        model.globalOverridden("ff") || model.globalOverridden("ffScale"),
    },
    {
      value: `α network ${pf.phi.width} × ${pf.phi.depth}`,
      why: "The steepest field: it carries the sigmoid(φ/ε) interface.",
      modified: model.fieldOverridden("phi"),
    },
    {
      value: `u, v, s networks ${pf.u.width} × ${pf.u.depth}`,
      why: "Smooth hidden fields; capacity follows α at ratio 1.0.",
      modified:
        model.fieldOverridden("u") ||
        model.fieldOverridden("v") ||
        model.fieldOverridden("s"),
    },
    {
      value: pOn
        ? `p, T networks ${pf.p.width} × ${pf.p.depth}`
        : "p, T · locked",
      why: pOn
        ? "Stage-B fields sized +33% width, +2 depth for stiffer residuals."
        : "Enable Momentum or Energy to unlock pressure and temperature.",
      modified:
        pOn && (model.fieldOverridden("p") || model.fieldOverridden("T")),
    },
  ];
}

/**
 * The shapes the drawing above is made of, in words, each with the reasoning
 * that produced it.
 *
 * These used to sit a panel away from the ensemble they describe; they are
 * captions for it, so they live with it.
 */
function DerivedShapes({ model }: { model: PhysicsModel }) {
  return (
    <div className="derived">
      <p className="reco-note">
        <b>Derived from your physics.</b> Parameter count is exact; time and
        memory are estimates. Hover a row for the reasoning; every shape here is
        set in the Capacity band, in the rail.
      </p>
      {recommendations(model).map((reco, i) => (
        <div key={i} className={reco.modified ? "rrow mod" : "rrow"}>
          <span className="rv">{reco.value}</span>
          <span className="hasinfo" tabIndex={0}>
            <span className="infob" aria-hidden="true">
              i
            </span>
            <div className="infopop rwhy" role="tooltip">
              {reco.why}
            </div>
          </span>
        </div>
      ))}
    </div>
  );
}

export function EnsembleCanvas({ model }: { model: PhysicsModel }) {
  const active = model.activeFields;
  const [selected, setSelected] = useState<FieldName>("phi");
  useEffect(() => {
    if (!active.includes(selected)) setSelected(active[0] ?? "phi");
  }, [active, selected]);

  const enabledEq = new Set(
    model.equations.filter((e) => e.on).map((e) => e.id),
  );
  const resNodes = RES_ORDER.filter(
    (id) => id === "data" || id === "bc" || enabledEq.has(id),
  );

  const H = Math.max(
    320,
    46 + Math.max(active.length * LANE_H, resNodes.length * 46) + 40,
  );
  const cy = H / 2 + 8;
  const laneY = (i: number) => cy + (i - (active.length - 1) / 2) * LANE_H;
  const resY = (i: number) => cy + (i - (resNodes.length - 1) / 2) * 46;
  const inspected = active.includes(selected) ? selected : (active[0] ?? "phi");

  return (
    <section className="viewport" aria-labelledby="ens-h">
      <div className="vhd">
        <span className="t" id="ens-h">
          Network ensemble · live
        </span>
        <span className="hudright">
          {active.length} networks · {fmtCount(model.totalParams)} params
        </span>
      </div>

      <div className="canvaswrap">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          role="img"
          aria-label={`Network ensemble: a shared space-time input through Fourier features into ${active.length} field networks, coupled through ${resNodes.length} residual terms.`}
        >
          <ColumnCaptions />
          <InputHub cy={cy} />
          <FourierBlock
            cy={cy}
            ff={model.globals.ff}
            ffScale={model.globals.ffScale}
          />
          <ResidualHub nodes={resNodes} resY={resY} />
          {active.map((f, i) => (
            <FieldLane
              key={f}
              field={f}
              arch={model.perField[f]}
              y={laneY(i)}
              cy={cy}
              selected={selected === f}
              params={model.fieldParamCount(f)}
              resNodes={resNodes}
              resY={resY}
              onSelect={() => setSelected(f)}
            />
          ))}
        </svg>
      </div>

      <Inspector
        field={inspected}
        arch={model.perField[inspected]}
        activation={model.globals.nodewise}
        ff={model.globals.ff}
        params={model.fieldParamCount(inspected)}
      />
      <Legend active={active} />
      <DerivedShapes model={model} />
    </section>
  );
}

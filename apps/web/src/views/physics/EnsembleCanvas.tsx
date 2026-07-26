import { useEffect, useState } from "react";

import { type FieldArch, type FieldName, FIELDS, fmtCount } from "./model";
import type { PhysicsModel } from "./usePhysicsModel";

// Which residual / supervision nodes train each field.
const LINKS: Record<FieldName, string[]> = {
  phi: ["data", "vof", "div", "mom"],
  u: ["vof", "div", "bc", "mom"],
  v: ["vof", "div", "bc", "mom"],
  s: ["div", "energy"],
  p: ["mom"],
  T: ["energy"],
};

const RES_LABEL: Record<string, string> = {
  data: "L_data · α frames",
  vof: "r_vof",
  div: "r_div + src",
  bc: "L_bc · inlet + walls",
  mom: "r_mom",
  energy: "r_energy",
};

const X = { in: 58, ff: 205, net: 385, hub: 950 };
const LANE_H = 62;
const W = 1160;

function bez(x0: number, y0: number, x1: number, y1: number): string {
  const mx = (x0 + x1) / 2;
  return `M ${x0} ${y0} C ${mx} ${y0}, ${mx} ${y1}, ${x1} ${y1}`;
}

function fieldParamBars(arch: FieldArch, ff: number): { label: string; p: number }[] {
  const { width: w, depth: d } = arch;
  const bars = [{ label: "γ→h₁", p: 2 * ff * w + w }];
  for (let i = 1; i < d; i++) bars.push({ label: `h${i}→h${i + 1}`, p: w * w + w });
  bars.push({ label: "head", p: w + 1 });
  return bars;
}

/** The live network ensemble: shared input → Fourier → per-field lanes →
 * residual/supervision hub. Click a lane to inspect it. */
export function EnsembleCanvas({ model }: { model: PhysicsModel }) {
  const active = model.activeFields;
  const [selected, setSelected] = useState<FieldName>("phi");
  useEffect(() => {
    if (!active.includes(selected)) setSelected(active[0] ?? "phi");
  }, [active, selected]);

  const enabledEq = new Set(model.equations.filter((e) => e.on).map((e) => e.id));
  const resNodes = ["data", "vof", "div", "bc", "mom", "energy"].filter(
    (id) => id === "data" || id === "bc" || enabledEq.has(id),
  );

  const H = Math.max(320, 46 + Math.max(active.length * LANE_H, resNodes.length * 46) + 40);
  const cy = H / 2 + 8;
  const laneY = (i: number) => cy + (i - (active.length - 1) / 2) * LANE_H;
  const resY = (i: number) => cy + (i - (resNodes.length - 1) / 2) * 46;

  const sel = FIELDS[selected];
  const selArch = model.perField[selected];
  const bars = fieldParamBars(selArch, model.globals.ff);
  const maxP = Math.max(...bars.map((b) => b.p));

  return (
    <section className="viewport" aria-labelledby="ens-h">
      <div className="vhd">
        <span className="t" id="ens-h">
          Network ensemble — live
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

          {/* input hub */}
          {["x*", "y*", "t*"].map((n, i) => {
            const y = cy + (i - 1) * 40;
            return (
              <g key={n}>
                <circle cx={X.in} cy={y} r={13} fill="var(--view2)" stroke="var(--viewline)" />
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

          {/* Fourier block */}
          <rect
            x={X.ff - 52}
            y={cy - 34}
            width={104}
            height={68}
            rx={8}
            fill="var(--view2)"
            stroke="var(--acc2)"
          />
          <text x={X.ff} y={cy - 6} textAnchor="middle" fontFamily="var(--mono)" fontSize="10.5" fill="var(--acc2)">
            γ(x,y,t)
          </text>
          <text x={X.ff} y={cy + 10} textAnchor="middle" fontFamily="var(--mono)" fontSize="9.5" fill="var(--viewtxt)">
            {model.globals.ff} pairs · σ_B {model.globals.ffScale}
          </text>

          {/* residual hub */}
          {resNodes.map((id, i) => {
            const y = resY(i);
            return (
              <g key={id}>
                <rect x={X.hub - 10} y={y - 12} width={150} height={24} rx={6} fill="var(--view2)" stroke="var(--viewline)" />
                <text x={X.hub} y={y + 3.5} fontFamily="var(--mono)" fontSize="9.5" fill="var(--viewtxt)">
                  {RES_LABEL[id]}
                </text>
              </g>
            );
          })}

          {/* field lanes */}
          {active.map((f, i) => {
            const meta = FIELDS[f];
            const arch = model.perField[f];
            const y = laneY(i);
            const isSel = selected === f;
            const bh = Math.max(12, Math.round(Math.sqrt(arch.width) * 3.4));
            const bw = 13;
            const gap = 4;
            const netW = arch.depth * (bw + gap);
            const pick = () => setSelected(f);
            return (
              <g
                key={f}
                className={isSel ? "lane sel" : "lane"}
                tabIndex={0}
                role="button"
                aria-label={`${meta.label} network, ${arch.width} wide by ${arch.depth} deep, ${fmtCount(model.fieldParamCount(f))} parameters`}
                onClick={pick}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    pick();
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
                  fill={isSel ? "var(--view2)" : "transparent"}
                  stroke={isSel ? "var(--acc2)" : "transparent"}
                />
                <path
                  className="ribbon"
                  d={bez(X.ff + 52, cy, X.net - 4, y)}
                  fill="none"
                  stroke={meta.hue}
                  strokeWidth={2}
                  opacity={0.45}
                />
                {Array.from({ length: arch.depth }, (_, L) => (
                  <rect
                    key={L}
                    x={X.net + L * (bw + gap)}
                    y={y - bh / 2}
                    width={bw}
                    height={bh}
                    rx={2.5}
                    fill={meta.hue}
                    opacity={isSel ? 0.95 : 0.7}
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
                <text x={X.net + netW + 14} y={y + 10} fontFamily="var(--mono)" fontSize="8.5" fill="var(--viewtxt)">
                  {meta.transform} · {fmtCount(model.fieldParamCount(f))}
                </text>
                {LINKS[f].map((rid) => {
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
                      opacity={isSel ? 0.85 : 0.22}
                    />
                  );
                })}
              </g>
            );
          })}
        </svg>
      </div>

      <div className="inspector" aria-live="polite">
        <span className="fname" style={{ color: sel.hue }}>
          {sel.label}
        </span>
        <span>
          {selArch.width} × {selArch.depth}
        </span>
        <span className="sep">|</span>
        <span>{sel.transform}</span>
        <span className="sep">|</span>
        <span>{fmtCount(model.fieldParamCount(selected))} params</span>
        <span className="sep">|</span>
        <span
          role="img"
          aria-label={`Per-layer parameter distribution for ${sel.label}`}
          style={{ display: "inline-flex", alignItems: "flex-end", gap: "3px", height: "26px" }}
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
                background: "var(--acc2)",
                opacity: 0.75,
              }}
            />
          ))}
        </span>
      </div>

      <div className="vlegend">
        {active.map((f) => (
          <span key={f}>
            <span className="sw2" style={{ background: FIELDS[f].hue }} />
            {FIELDS[f].label}
          </span>
        ))}
        <span style={{ marginLeft: "auto" }}>block height ∝ √width · blocks = layers · click a lane</span>
      </div>
    </section>
  );
}

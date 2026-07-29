import { useEffect, useRef, useState } from "react";

import { Callout, Panel } from "../../components";
import { api, ApiError, type FieldMap } from "../../lib/api";
import { errorMessage } from "../../lib/errors";
import { cmap, cssGradient, type ColormapName } from "./fieldColormaps";

interface FieldDef {
  k: "alpha" | "u" | "v" | "umag" | "s" | "p" | "T";
  chip: string;
  label: string;
  /** The governing statement this field answers to. */
  eq: string;
  cm: ColormapName;
  /** Signed field → symmetric range around zero with the diverging map. */
  signed?: boolean;
}

const FIELD_DEFS: FieldDef[] = [
  {
    k: "alpha",
    chip: "α",
    label: "volume fraction α",
    eq: "∂α/∂t + u·∇α = 0 — VOF interface transport",
    cm: "blue",
  },
  {
    k: "umag",
    chip: "|u|",
    label: "speed |u|",
    eq: "|u| = √(u² + v²) — speed of the reconstructed flow",
    cm: "blue",
  },
  {
    k: "u",
    chip: "u",
    label: "streamwise velocity u",
    eq: "∂u/∂x + ∂v/∂y = s — continuity constrains the reconstructed flow",
    cm: "div",
    signed: true,
  },
  {
    k: "v",
    chip: "v",
    label: "transverse velocity v",
    eq: "∂u/∂x + ∂v/∂y = s — continuity constrains the reconstructed flow",
    cm: "div",
    signed: true,
  },
  {
    k: "s",
    chip: "s",
    label: "dilatation source s",
    eq: "s = (ρℓ/ρᵥ − 1)·St·j·δ_int — evaporation mass closure",
    cm: "amber",
  },
  {
    k: "p",
    chip: "p",
    label: "pressure p*",
    eq: "Du/Dt = −∇p + Re⁻¹∇²u − c_HS·u + We⁻¹·κ·n·δ_int — momentum · Stage B",
    cm: "blue",
  },
  {
    k: "T",
    chip: "T",
    label: "superheat θ",
    eq: "DT/Dt = Pe⁻¹∇²T + q̇_wall − St·j·δ_int — energy · Stage B",
    cm: "thermal",
  },
];

/** The |residual| maps: where each governing equation is least satisfied. */
const RESIDUAL_DEFS = [
  { k: "res_vof", label: "vof · interface transport" },
  { k: "res_div", label: "div · continuity" },
  { k: "res_mom", label: "momentum · Stage B" },
  { k: "res_energy", label: "energy · Stage B" },
] as const;

/** How long the time scrubber waits before asking the model to re-evaluate. */
const SCRUB_DEBOUNCE_MS = 180;
/** Playback: one evaluation step per tick (server caches by rounded t). */
const PLAY_STEP = 1 / 24;
const PLAY_TICK_MS = 400;

/** Paint a field grid into a canvas with the given colormap + range. */
function paintGrid(
  canvas: HTMLCanvasElement,
  map: FieldMap,
  cm: ColormapName,
  lo: number,
  hi: number,
) {
  const context = canvas.getContext("2d");
  if (!context) return; // jsdom
  const width = map.x_um.length;
  const height = map.y_um.length;
  canvas.width = width;
  canvas.height = height;
  const image = context.createImageData(width, height);
  const span = hi - lo || 1;
  map.values.forEach((row, y) =>
    row.forEach((value, x) => {
      const [r, g, b] = cmap(cm, (value - lo) / span);
      const offset = (y * width + x) * 4;
      image.data[offset] = r;
      image.data[offset + 1] = g;
      image.data[offset + 2] = b;
      image.data[offset + 3] = 255;
    }),
  );
  context.putImageData(image, 0, 0);
}

/** Range the colormap spans: symmetric around zero for signed fields. */
function colorRange(def: FieldDef, map: FieldMap): [number, number] {
  if (!def.signed) return [map.vmin, map.vmax];
  const limit = Math.max(Math.abs(map.vmin), Math.abs(map.vmax));
  return [-limit, limit];
}

interface ResidualMapProps {
  runId: string;
  dataset: string | undefined;
  name: string;
  label: string;
  /** t* to evaluate at (the tab's settled scrubber time). */
  t: number;
  available: boolean | null;
}

/** One equation's |residual| on the grid at the tab's current instant. */
function ResidualMap({
  runId,
  dataset,
  name,
  label,
  t,
  available,
}: ResidualMapProps) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [map, setMap] = useState<FieldMap | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (available === false) return;
    let alive = true;
    api
      .getField(runId, name, t, dataset)
      .then((payload) => {
        if (alive) {
          setMap(payload);
          setFailed(false);
        }
      })
      .catch(() => {
        if (alive) setFailed(true);
      });
    return () => {
      alive = false;
    };
  }, [runId, dataset, name, t, available]);

  useEffect(() => {
    if (canvas.current && map)
      paintGrid(canvas.current, map, "amber", map.vmin, map.vmax);
  }, [map]);

  if (available === false)
    return (
      <div className="res-map res-map-off">
        <span>{label}</span>
        <p>Stage B off — enable momentum &amp; energy and retrain.</p>
      </div>
    );

  return (
    <div className="res-map">
      <canvas
        ref={canvas}
        role="img"
        aria-label={`${label} residual magnitude map`}
      />
      <div className="res-map-cap">
        <span>{label}</span>
        <span>
          {failed
            ? "unavailable"
            : map
              ? `max ${map.vmax.toExponential(1)}`
              : "…"}
        </span>
      </div>
    </div>
  );
}

interface FieldsTabProps {
  runId: string;
  /** The viewing condition for joint runs; null on single-dataset runs. */
  dataset: string | null;
  /** The series' display label (ids stay in URLs; labels in copy). */
  datasetName: string | null;
  joint: boolean;
}

/**
 * The fields the camera never measured, evaluated live from the run's own
 * checkpoint — plus the |residual| of each governing equation, the map of
 * where the physics is least satisfied.
 */
export function FieldsTab({
  runId,
  dataset,
  datasetName,
  joint,
}: FieldsTabProps) {
  const [field, setField] = useState<FieldDef>(FIELD_DEFS[1]);
  const [tRatio, setTRatio] = useState(0.34);
  const [playing, setPlaying] = useState(false);
  const [map, setMap] = useState<FieldMap | null>(null);
  const [available, setAvailable] = useState<string[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [unavailable, setUnavailable] = useState(false);
  const canvas = useRef<HTMLCanvasElement>(null);
  const debounce = useRef<ReturnType<typeof setTimeout>>(undefined);

  const scopedDataset = joint ? (dataset ?? undefined) : undefined;

  useEffect(() => {
    let alive = true;
    clearTimeout(debounce.current);
    debounce.current = setTimeout(() => {
      const span = map ? map.t_max_star - map.t_min_star : 1;
      const t = (map?.t_min_star ?? 0) + tRatio * span;
      api
        .getField(runId, field.k, t, scopedDataset)
        .then((payload) => {
          if (!alive) return;
          setMap(payload);
          setAvailable(payload.fields_available);
          setError(null);
          setUnavailable(false);
        })
        .catch((exc: unknown) => {
          if (!alive) return;
          if (exc instanceof ApiError && exc.status === 404) {
            setUnavailable(true);
            setError(null);
          } else {
            setError(errorMessage(exc));
          }
        });
    }, SCRUB_DEBOUNCE_MS);
    return () => {
      alive = false;
    };
    // `map` is deliberately not a dependency: it only supplies the time span
    // for the ratio→t* mapping, and refetching on every payload would loop.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId, scopedDataset, field, tRatio]);

  // Playback: step the scrubber; the fetch effect above follows each step
  // (the server caches per rounded t, so a second loop replays instantly).
  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(
      () => setTRatio((ratio) => (ratio + PLAY_STEP) % 1),
      PLAY_TICK_MS,
    );
    return () => clearInterval(timer);
  }, [playing]);

  // Paint the grid whenever a payload lands.
  useEffect(() => {
    if (!canvas.current || !map) return;
    const [lo, hi] = colorRange(field, map);
    paintGrid(canvas.current, map, field.cm, lo, hi);
  }, [map, field]);

  const stageBOff = available !== null && !available.includes("p");
  const [lo, hi] = map ? colorRange(field, map) : [0, 1];
  const settledT = map
    ? map.t_min_star + tRatio * (map.t_max_star - map.t_min_star)
    : 0;
  const tReadout =
    map != null
      ? `t* ${map.t_star.toFixed(2)} of ${map.t_max_star.toFixed(2)}`
      : "…";

  return (
    <>
      <Panel
        title="Predicted fields"
        subtitle={`${datasetName ?? dataset ?? "run"} · evaluated live from ckpt.pt`}
      >
        <div
          className="fieldchips"
          role="group"
          aria-label="Field"
          data-testid="field-chips"
        >
          {FIELD_DEFS.map((def) => {
            const disabled = available !== null && !available.includes(def.k);
            return (
              <button
                key={def.k}
                type="button"
                className="fchip"
                aria-pressed={def.k === field.k}
                disabled={disabled}
                title={
                  disabled
                    ? `${def.label} — needs a Stage-B checkpoint`
                    : def.label
                }
                onClick={() => setField(def)}
              >
                {def.chip}
              </button>
            );
          })}
        </div>
        {stageBOff && (
          <p className="note">
            <b>p and θ unavailable on this run.</b> It trained Stage A only —
            enable momentum &amp; energy in Physics &amp; model and retrain to
            infer pressure and temperature.
          </p>
        )}
        <p className="eqline">{field.eq}</p>

        {error && (
          <Callout tone="error" title="Could not evaluate the field">
            {error}
          </Callout>
        )}
        {unavailable && (
          <p className="state-note">
            No trained model to evaluate; train this run in the Solver first.
          </p>
        )}
        {!error && !unavailable && (
          <div className="field-view">
            <canvas
              ref={canvas}
              className="field-canvas"
              role="img"
              aria-label={`${field.label} field map`}
            />
            <div className="cbar">
              <span>{lo.toFixed(field.k === "alpha" ? 1 : 0)}</span>
              <span
                className="cbar-g"
                style={{ background: cssGradient(field.cm) }}
                aria-hidden="true"
              />
              <span>{hi.toFixed(field.k === "alpha" ? 1 : 0)}</span>
              <span className="cbar-unit">{map?.unit ?? ""}</span>
            </div>
            <div className="field-ctl">
              <button
                type="button"
                className="field-play"
                aria-label={
                  playing ? "Pause field animation" : "Play field animation"
                }
                aria-pressed={playing}
                onClick={() => setPlaying((state) => !state)}
              >
                {playing ? "⏸" : "▶"}
              </button>
              <input
                type="range"
                min={0}
                max={1000}
                value={Math.round(tRatio * 1000)}
                onChange={(event) => {
                  setPlaying(false);
                  setTRatio(Number(event.target.value) / 1000);
                }}
                aria-label="Field time"
              />
              <span className="field-t">{tReadout}</span>
            </div>
          </div>
        )}
        <p className="figcap">
          <b>Figure 4.</b> Fields the camera never measured, inferred by the
          governing equations from the interface motion. Play or scrub time to
          watch the flow develop; every value is a live evaluation of this run's
          checkpoint.
        </p>
      </Panel>

      <Panel
        title="Equation residuals"
        subtitle="|residual| on the grid · where the physics is least satisfied"
      >
        <div className="res-grid">
          {RESIDUAL_DEFS.map((def) => (
            <ResidualMap
              key={def.k}
              runId={runId}
              dataset={scopedDataset}
              name={def.k}
              label={def.label}
              t={settledT}
              available={available === null ? null : available.includes(def.k)}
            />
          ))}
        </div>
        <p className="figcap">
          <b>Figure 5.</b> Residuals concentrate in a thin band at the interface
          — the hardest region for the physics to satisfy. A residual that grows
          or spreads flags an under-resolved region, not a converged one.
          Evaluated by differentiating the checkpoint on the grid at the instant
          above.
        </p>
      </Panel>
    </>
  );
}

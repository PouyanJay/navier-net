import { DL, type KV, Panel } from "../../components";
import type { OperatingConditions } from "../../lib/api";

function rows(c: OperatingConditions): KV[] {
  return [
    { label: "Fluid", value: c.fluid },
    { label: "Saturation temperature", value: c.T_sat_C.toFixed(1), hint: "°C" },
    { label: "Wall heat flux", value: c.q_wall_W_cm2.toFixed(1), hint: "W/cm²" },
    { label: "Flow rate", value: c.flow_rate_mL_hr.toFixed(1), hint: "mL/hr" },
    {
      label: "Channel",
      value: `${c.channel_width_um}×${c.channel_height_um}`,
      hint: "µm",
    },
    { label: "Frame interval", value: c.dt_frame_ms.toFixed(2), hint: "ms" },
    { label: "Flow direction", value: c.flow_direction },
    { label: "Frames (raw / usable / event)", value: `${c.n_frames_raw} / ${c.n_frames_usable} / ${c.n_frames_event}` },
  ];
}

/** The dataset's operating conditions, read from the composed config. */
export function ConditionsPanel({ conditions }: { conditions: OperatingConditions }) {
  return (
    <Panel title="Operating conditions" subtitle="From the experiment config">
      <DL items={rows(conditions)} />
    </Panel>
  );
}

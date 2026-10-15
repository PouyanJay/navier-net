# Plan: trustworthy hidden fields

Status: draft for review, 2026-10-15. Nothing in this plan has been implemented yet.
Companion page in plain language: https://claude.ai/artifact/EGxkpeq5ByLLESkUCP6W6E

## Why

The audit of `root-bench-full-s0` (October 2026) found that the hidden fields
(u, v, p, T, s) are not determined by the problem as posed and trained:

1. Causal time weighting with a fixed eps schedule zeroed every PDE residual after
   the first ~0.5 ms of the event, so at 2 and 5 ms the fields obeyed only the
   interface and wall terms. Confirmed by the control runs `asym-probe-nocausal-s{0,1}`.
2. Even where the equations hold, the images plus equations leave null spaces:
   the vapour velocity (Darcy drag 37x weaker inside the bubble, cheap divergence),
   the flow along the interface, and the temperature (no inlet, initial or wall
   condition; one-way evaporation closure; a wall source of 0.004).
3. The evaporation closure is inconsistent: the energy sink and the volume source
   should follow from one mass flux, but the implemented coefficient is 0.992 where
   mass conservation gives (rho_l/rho_v - 1) Ja = 44.4, and the target is detached.
4. The boundary-value problem has no exact solution: no-slip side walls with a
   first-order Darcy law, a Young-Laplace jump that Darcy cannot supply at the caps,
   one continuous pressure network across a pressure discontinuity, walls placed at
   the crop edges instead of the channel walls, no outlet condition.

The goal is a reconstruction whose hidden fields pass five acceptance tests at
every frame: seed agreement, mirror symmetry, small residuals in every time
slice, flux balance at every cross-section, and a closed heat budget.

## Ground rules

1. **No problem-specific values in code.** Every physical input is a typed config
   field with its unit in the name (`T_inlet_C`, `heater_radius_um`). Per-series
   values live in `experiment`, modelling choices in `model`, optimisation in
   `training`, thresholds in `evaluation`. Values that only exist after
   preprocessing (wall rows, calibration) are read from the tensor archive's `meta`,
   never typed by hand. Anything per-series or per-run is editable in the app, and
   each such field is declared once, in the field table of WP7, from which the API
   and the web forms are generated.
2. **Old results stay reproducible.** Existing processed archives, runs under
   `outputs/`, and `paper/` are never overwritten. The 300 um calibration produces a
   new dataset name. Every new flag defaults to the current behaviour, so existing
   configs compose byte-for-byte; the new behaviour lives in new recipe files.
3. **Every new equation goes through the registry** (`physics/registry.py`), so the
   trainer, the API and the UI pick it up from one table.
4. **Each work package ships with** fast synthetic tests, a named benchmark recipe,
   and an acceptance gate that must pass before the next package starts. One branch
   and one pull request per package.
5. **Three seeds per benchmark.** A hidden field is reported only if the seeds agree.

## Inputs settled with the authors (2026-10-15)

| Input | Value | Where it lives | Note |
|---|---|---|---|
| Channel width | 300 um | `experiment.channel_width_um` | Was 291 in the stored archive. Changing it rescales x*, y* and the nose speed by 3.1%; needs a re-preprocess under a new dataset name. |
| Channel height | 150 um | `experiment.channel_height_um` | unchanged |
| Cavity position | centreline | `experiment.cavity_y_fraction = 0.5` | fraction of the wall-to-wall width; the streamwise position stays the measured root |
| Inlet temperature | saturation | `experiment.T_inlet_C` (None = T_sat) | table says 56 C; the fluid's T_sat is 56.6 C; default is the fluid value |
| Initial liquid temperature | 58 C | `experiment.T_init_C` | 1.4 K above the fluid's T_sat; set 58.6 if exactly 2 K is intended |
| Side walls, thermal | adiabatic, at the real wall rows | `experiment.side_wall_thermal = "adiabatic"`, `model.walls_at = "physical"` | wall rows come from preprocessing meta |
| Cavity heater | disk of radius 100 um at the cavity, q'' per series (2, 3, 4 W/cm2) | `experiment.q_cavity_W_cm2`, `experiment.heater_radius_um`, `experiment.heater_on_ms` (None = on throughout) | assumption: on for the whole event; the 0.3 V pulse is only the trigger |
| Background wall flux | uniform along the channel, sized so the overall latent heat supplied meets the growth need (page 2 of the authors' table) | `experiment.q_wall_mode = "fit"`, `experiment.q_wall_W_cm2` (fixed value, or the initial guess; None = energy-balance estimate from the masks), `experiment.q_wall_max_W_cm2` | implemented as ONE learned scalar bounded in [0, max], constrained by the observed growth through the thermal chain, initialised from a global energy balance of the measured growth; reported in W/cm2 next to that estimate |
| Side walls, momentum | no-slip with a Brinkman term, impermeable | `model.momentum_closure = "brinkman"`, `model.wall_condition = "no_slip"` | no-penetration-only remains available |
| Vapour pressure | uniform p_v(t) | existing | unchanged |
| Interface pressure | Young-Laplace | existing, plus `model.jump_model = "park_homsy" | "plain"` | default keeps the validated Park-Homsy weights |
| Inlet flow | 5 mL/h | `experiment.flow_rate_mL_hr` | unchanged |
| Outlet | long channel, pressure unknown | `model.outlet_condition = "zero_gradient"` | plus the integral mass-balance term below |
| Interface temperature | saturation | `model.interface_temperature = "saturation"` | conduction-limited evaporation; no fitted resistance. The kinetic (Schrage) route stays as an option for other problems. |
| Mass conservation along the channel | enforced and audited | `training.weights.mass_balance`, audit stage | volume flux at every cross-section equals the inlet flux plus the vapour created upstream |

## The physics after the change

Dimensionless, lengths on L_ref, velocities on U_ref, time on t_ref = L_ref/U_ref,
superheat theta = (T - T_sat)/dT_ref with dT_ref = T_init - T_sat (config
`scales.dT_ref_K`, None = this).

Liquid momentum (Brinkman-Hele-Shaw), evaluated where alpha < 1/2:

    grad p_l = -C_HS mu~(alpha) u + (1/Re) mu~(alpha) lap u,   C_HS = 12 (L/h)^2 / Re

Continuity everywhere: div u = s. Vapour: the Darcy law with mu_v/mu_l = 1/37
makes the vapour pressure uniform and the vapour velocity a potential flow, so
inside the bubble: curl u = 0 (new residual `vapour_potential`). This removes the
vapour-velocity null space at no modelling cost.

Pressure representation: p = alpha_d p_v(t) + (1 - alpha_d) p_l(x, t) with alpha_d
the detached indicator; the jump condition reads the liquid branch on the front.

Liquid energy, no sink term (the latent heat is the boundary condition):

    d theta/dt + u . grad theta = (1/Pe) lap theta + q*(x, y, t) (1 - alpha)
    q* = q_bg_star * m  +  q_cav_star * H(r_h - |x - x_cavity|)
    (uniform background flux with the learned multiplier m >= 0 in fit mode, plus a
     smooth disk of radius heater_radius_um at the cavity; both gated by liquid contact)
    theta = 0 on the interface and inside the vapour
    theta = theta_in at the inlet, d theta/dn = 0 on the side walls, theta = theta_0 at t = t_min

Evaporation flux from the liquid-side normal gradient at the front, g = d theta/dn:

    volume source     s = C_m * sum_i g_i dl_i K_w(x - X_i),   C_m = (Ja/Pe) (rho_l/rho_v - 1)
    kinematic         V_n = u . n + (Ja/Pe) g                  (liquid side; the correction is ~1e-3)

K_w is a normalised Gaussian of width `model.source_kernel_width` (default alpha_eps)
and X_i, dl_i the front samples and their arclength weights, so the integral of s is
exactly C_m times the flux integrated along the front. This is the Hardt-Wondra
distribution with a kernel in place of the Helmholtz solve. The free source
network is kept as `model.source_mode = "free"` for ablations; the rebuilt recipe
uses `"thermal"`.

Temperature parameterisation `model.theta_parameterization = "gated"`:
theta = d_liquid(x, t) * N(x, t), with d_liquid the capsule's distance into the
liquid (zero on the front and in the vapour). The interface condition and the
vapour condition hold exactly, and g = N on the front. `"free"` keeps a plain
network with soft interface and vapour penalties.

Integral mass balance, at `training.mass_balance_stations` x-stations and
`training.mass_balance_times` times:

    int_wall^wall u dy  -  Q_in*  -  int_0^x int s dy dx'  =  0

For FC-72 at the current scales: Ja/Pe = 1.85e-4, C_m = 0.0219, Re = 215,
C_HS = 0.223, Brinkman wall layer ~ sqrt(1/(Re C_HS)) = 0.14 L = 43 um, thermal
layer ~ sqrt(t*/Pe) = 0.02 to 0.04 L = 6 to 12 um (the T net needs a larger
Fourier scale than the others: `model.per_field.T.fourier_scale`).

Magnitudes to expect: the cavity disk supplies 0.6 to 1.3 mW; the observed growth
needs ~3 mW of latent heat early and ~1.5 mW late; conduction from the initially
superheated liquid through a ~10 um layer over the interface can supply ~4 mW. The
global balance over the imaged footprint (5.0e-7 m2) puts the background flux near
0.5 W/cm2 for the latent need alone, before sensible heating of the through-flow
(every 1.5 K of outlet warming at 5 mL/h costs ~3.7 mW), so the fitted value may land
anywhere between 0.5 and 2 W/cm2. Whatever it is, the latent heat still has to pass
through the interface by conduction, so the thermal layer must be resolved.

## Work packages, in order

### WP0  Baseline freeze and 300 um re-preprocess
- Tag the current state; record the five published seeds and the two controls as
  the baseline in `outputs/`.
- Re-preprocess Series-1 with `channel_width_um=300` into a new dataset name
  (`Series-1-w300`), raw frames shared. Store wall rows, ROI, um/px in `meta` (already
  done) and add `y_wall_lo_star`, `y_wall_hi_star`, `y_center_star` to `meta`.
- Add a fast test that `Domain` reads the wall positions from `meta` and falls back
  to the crop edges with a warning for old archives.
- Gate: new archive loads; walls 1.000 L apart; old archive untouched.

### WP1  Training fixes
- `training.causal_gate_min_weight` (default None = today's fixed schedule; recipe
  0.99): eps advances only when every chunk's causal weight exceeds the threshold.
- Log per-chunk residuals (all collocation terms) to the checkpoint history every
  `log_every`; `evaluate` reports the final per-slice table.
- Rebalance during Stage B relative to the engagement step; `training.lbfgs_steps`
  (default 0) for a full-batch polish at the end.
- Recipe `configs/training/trusted.yaml`: 3000 warm-up + 9000 physics steps,
  rebalance every 500, gate 0.99, schedule [0.01, 0.1, 1, 10].
- Fast tests: the gate holds eps while a chunk weight is below threshold and advances
  when all exceed it; per-chunk logging shape.
- Bench: `trust-wp1-s{0,1,2}` on the old physics. Gate: per-slice Darcy and energy
  RMS below 0.1 at every slice; pressure seed correlation above 0.95 (already seen in
  the controls).

### WP2  Domain, walls, outlet, mass balance
- `model.walls_at = "crop" | "physical"`; samplers (walls, inlet, uniform collocation)
  use the wall positions; the geometry's width bound uses the wall half-width.
- `model.wall_condition = "no_slip" | "no_penetration"`; `model.momentum_closure =
  "darcy" | "brinkman"` (new registry equation `brinkman`, liquid-weighted). Darcy with
  no-slip stays allowed for old configs and logs a warning.
- `model.outlet_condition = "none" | "zero_gradient"` (new boundary batch at x_max).
- New registry term `mass_balance` (functional, outside the rebalancer like the
  front terms) with `training.weights.mass_balance`, `mass_balance_stations`,
  `mass_balance_times`.
- Fast tests: Brinkman residual vanishes on an analytic Poiseuille-type profile;
  mass balance vanishes on a synthetic divergence-free field plus a known source and
  detects a 10% defect; wall samples land on the wall rows.
- Bench: `trust-wp2-s{0,1,2}`. Gate: axial flux equals the inlet flux upstream of the
  root within 5% at every frame; wall-pressure difference RMS below 0.05.

### WP3  Pressure representation and vapour flow
- `model.pressure_representation = "single" | "two_phase"`; Laplace jump reads the
  liquid branch; momentum residual liquid-weighted.
- New registry equation `vapour_potential` (alpha-weighted curl-free velocity).
- Fast tests: the blended pressure equals p_v(t) where alpha = 1 and p_l where
  alpha = 0; the jump residual uses p_l on the front; curl residual vanishes for a
  synthetic potential flow.
- Bench: `trust-wp3-s{0,1,2}`. Gate: Darcy/Brinkman and Laplace residuals both below
  0.1 RMS at the same time; u seed correlation above 0.95 inside and outside the bubble.

### WP4  Thermal rebuild
- Config: `experiment.T_inlet_C`, `T_init_C`, `side_wall_thermal`, `T_wall_C`,
  `q_cavity_W_cm2`, `heater_radius_um`, `heater_on_ms`, `cavity_y_fraction`,
  `q_wall_mode`, `q_wall_max_W_cm2`; `scales.dT_ref_K`;
- Background flux in fit mode: one learned scalar on the model (`_log_q_wall_scale`,
  softplus, the `q_pulse_star` pattern), bounded by `q_wall_max_W_cm2`, initialised
  from `estimate_background_flux(data, fluid, geometry)`: latent power of the measured
  projected growth (rho_v h dA/dt h_lv) minus the cavity disk's input, divided by the
  imaged heated footprint. Reported in `metrics.json` as `q_wall_fit_W_cm2` beside
  `q_wall_estimate_W_cm2`.
  `model.thermal_model = "legacy" | "sharp"`, `interface_temperature = "saturation" |
  "kinetic"`, `theta_parameterization = "gated" | "free"`, `source_mode = "free" |
  "thermal"`, `source_kernel_width`.
- Registry: `energy_liquid`, `theta_inlet`, `theta_initial`, `theta_wall`,
  `theta_interface` (free parameterisation only), `evap_flux` (free source mode
  only, two-way, coefficient C_m). The legacy `energy`/`evap` pair stays for old configs.
- Kinetic mode keeps a resistance anchored to the Schrage value: `R = R_gamma_star *
  exp(w)`, with the mapping to `R_int_star` documented in the supplement derivation
  (`Ja * Pe * R_gamma_star`).
- Dimensional groups: `theta_in`, `theta_0`, `heater_radius_star`, `C_m` computed in
  `groups.py`, never in residuals.
- Fast tests: gated theta is zero on the front and inside the vapour, g equals N on
  the front; the kernel-distributed source integrates to C_m times the front flux
  integral to 1%; C_m equals (Ja/Pe)(rho_l/rho_v - 1) from the fluid config; the
  heater disk integrates to q'' pi r_h^2 in dimensional units; the background-flux
  estimate reproduces a synthetic growth curve's latent power; the fitted multiplier
  stays within its bound and is recorded in dimensional units.
- Bench: `trust-wp4-s{0,1,2}` with `per_field.T.fourier_scale` swept (10, 20, 30).
  Gate: heat budget closes within 30% at every frame; T seed correlation above 0.95;
  energy residual per slice below 0.1.

### WP5  Mirror symmetry option
- `model.mirror_symmetry` (default off): field nets averaged with their reflection
  about `y_center` (even for u, p, T, N, phi; odd for v); geometry centreline fixed at
  `y_center`; root anchor y set to `y_center`.
- Fast test: every accessor is exactly even or odd about the centreline.
- Bench: `trust-wp5-s{0,1,2}`. Gate: mirror breaking fraction below 1e-6 (exact) and
  the IoU not worse than WP4 by more than 0.005.

### WP6  Acceptance stage
- New pipeline stage `audit` after `evaluate`, writing `audit.json`: mirror breaking
  fraction per field per frame; per-time-slice residual RMS; flux balance per station
  per frame; heat budget per frame (power from s, from the interface flux, from the
  observed growth, from the heater); interface-condition residuals per side; pass/fail
  against `evaluation.acceptance.*` thresholds (`seed_corr_min`, `mirror_max`,
  `residual_rms_max`, `flux_tol`, `heat_budget_tol`).
- `scripts/bench_seeds.py`: launches N seeds (`--multirun training.seed=...`), runs
  the audit on each, computes cross-seed correlations, writes `ensemble.json`.
- Fast tests on synthetic fields for every metric.

### WP7  App exposure: declare each input once

Today nothing in the app is schema-driven. A new input must be declared in the
Python schema, in the API models and their Hydra-key maps, in the TypeScript types
and in a React spec list; string-valued options are refused by the advanced-overrides
gate (`apps/api/src/naviernet_api/services/overrides.py:105-183`); `evaluation.*` has
no route at all; the new-series modal hard-codes its defaults. Twenty new fields
typed four times each is how values drift, so WP7 adds one declarative layer and
renders from it.

- **One field table on the Python side**, `src/naviernet/config/fields.py`: for every
  user-facing field its Hydra key, label, unit, type (float, int, bool, enum with
  options), bounds, section (series conditions, physics choices, training,
  acceptance), whether it is baked into the tensors, and whether it is advanced. A
  fast test introspects the dataclasses and fails when a user-facing dataclass field
  is missing from the table or a table entry has no dataclass field.
- **API**: new `GET /config/fields` serving the table. `CONDITION_FIELDS`
  (`services/datasets.py:62-69`), `BAKED_CONDITION_FIELDS` (`data/preprocess.py:51-60`),
  `FORM_OWNED_KEYS` and the scalar-only gate (`services/overrides.py:41-89, 105-183`)
  are generated from the table, so an enum option passes the gate when the table
  declares it and `evaluation.acceptance.*` becomes reachable. `OperatingConditions`
  and `ConditionsUpdate` (`models.py:129-143, 194-207`) gain the new experiment fields;
  `RunLaunchRequest` (`models.py:334-624`) and `_form_overrides`
  (`services/run_manager.py:577-642`) gain the choices; `conditions_from_cfg`
  (`services/datasets.py:764-780`) is built from the table.
- **Web**: the table reaches the web as a committed copy
  (`apps/web/src/lib/fieldTable.json`, written by `scripts/export_field_table.py`;
  `tests/test_field_table.py` fails when it is stale), so the forms render from it
  without a request and the web tests need no stub. The Datasets conditions editor
  (`apps/web/src/views/datasets/conditions.tsx`, `EditConditionsModal.tsx`,
  `NewSeriesModal.tsx`) renders the thermal section from the table (a number with
  unit and bounds, or a listbox), a "Physics choices" band on the Solver page renders
  the choices from it (`views/solver/PhysicsChoicesBand.tsx`, `form.ts`,
  `formulation.ts`, `recipes.ts` with a "Trusted fields" recipe), and an "Acceptance
  audit" panel on the Results page shows the verdicts the evaluate stage wrote into
  `metrics.json` with pass/fail (`views/results/AcceptancePanel.tsx`). `lib/api.ts`
  types follow the table's JSON shape. Existing hand-written fields stay and read
  their bounds, units and labels off the table.
- **Registry flags**: `enabled_equations` (`physics/registry.py:497-520`) gains the new
  gating flags (momentum_closure, outlet_condition, pressure_representation,
  thermal_model, interface_temperature, source_mode) and its callers are updated
  (`training.py:1202, 1640`, `evaluation.py:179-183`, `routes/physics.py:26-28`,
  `services/datasets.py:519`); `EquationState` gains an `enabled_by` field so the web
  can say why a term is on.
- **Series defaults**: an uploaded series inherits `highest_t`'s experiment block
  (`services/datasets.py:553-587`). The new experiment fields default to None, so a new
  series never silently inherits Series-1's temperatures; the editor shows "not set".
- **Baked inputs**: `channel_width_um` already carries the re-preprocess marker; the
  new thermal and heater fields do not change the tensors and need none.
- **Run record**: inputs live in the `.hydra/config.yaml` snapshot
  (`run_manager.py:726-733`); `run.json` (`services/run_manifest.py:50-77`) gains a
  `config_sha256` so a run is matched to its exact inputs; `SCHEMA_VERSION` bumps.
- **Tests**: `test_defaults_parity.py` compares request defaults with the dataclass
  defaults for every table entry (today: against `stage_a.yaml`, listed fields only);
  `test_overrides_api.py`'s rejection matrix, `test_physics_api.py`'s equation set, and
  the web suites (`recipes.test.ts`, `solver.test.tsx`, `physics.test.tsx`,
  `datasets.test.tsx`) get fixtures generated from the table.

### WP8  Manuscript
- Only after the full chain passes: replace the hidden-field section with the
  trusted fields, add the evaporation derivation to the supplement, state the 300 um
  calibration, and add the acceptance table.

## New inputs, all in one place

| Field | Group | Unit / type | Default | In app | Used by |
|---|---|---|---|---|---|
| `T_inlet_C` | experiment | C, optional | None = T_sat | yes, series conditions | theta_inlet |
| `T_init_C` | experiment | C, optional | None (no initial term) | yes | theta_initial, dT_ref |
| `side_wall_thermal` | experiment | adiabatic / isothermal | adiabatic | yes | theta_wall |
| `T_wall_C` | experiment | C, optional | None | yes (isothermal only) | theta_wall |
| `q_cavity_W_cm2` | experiment | W/cm2, optional | None = no cavity heater | yes | cavity disk source |
| `heater_radius_um` | experiment | um, optional | None = no cavity heater | yes | cavity disk source |
| `heater_on_ms` | experiment | ms, optional | None = always on | yes | cavity disk source |
| `q_wall_mode` | experiment | fixed / fit | fixed | yes | background flux |
| `q_wall_W_cm2` | experiment | W/cm2, optional in fit mode | existing | yes | background flux value or initial guess |
| `q_wall_max_W_cm2` | experiment | W/cm2 | 10 | advanced | bound on the fitted flux |
| `cavity_y_fraction` | experiment | 0..1 | 0.5 | yes | heater centre, mirror axis |
| `channel_width_um` | experiment | um | existing | yes, marks re-preprocess | calibration, walls |
| `dT_ref_K` | scales | K, optional | None = T_init - T_sat, else conduction scale | yes | theta scaling |
| `walls_at` | model | crop / physical | crop | yes | samplers, geometry bound |
| `wall_condition` | model | no_slip / no_penetration | no_slip | yes | bc term |
| `momentum_closure` | model | darcy / brinkman | darcy | yes | momentum residual |
| `outlet_condition` | model | none / zero_gradient | none | yes | outlet batch |
| `pressure_representation` | model | single / two_phase | single | yes | pressure, jump |
| `jump_model` | model | park_homsy / plain | park_homsy | yes | curvature |
| `thermal_model` | model | legacy / sharp | legacy | yes | energy residuals |
| `interface_temperature` | model | saturation / kinetic | saturation | yes | flux closure |
| `theta_parameterization` | model | gated / free | gated | yes | temperature accessor |
| `source_mode` | model | free / thermal | free | yes | source accessor |
| `source_kernel_width` | model | L_ref units | alpha_eps | advanced | source distribution |
| `mirror_symmetry` | model | bool | false | yes | accessors, geometry |
| `causal_gate_min_weight` | training | optional float | None | yes | causal eps schedule |
| `lbfgs_steps` | training | int | 0 | yes | final polish |
| `weights.mass_balance`, `mass_balance_stations`, `mass_balance_times` | training | float, int, int | 1.0, 8, 8 | advanced | mass_balance term |
| `weights.brinkman`, `weights.vapour_potential`, `weights.theta_*`, `weights.evap_flux` | training | float | 1.0 | advanced | new terms |
| `acceptance.*` | evaluation | thresholds | 0.95, 0.05, 0.05, 0.05, 0.30 | yes, results page | audit stage |

## Open points for the authors

1. DECIDED 2026-10-15: saturation temperature is the fluid's 56.6 C and the inlet equals
   it (the 56 C in the table was a reporting slip).
2. Initial superheat: 58 C gives 1.4 K over 56.6 C. Set `T_init_C = 58.6` for 2.0 K.
   Default 58.0 C as given.
3. DECIDED 2026-10-15: the cavity heater is on for the whole event, and the uniform
   background heater comes IN ADDITION to the cavity disk, constant in time and uniform
   in space, with its magnitude set so the overall heat supplied maintains the ~3 mW
   latent need. Implemented as one fitted constant (recommended) initialised from the
   energy-balance estimate; `q_wall_mode = "fixed"` remains available.
4. The 300 um calibration changes every x*, y* and the nose speed by 3.1%. The paper's
   current numbers remain tied to the 291 um archive until WP8.
5. Three heat-flux values: treated as three series, each with its own `q_wall_W_cm2`.

## Sequence and gates

WP0 -> WP1 -> WP2 -> WP3 -> WP4 -> WP6 (audit can be built in parallel from WP1) ->
WP5 -> WP7 (UI follows each package's schema change) -> WP8.
Each package is merged only when its gate passes on three seeds of `Series-1-w300`.

## Implementation log

- 2026-10-15, WP0 (#2): baseline tag `baseline-2026-10-15`; wall positions in the
  tensor meta and on `Domain`; `Series-1-w300` preprocessed at 300 um (1.685 um/px,
  walls 1.000 apart). The archive carries `n_frames_event = 10` (the CLI's
  `highest_t` value); every benchmark in this series uses that same archive so the
  packages compare like for like. An 11-event-frame archive for the manuscript is a
  WP8 item.
- 2026-10-15, WP1 (#3): gated causal schedule, per-chunk residual record, trusted
  recipes. Benchmark `trust-wp1-s{0,1,2}` launched from a frozen worktree of the
  WP1 commit.
- 2026-10-15, WP1 benchmark result (`trust-wp1-s{0,1,2}`, reported on #3): gate
  FAILED on the old physics. IoU 0.974-0.976 (validation frame 0.965-0.968); seed
  correlation u 0.88, v 0.69, p 0.64, T -0.36, s 0.64; audit mirror 0.55-0.72,
  residual RMS 1.6-1.9, flux defect 4.3-4.9, heat spread 0.28-0.74; per-chunk Darcy
  up to 7. The record showed the causal gate advancing inside the Stage-A warm-up
  (fixed in #10: the gate is armed only once every collocation term is in the loss)
  and the gradient-norm rebalancer driving the Darcy weight to 0.03-0.8 while the
  kinematic weight climbed to 300-900. The WP2 bench onwards runs from a worktree
  at the #10 commit (`navier-net-bench`).
- 2026-10-15, WP2 (#4): physical walls, Brinkman no-slip, outlet, mass balance.
  Joint runs refuse the mass balance for now.
- 2026-10-15, WP3 (#5): two-phase pressure, curl-free vapour. Joint runs refuse it.
- 2026-10-15, WP4: sharp thermal model built with `interface_temperature =
  saturation` and `theta_parameterization = gated`; the kinetic interface and the
  free parameterisation are declared in the schema but refused by name until built
  (the legacy model carries the kinetic route). The thermal source needs binned
  collocation times. `configs/experiment/trusted_w300.yaml` holds the authors'
  thermal conditions and heaters; `highest_t` is untouched so the published runs
  and the paper scripts keep composing what they were made with. Joint runs refuse
  the sharp thermal model.
- 2026-10-15, WP5 (#8): `model.mirror_symmetry` (opt-in): every field network
  averaged with its reflection about the centre of the wall span (v odd), the
  capsule centreline pinned to the axis; `configs/model/trusted_mirror.yaml`.
  Joint runs refuse it.
- 2026-10-15, WP6 (#7): `naviernet/audit.py` run by `evaluate` on every checkpoint
  (`audit.json`, `audit_fields.npz`, verdicts in `metrics.json`); thresholds under
  `evaluation.acceptance`; `scripts/audit_ensemble.py` for the seed-agreement test.
- 2026-10-15, WP7 (#9): the field table `naviernet/config/fields.py` (45 entries:
  series conditions, physics choices, training switches, acceptance thresholds)
  served at `GET /api/config/fields`; the API's condition allow-list, the baked-field
  list, the form-owned key set and the launch form's override list are generated
  from it, and the override gate accepts a table-declared choice or threshold by
  its declared kind, options and bounds (so `evaluation.acceptance.*` and the named
  physics choices are reachable). `RunLaunchRequest` carries the physics choices and
  training switches with the schema's defaults and mirrors the trainer's dependency
  tree; the launcher refuses a choice the series cannot carry (no `p`, no `T`, no
  wall positions in the tensors) before a slot is taken. `run.json` v2 records the
  SHA-256 of the config snapshot. The web renders from a committed copy of the table
  (`apps/web/src/lib/fieldTable.json`): thermal conditions in both series modals,
  a "Physics choices" band and a "Trusted fields" recipe on the Solver, the
  acceptance verdicts on the Results physics tab, and the admitting choice on each
  equation's popover. The "Trusted fields" recipe is a committed export of what
  `model=trusted training=trusted` compose to (`apps/web/src/lib/trustedRecipe.json`,
  same script), so the card cannot drift from the CLI recipe. A conditions edit is
  checked as a whole before it is saved (the groups refuse a cavity flux without a
  radius, isothermal walls without their temperature, an initial temperature below
  saturation, a fitted flux capped below its start). Parity tests hold the table,
  the schema dataclasses, the request models, the form defaults, the request
  bounds and both committed copies equal.

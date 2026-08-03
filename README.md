# naviernet

**Physics-informed neural networks for multiphase flow in microchannels.**

Ten high-speed camera frames, 0.5 ms apart, of a vapour bubble growing in a
heated 300 × 150 µm channel. From them, `naviernet` reconstructs the *continuous*
interface and velocity fields — resolved at any instant between the frames — by
training a neural network that is constrained to satisfy the governing
conservation laws, not merely to interpolate the pixels.

The test of whether that worked is the holdout frame: one camera frame is
withheld from supervision entirely, and the model is asked to predict it.

| Result (Stage A, 1500 steps, laptop CPU) | |
| --- | --- |
| **Holdout frame 6 IoU** (never supervised) | **0.968** |
| IoU across frames 1–10 | 0.93 – 0.98 |
| Inferred nose speed *(no velocity data was ever supplied)* | 177 mm/s vs 180 mm/s measured |
| Bretherton film thickness | 4.9 µm, matching the imaged side films |
| Energy budget closure | implies ≈1.9 W/cm² against a 2 W/cm² setpoint |

---

## Quick start

```bash
make run                      # everything: install deps, then start the platform
```

One command installs the toolchain ([uv](https://docs.astral.sh/uv/)-managed
Python pinned by `.python-version`, both editable packages, npm dependencies)
and boots the API plus the web dev server, allocating free ports automatically
(defaults 8000/5173; if a port is taken the next free one is used and printed).
`make stop` tears everything down; `make status` shows what is running.

For the solver pipeline only:

```bash
make setup                    # install dependencies (idempotent)
naviernet stage=all           # preprocess -> train -> evaluate -> figures -> video
```

The raw TIFFs are not distributed in this repository (see
[data/raw/README.md](data/raw/README.md)); drop them into
`data/raw/highest_t/` first. `ffmpeg` must be on `PATH` for the video stage
only. A full run takes a few minutes on a laptop CPU.

Stages are independent and each reads its inputs from disk, so they can be run
one at a time or repeated without redoing the ones before:

```bash
naviernet stage=preprocess
naviernet stage=train training.steps=500     # resumable; run it again to continue
naviernet stage=evaluate
naviernet stage=figures
naviernet stage=video video.n_timesteps=200
```

## The platform (web UI)

The same pipeline is drivable from the browser — upload/preprocess, live
governing equations and model topology, a solver with a streaming loss
console, seed sweeps, run comparison, and an interactive interface
reconstruction.

```bash
make run          # development: API + Vite dev server (hot reload, /api proxied)
make serve        # production mode: build the UI, serve everything on one port
```

In production mode one process serves the whole platform (the API under
`/api`, the built UI everywhere else) on the API port — pin it in a local
`.env` via `NAVIERNET_API_PORT`. In development, `NAVIERNET_WEB_PORT` pins
the Vite port the same way. Busy ports fall through to the next free one
automatically.
The server is **local/trusted-network only**: it has no
authentication and can start training jobs and write under the repository, so
do not expose it to the public internet as-is.

## Configuration

Configuration is [Hydra](https://hydra.cc)-composed from typed dataclasses, so
every value is overridable from the command line and a wrong key or a wrong type
fails immediately rather than silently producing wrong physics:

```bash
naviernet stage=all model.alpha_eps=0.03 run_name=sharper   # sharper interface
naviernet --multirun training.seed=0,1,2                    # seed sweep
naviernet --cfg job --resolve                               # print config, run nothing
```

`configs/` holds one directory per config group. Swapping a group swaps a whole
coherent block of settings — a different fluid, a different dataset's operating
conditions — rather than a scattering of individual values:

```
configs/
├── config.yaml            # composition root, run/stage selection, Hydra settings
├── experiment/            # operating conditions per imaged dataset
├── fluid/                 # saturated two-phase properties
├── imaging/               # segmentation and calibration parameters
├── scales/                # non-dimensionalisation
├── model/                 # network architecture
└── training/              # optimisation and loss weights
```

The schema those YAML files are validated against lives in
[src/naviernet/config/schema.py](src/naviernet/config/schema.py). Derived
quantities — reference time, every dimensionless group — are deliberately *not*
configurable: they are computed in [physics/groups.py](src/naviernet/physics/groups.py)
so they can never drift out of sync with the inputs they come from.

## Layout

```
configs/                   Hydra config groups (above)
data/
├── raw/<dataset>/         input TIFFs, numbered from 1        [gitignored]
└── processed/<dataset>/   tensors + preprocessing QC figure   [gitignored]
outputs/<run_name>/        checkpoints, metrics, figures,      [gitignored]
                           video, and Hydra's config snapshot
src/naviernet/
├── cli.py                 Hydra entry point (the `naviernet` command)
├── pipeline.py            stage orchestration
├── config/schema.py       typed configuration schema
├── data/
│   ├── preprocess.py      wall detection, segmentation, tensor assembly
│   └── dataset.py         interface-weighted sampling, holdout split, domain
├── models/
│   ├── layers.py          Fourier features, adaptive tanh
│   └── pinn.py            per-field MLP ensemble; alpha = sigmoid(phi/eps)
├── physics/
│   ├── groups.py          dimensionless groups, derived from config
│   └── residuals.py       autodiff PDE residuals and boundary losses
├── training.py            resumable trainer, gradient-norm loss rebalancing
├── evaluation.py          IoU, nose trajectory, growth kinematics
├── viz/                   QC plot, result figures, slow-motion video
└── utils/                 run layout, logging
tests/                     fast unit tests + data-dependent integration tests
```

`data/` and `outputs/` are ignored by git in their entirety — they hold large
binaries and fully regenerable results. Everything under `outputs/` can be
rebuilt with `naviernet stage=all`.

Each run writes its fully-resolved config to `outputs/<run_name>/.hydra/`, so
any result can be reproduced exactly from the directory it landed in.

## The experiment

FC-72 at T_sat = 56.6 °C in a horizontal 300 × 150 µm channel, bottom wall
heated at 2 W/cm², flowing at 5 mL/hr. Frames every 0.5 ms; flow runs right to
left in the raw images, and the x axis is flipped during preprocessing so that
downstream is +x. Nucleation at a wall cavity is triggered by a 0.3 V pulse near
the inlet, whose effect is absorbed into the frame-1 initial condition.

Frames 1–10 are one continuous growth event. Frame 11 is cut by the edge of the
field of view and is masked rather than discarded; frame 12 belongs to the next
ebullition cycle and is excluded. Inlet temperature is uncertain (~55 °C at the
ceiling) and is treated as an inverse unknown in Stage B.

Derived operating point: Re = 215, We = 2.30, Ca = 0.0107, Pr = 9.41,
Hele-Shaw drag = 0.223, Bretherton film = 4.9 µm.

## How it works

**Volume fraction is never predicted directly.** A network outputs a level-set
field `phi`, and `alpha = sigmoid(phi / eps)`. This bounds alpha in (0, 1) by
construction — no clamping, no penalty needed to keep it physical — and makes
the interface half-thickness `eps` an explicit parameter that can be annealed.

**Supervision targets are smoothed, not binary.** The network is fit against
`sigmoid(-sdf / eps)` rather than the raw 0/1 mask, because fitting a
smeared-but-controlled profile is far easier than fitting a step.

**Sampling is interface-weighted.** Supervised and collocation points are drawn
with probability peaking at the interface, so neither is wasted on uniform bulk
liquid where nothing happens.

**Loss weights are rebalanced, not hand-tuned.** Every `rebalance_every` steps
the per-term gradient norms are measured and the weights nudged so no single
term dominates the others (`training.weighting=gradnorm`, the default). This
gradient-norm rebalancer is unbounded, though: on long runs it can ratchet a
physics weight to its cap and crush the data fit, so it needs a hand-picked
`rebalance_every`.

**Accuracy techniques (opt-in, all default-off).** Three literature methods are
implemented behind config flags; an unchanged config trains exactly as before.

- **Residual-Based Attention** (`training.weighting=rba`; Anagnostopoulos et al.,
  [arXiv:2307.00379](https://arxiv.org/abs/2307.00379)) replaces the rebalancer with a
  *bounded* per-point, per-term attention on a fixed collocation pool. Its weights
  cannot exceed `rba_eta/(1-rba_gamma)`, so it trains stably with **no** `rebalance_every`
  tuning and no weight blow-up — the robust choice for long runs.
- **Temporal causal weighting** (`training.causal_weighting=true`, `causal_mode=weight|march`;
  Wang et al., [arXiv:2203.07404](https://arxiv.org/abs/2203.07404)) weights the PDE
  residual by time so the network learns forward in time (early times before late).
- **Residual-adaptive collocation** (`training.adaptive_collocation=true`, requires RBA;
  Wu et al., [arXiv:2207.10289](https://arxiv.org/abs/2207.10289)) periodically refreshes
  the collocation pool toward high-residual regions, keeping a uniform base.

**Measured front velocity** (`training.front_velocity=true`, requires
`model.front_geometry`). The interface's *shape* is supervised at every training
frame; its *rate* is not — and under the front geometry the rate is what carries
the shape past the last supervised frame. This measures it from the masks and
supervises it: the front's normal velocity via the level-set form
`v_n = -(sdf_next - sdf_now) / (dt·|∇sdf_now|)`, plus the nose apex's tracked 2-D
displacement. Only a pair of *consecutive, training-visible* frames is used, so a
held-out frame is never differenced into a training target.

Measured on Series-1 (2 seeds, seed spread 0.003), it is **neutral**: held-out
frame 11 goes 0.8705 → 0.8720 and the trained-frame mean is unchanged at 0.9275.
Two results worth keeping from getting there, both recorded rather than tidied
away:

- Supervising the nose with the level-set estimate actively **hurt** (frame 11
  −0.0075, same sign on both seeds). The estimate is first-order in the distance
  the front travels, and at the nose that is ~52 px per frame against a ~75 px
  cap radius: the nose cap was 10% of the profile bins and 76% of the term's
  residual, pulling the nose *down* against the apex term that had correctly
  pulled it up. The nose is now left to the apex term, which tracks a real
  feature and involves no approximation.
- The likeliest reason for the neutral result is that `v_n` is not independent
  information — it is a time derivative of the same masks the shape supervision
  already fits to 0.93 IoU. It is a different functional of the same data, not
  new data.

Measured on Series-1 (3000 steps, held-out tail frames): RBA fixes the rebalancer's
mean-collapse (mean IoU 0.88 → 0.90, weights bounded, no manual tuning). None of the
three reliably lifts the *last* held-out extrapolation frame above the tamed-rebalancer
baseline — extrapolating past all supervision is genuinely hard, and this is recorded
rather than papered over. See `.memory/pinn-accuracy/` for the full per-technique numbers.

Stage A solves VOF transport plus continuity with an *inferred* dilatation
source `s`, penalised away from the interface so it cannot become a free sink
absorbing divergence errors. The equations, and the Stage-B terms that replace
`s` with a real evaporation closure, are written out in
[physics/residuals.py](src/naviernet/physics/residuals.py).

**Known gap:** global mass closure of the free dilatation source is not yet
quantitative. Stage B fixes this by construction.

## Stage B roadmap

1. Add `p` and `T` networks; momentum with Hele-Shaw drag and CSF surface tension.
2. Energy equation with wall source q″(x) = 2 W/cm² plus a learnable δq(x) near
   the pulse heater; inlet temperature as a bounded trainable scalar.
3. Replace the free source with the Hardt-Wondra closure
   `j_evap = (T_int − T_sat) / (R_int · h_lv)`.
4. Interface-sharpening curriculum (anneal `model.alpha_eps`); validate film
   thickness against Bretherton.
5. Transfer across the other heat-flux datasets — a new `data/raw/<name>/` and a
   new `configs/experiment/<name>.yaml`, no code change.

Stage-B fields are already wired: list them in `model.fields` and
`BubblePINN.pressure` / `.temperature` become available.

## Development

```bash
make test        # fast Python suite + web unit tests
make test-all    # everything: slow + data-dependent tests, web, e2e
make lint        # ruff, eslint, tsc (mirrors the CI lint jobs)
make lint-fix    # apply autofixes, then re-check
make help        # all targets
```

The Makefile is a thin dispatcher — each target calls a script under
`scripts/` (the runner scripts take `--help` for finer-grained flags).

The fast suite covers the config schema, the dimensionless groups (pinned to the
published values above), autodiff gradients against analytic derivatives, the
bounded-alpha invariant, checkpoint round-tripping, and the loss rebalancer.
Tests needing the raw dataset or a trained checkpoint are marked `needs_data`
and skip cleanly without them.

## Licence

GPL-3.0 — see [LICENSE](LICENSE).

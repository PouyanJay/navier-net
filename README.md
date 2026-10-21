<div align="center">

<img src="apps/web/public/brand/navnet-mark.svg" width="84" height="84" alt="NavierNet" />

# NavierNet

**Ten camera frames of a boiling bubble in, the velocity, pressure and temperature around it out.**

[![License: GPL v3](https://img.shields.io/badge/License-GPL_v3-2e6cca?style=flat-square)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](.python-version)
[![PyTorch](https://img.shields.io/badge/PyTorch-PINN-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](src/naviernet)

</div>

A high-speed camera sees only a bubble's outline. NavierNet fits a physics-informed neural network to
those outlines and to the conservation laws at once, and so recovers the fields the camera cannot see,
at any instant between the frames. The case studied is FC-72 boiling in a 300 × 150 µm heated channel.

## How it works

Training runs in two stages.

1. **Outline.** The bubble is an explicit capsule rooted at the nucleation site. It is fitted to the
   traced outlines (the centre of the dark rim, made mirror-symmetric about the channel), then frozen.
2. **Fields.** With the outline fixed, the network solves for the flow, pressure and temperature
   around it.

The physics, in the depth-averaged 2D model:

| | |
|---|---|
| Flow | Brinkman law: pressure against the floor and ceiling drag and in-plane viscous friction |
| Channel | Developed velocity profile at the inlet (10 mL/h); stress-free outlet at a fixed pressure |
| Bubble edge | Moves with the liquid; held at the boiling point; Young–Laplace imposed on average, which sets the vapour pressure |
| Heat | Liquid energy equation, two heater zones on the floor |
| Evaporation | A thin film under the bubble dries out as the floor heats it; the total vapour made is held to the bubble's measured growth |

## Where it stands

From the current default (one seed), measured against the images and against each equation's own terms:

| | |
|---|---|
| Outline against the frames | IoU 0.979 to 0.989 |
| Vapour pressure | about 180 Pa above the outlet, as Young–Laplace predicts for this channel |
| Equation errors near the bubble, relative to their terms | temperature 10–17%, continuity 7–11% |
| Still open | temperature in the open liquid (50–74%), flow next to the bubble (29–36%) |

Every run ends with an audit and a report: the five-check gate, these relative errors, and figures of
every field.

## Quick start

```bash
make run    # install everything, start the API and the web app, print the URLs
make test   # Python and web tests
make lint   # ruff, eslint, tsc
```

Run the pipeline from the command line:

```bash
naviernet dataset=Series-1-w300 experiment=trusted_w300 model=trusted training=trusted
naviernet --cfg job --resolve        # print the composed config, run nothing
```

On a rented GPU through [Modal](https://modal.com), with a report afterwards:

```bash
modal run scripts/modal_bench.py --mode bench --seeds 0 --prefix my-run
python scripts/run_report.py my-run-s0
```

The outline fit is cached and reused by every later run on the same outlines, settings and code, so a
run is about 45 minutes after the first.

The raw camera frames are not in the repository. Put TIFFs in `data/raw/<name>/` with a matching
`configs/experiment/<name>.yaml`, or upload them in the browser.

## Layout

| | |
|---|---|
| `src/naviernet` | The solver: models, physics, training, audit |
| `configs` | [Hydra](https://hydra.cc) configs; `trusted` is the default recipe |
| `apps/api`, `apps/web` | FastAPI server and React app over the same pipeline |
| `scripts` | Modal runs, run reports |

The server has no authentication and can start training jobs. Keep it on a local or trusted network.

## License

[GNU GPL-3.0](LICENSE)

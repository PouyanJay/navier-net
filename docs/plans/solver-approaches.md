# Solver approaches: how the bubble's shape and its physics are solved together

Status: 16 October 2026. Approach 1 is built and running (`training.freeze_geometry`,
PR #12). Approach 2 adds a validation gate (built, `scripts/run_report.py`) and a
coupled third stage plus a predictive test (not built). Approach 3 is a proposal.
Companion plan: `docs/plans/trustworthy-hidden-fields-plan.md`.

## The problem in one paragraph

We see the bubble's outline in eleven camera frames, 0.5 ms apart. We want the
fields we cannot see (velocity, pressure, temperature, the vapour source), and
we want them to be the ones the physics demands. In the real experiment the
physics drives everything: heat from the wall and the superheated liquid
evaporates the film under the bubble, the vapour pushes the liquid, and the
outline is the result. Our solver is a physics-informed network (PINN) that
fits the outline to the images and the fields to the equations at the same
time. The question this document answers is in what order, and how coupled,
those two jobs should be done.

## What went wrong when everything was trained at once

The trusted recipe trained the outline and the physics together, with an
automatic balancer adjusting the weight of each equation against the image
term.

| Run | What we saw |
|---|---|
| `wpf2-s0` (no film evaporation) | The physics outvoted the images (momentum weight 1000 against images 10). The outline drifted to IoU 0.74 and the nose slowed to 67 mm/s against the measured 181. |
| `trust-wpf4-s0/s1` (weights capped) | The outline was right (IoU 0.93 to 0.98, nose 176 to 178 mm/s), but the physics held only at the fixed instants it was checked at. Between them the flux error was 1.5 to 2.5 times the inlet flow. |
| `trust-wpf5-s0` (instants jittered) | The physics held equally at every instant (flux error 0.21), but the velocity and temperature stayed lopsided (mirror breaking u 0.44 to 0.50, v about 0.8, T 0.3 to 0.5). The momentum and jump residuals sat flat at about 0.3. |

The last run is the reason for the approaches below. The balancer held the
momentum weight below 1 against the images' 10 for the whole run. The physics
was enforced only as far as it did not pull the outline off the images, and
each seed's random asymmetry survived in that slack. Raising the physics
weights reproduces the first row. The two jobs fight when they are trained at
once with a balancer between them.

## Approach 1: two stages (decoupled)

1. **Fit the outline.** The warm-up trains the outline against the images, the
   measured front speed and the root shape, with light physics. It ends with an
   IoU of about 0.97.
2. **Freeze the outline, solve the fields.** Every parameter of the outline is
   frozen at that fit. Only the fields train (velocity, pressure, temperature,
   vapour pressure, the wet share of the footprint), against the equations. The
   frozen outline is a moving boundary: the kinematic condition makes the
   liquid move with it, and the Young-Laplace condition sets the pressure jump
   across it.

What it answers: given the bubble we saw, is there a flow, pressure and
temperature that obey every equation? If yes, the residuals fall to near zero
and the fields come out symmetric and the same for every seed. If not, the
physics in the model cannot produce the measured shape.

What it gives up:
- The physics can no longer correct the shape. The fitted outline carries pixel
  noise and smoothing, and the jump condition reads its curvature, a second
  derivative, which amplifies that noise.
- Whatever the images do not pin down (the shape between frames, the caps, the
  last frame) is decided by the fit, not by the physics.
- It cannot show that the physics causes the growth. The shape is an input.

Implementation: `training.freeze_geometry` (on in the trusted recipe). With the
images out of the loss, the balancer equalises the physics terms among
themselves. Early evidence from `trust-wpf6-s0`: the outline holds at 0.973 as
intended, but equal pull per term turns the momentum weight down (0.26) because
its gradient is the largest. If the final audit confirms this, stage 2 should
hold every physics term at its configured weight instead of rebalancing.

## The validation gate between stage 2 and stage 3

Before coupling, the decoupled solution is judged on its own. If it fails, the
coupled stage would inherit the problem and we would not know which stage
caused it. Five checks (all in the audit; `scripts/run_report.py` turns a run
into one page with the gate and every figure):

| Check | Passes when | Where |
|---|---|---|
| 1. The equations hold everywhere | Every residual is small at the camera frames and at 61 instants between them | `residuals`, `residuals_in_time` |
| 2. The seeds agree | Two or more seeds give the same velocity, pressure and temperature (correlation above 0.95) | `scripts/audit_ensemble.py` |
| 3. The fields are mirror-symmetric | u, p, T even and v odd about the centreline, breaking below 0.05 | `mirror` |
| 4. Mass and heat close | Flux at every cross-section equals inlet flow plus vapour made upstream, at the frames and between them; heat budget closes within 30 % | `flux`, `flux_in_time`, `heat` |
| 5. Where the shape blocks the physics | The Young-Laplace mismatch on the outline, expressed as the shape change that would remove it, stays within 2 camera pixels | `jump_shape` |

Check 5 turns the decoupled solution into a measurement of how far the measured
shape is from a physically consistent one. Within the pixel error, the coupled
stage should be a small correction. Larger, and the physics is missing
something, and coupling will not fix it.

## Approach 2: three stages with validation

1. Fit the outline (as in approach 1).
2. Freeze it and solve the fields (as in approach 1).
3. **Gate:** the five checks above.
4. **Couple within the measurement error.** Release the outline and train
   everything together, with the images treated as data with a known error
   rather than a target to maximise: the outline may move only as far as the
   measurement allows, about one to two pixels. Inside that band the physics
   decides the shape. The first coupled runs failed because the images
   outweighed the physics a hundred to one, not because coupling is wrong; a
   tolerance band sets that balance on principle instead of by tuning weights.
   Starting from the stage-2 fields means the coupled stage begins from a
   consistent state.
5. **Predictive test.** Train on frames 1 to 7 only, then let the physics
   carry the bubble through frames 8 to 10 and compare with the camera. That
   agreement cannot come from fitting, because those frames are never seen.

Stage 3 is the final solve: it makes the outline and the fields consistent with
each other. It shows that the physics and the images can agree. The predictive
test is the validation that the captured physics is what grows and moves the
bubble.

Not built yet: the tolerance band (stage 3) and the frames-1-to-7 predictive
run.

## Approach 3: fully coupled, the physics drives from the start

The physics produces the shape; the images are used only to check it and to
calibrate a handful of unknowns. One limit applies to every option: no
continuum model predicts nucleation itself. The run starts from a small seed
bubble at the cavity at the measured superheat, or from the frame-1 bubble.

| Option | What it is | Strength | Risk and cost |
|---|---|---|---|
| A. Forward PINN | The current network given only frame 1, the boundary conditions and no other images; the interface conditions must move the front | Uses the machinery we have | PINNs are weak at long forward problems with a moving interface and tend to drift or stall |
| B. Differentiable 2-D forward solver | Our depth-averaged equations (Brinkman flow, uniform vapour pressure, Young-Laplace with Bretherton, film evaporation, liquid temperature) solved step by step on a grid, with a level set or front tracking for the interface, written in PyTorch so the few unknowns (wet share, contact angle, evaporation resistance) are fitted by gradient descent through the simulation | The physics holds exactly by construction; the data tune only a few scalars; minutes per run on a CPU; Hele-Shaw moving-boundary problems are well studied | About a week for a validated first version, mostly the interface jump conditions |
| C. Full 3-D two-phase simulation | Volume of fluid with phase change (Basilisk, OpenFOAM) resolving the gap direction, the film and the nose | The most physical | The microlayer is thinner than any affordable grid and still needs a submodel; weeks to set up, serious compute |

Recommendation: option B. It answers the question directly. If the calibrated
physics reproduces the outline and its growth over time, the physics is what
drives the bubble. If it does not, we learn which part of the physics is
missing, and a network cannot hide it by bending fields. It also gives an
independent check of the PINN's hidden fields from a completely different
method.

Caveats shared with the PINN: depth averaging is an approximation at the nose
and in the films, and the wet share and contact angle stay calibrated rather
than predicted until the film and the wall are modelled from first principles
(the conjugate wall and film thinning of case 4 in the boundary-condition
discussion).

## Side by side

| | Approach 1: two stages | Approach 2: three stages + gate | Approach 3: fully coupled |
|---|---|---|---|
| Who sets the shape | The images | The images, then the physics within the pixel error | The physics |
| What it proves | The measured shape admits a physical flow, or where it does not | The images and the physics agree; with the predictive test, the physics drives the growth | The physics alone reproduces the growth |
| Status | Built, running (`trust-wpf6-s0`) | Gate built; stage 3 and the predictive test not built | Proposal |
| Cost to finish | One run per seed (about $4 each) | About a day of work plus runs | About a week for option B |

## Order of work

1. Finish approach 1: read the `trust-wpf6-s0` report; if the momentum weight
   was the problem, hold the physics weights fixed in stage 2 and rerun; then a
   second seed for the agreement check.
2. When the gate passes, build stage 3 and the predictive test (approach 2).
3. Build approach 3, option B, as the independent check, in parallel or after.

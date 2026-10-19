"""Typed configuration schema.

These dataclasses are the *single source of truth* for every tunable quantity in
the platform: experimental conditions, fluid properties, imaging parameters,
non-dimensionalisation, network architecture, and optimisation settings.

They are registered with Hydra's :class:`~hydra.core.config_store.ConfigStore`,
so the YAML files under ``configs/`` are type-checked at composition time: a
typo in a key name or a string where a float belongs fails immediately with a
clear error rather than silently producing wrong physics.

Derived quantities (reference time, dimensionless groups) deliberately do *not*
live here -- they are computed from these inputs in :mod:`naviernet.physics`,
so they can never drift out of sync with the values they depend on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

# Stages of the pipeline, in dependency order. ``all`` runs each in turn.
STAGES = ("preprocess", "train", "evaluate", "figures", "video")


@dataclass
class ExperimentConfig:
    """Operating conditions of one imaged ebullition dataset."""

    name: str = MISSING
    # Both derived from the selected fluid group (``${fluid.name}`` /
    # ``${fluid.T_sat_C}``): the working fluid fixes its own label and its
    # atmospheric saturation temperature. Kept on the experiment so every
    # existing consumer (groups, residuals, API) reads one stable path.
    fluid: str = MISSING
    T_sat_C: float = MISSING  # saturation temperature at operating pressure (from fluid)
    q_wall_W_cm2: float = MISSING  # bottom-wall heat flux setpoint
    flow_rate_mL_hr: float = MISSING
    channel_width_um: float = MISSING  # imaged (in-plane) direction
    channel_height_um: float = MISSING  # depth; bottom wall is heated
    dt_frame_ms: float = MISSING  # camera inter-frame time
    flow_direction: str = MISSING  # direction of flow in the raw camera frames

    n_frames_raw: int = MISSING  # TIFFs present on disk
    n_frames_usable: int = MISSING  # end of the usable window (1..n)
    n_frames_event: int = MISSING  # frames of one continuous growth event

    # 1-based camera frames dropped from the usable window (a mis-segmented
    # frame, a stray reflection). Absolute frame times are preserved, so the
    # frames that remain keep their true t* -- only tensor rows disappear.
    excluded_frames: list[int] = field(default_factory=list)

    notes: str = ""

    # --- Thermal conditions and heating (trustworthy-fields WP4). Every field is
    # optional so an existing series composes unchanged; the sharp thermal model
    # (model.thermal_model=sharp) reads them. Temperatures in degrees C. ---
    T_inlet_C: Optional[float] = None  # noqa: UP045 -- liquid at the inlet; None = saturation
    T_init_C: Optional[float] = None  # noqa: UP045 -- the whole liquid before frame 1
    # The thermal layer the bubble already has at the first frame: the initial
    # temperature is imposed only on liquid farther than this from the frame-1
    # outline, and the heat equation forms the layer inside it. The bubble is
    # already 0.57 mm long at frame 1, so the liquid touching it has been
    # cooling for milliseconds; imposing T_init right up to the interface
    # forced a jump to saturation across zero distance at the first instant.
    # 0 imposes it everywhere in the liquid (existing runs).
    T_init_layer_um: float = 0.0
    # The floor heated in two zones along the channel: q_upstream_W_cm2 from the
    # field of view's inlet edge to heater_edge_um, q_wall_W_cm2 beyond. Both
    # None: one flux, q_wall_W_cm2, over the whole floor (existing series).
    q_upstream_W_cm2: Optional[float] = None  # noqa: UP045
    heater_edge_um: Optional[float] = None  # noqa: UP045
    side_wall_thermal: str = "adiabatic"  # "adiabatic" | "isothermal"
    T_wall_C: Optional[float] = None  # noqa: UP045 -- side-wall temperature (isothermal only)
    # Film evaporation under the bubble (model.footprint_evaporation): the wall
    # flux under the footprint evaporates the liquid film the bubble leaves on
    # the heated wall, over the share of the footprint that is still wet.
    # "fixed" uses wet_fraction as given; "fit" starts ONE bounded scalar there
    # (strictly inside (0, 1)) and lets the observed growth set it through the
    # chain. Reported beside the value the energy balance of the masks implies.
    wet_fraction_mode: str = "fixed"  # "fixed" | "fit"
    wet_fraction: float = 1.0  # 0..1; the wet share of the footprint
    # How the footprint stays wet. "uniform": the wet share above, constant over
    # the event. "microlayer": the bubble leaves a film of microlayer_um on the
    # floor it covers, and the floor's local flux evaporates it, so a spot is wet
    # only for microlayer_um * rho_l * h_lv / q after the images first show it
    # covered; vapour then comes from newly covered floor. The uniform share is
    # not read under the microlayer. footprint_age_ms is how long before the
    # first frame the first frame's footprint was covered (0: freshly covered).
    footprint_wetting: str = "uniform"  # "uniform" | "microlayer"
    microlayer_mode: str = "fixed"  # "fixed" | "fit"
    microlayer_um: float = 0.5
    footprint_age_ms: float = 0.0


@dataclass
class FluidConfig:
    """Saturated two-phase properties at ``T_sat`` (SI units)."""

    name: str = MISSING
    # Saturation temperature at the operating pressure (atmospheric). A property
    # of the fluid, not a free operating input: the experiment derives its
    # ``T_sat_C`` from this, so selecting a fluid sets the saturation point.
    T_sat_C: float = MISSING  # deg C
    rho_l: float = MISSING  # kg/m^3
    rho_v: float = MISSING
    mu_l: float = MISSING  # Pa.s
    mu_v: float = MISSING
    k_l: float = MISSING  # W/m/K
    k_v: float = MISSING
    cp_l: float = MISSING  # J/kg/K
    cp_v: float = MISSING
    sigma: float = MISSING  # N/m
    h_lv: float = MISSING  # J/kg
    # kg/mol. The Schrage kinetic resistance closing the liquid film's
    # evaporation scales as sqrt(R_universal / molar_mass); without it the film
    # flux q = k dT / delta diverges as the film thins.
    molar_mass: float = MISSING
    # Wetting conditions (the nucleation root's closures). This is where the
    # offered fluids genuinely diverge -- their kinematic viscosities nearly
    # coincide while their contact-angle physics does not -- so every attachment
    # closure reads these, never an FC-72 literal in code. Angles in degrees.
    # Static equilibrium angle, and the hysteresis window it sits in. For the
    # perfectly wetting dielectrics the window is degrees-wide at most; for
    # water it spans tens of degrees and IS the closure (see wetting_source).
    theta_e_deg: float = MISSING
    theta_adv_deg: float = MISSING
    theta_rec_deg: float = MISSING
    # Evaporating-angle law theta_ev[deg] = coeff * dT^exp (dT in K). Null =
    # no law is measured for this fluid; the hysteresis window carries the
    # closure instead -- which is water's path, not a degraded default.
    theta_ev_coeff: Optional[float] = None  # noqa: UP045 -- OmegaConf needs Optional
    theta_ev_exp: Optional[float] = None  # noqa: UP045
    # Provenance of the values above: "measured" (a published fit for THIS
    # fluid), "partial" (measured in a neighbouring regime, anchor only),
    # "extrapolated" (wetting-class transfer from a sibling fluid), or
    # "hysteresis" (no law; the window is the closure). Recorded in every
    # run's diagnostics so a conclusion drawn on extrapolated constants says so.
    wetting_source: str = MISSING


@dataclass
class ImagingConfig:
    """Segmentation and calibration parameters for the raw TIFF frames."""

    # Row band searched for the two channel walls (gives the um/px calibration).
    wall_search_rows: list[int] = MISSING
    wall_margin_top: int = 3  # rows trimmed inside the upper wall
    wall_margin_bottom: int = 2  # rows trimmed inside the lower wall
    dark_thresh: int = MISSING  # intensity below which a pixel is "dark"
    open_kernel: int = MISSING  # morphological opening; removes heater traces
    close_kernel: int = MISSING  # morphological closing; seals the bubble ring
    # The imaged bubble edge is a thick dark rim, not a line; the interface is
    # its centreline. Below this interior-hole fraction the rim has no enclosed
    # interior to centre in, so the filled outline is used instead.
    min_rim_hole_fraction: float = 0.05
    # Pixel scale that seeds the active contour (blur on the medial field it
    # starts from) and smooths the solid-nucleus fallback. The interface's own
    # smoothness comes from the contour's rigidity, not this. 0 disables the blur.
    contour_smooth_px: float = 5.0
    # Arc length below which the traced interface's detail is faired away
    # (contour.fair_closed_contour), in pixels.
    fair_wavelength_px: float = 100.0
    # Average the traced bubble with its reflection about the channel centreline
    # (midway between the detected walls) before fairing, so the outline the
    # model fits is exactly mirror-symmetric, as the channel and its flow are.
    mirror_symmetric: bool = False
    # Columns masked out on the last usable frame, where the bubble leaves the
    # field of view (expressed in flipped, downstream-positive coordinates).
    truncated_cols: int = 0


@dataclass
class ScalesConfig:
    """Non-dimensionalisation. ``x*`` runs downstream with the inlet at zero."""

    L_ref_um: float = MISSING
    U_ref: float = MISSING  # m/s; measured nose-speed scale
    # The superheat scale dT_ref in kelvin. None: the initial superheat
    # T_init_C - T_sat_C when the series gives T_init_C, else the conduction scale
    # q''_w h / (2 k_l) the legacy thermal model uses.
    dT_ref_K: Optional[float] = None  # noqa: UP045 -- OmegaConf needs Optional


@dataclass
class FieldArch:
    """Per-field architecture override. Any field left ``None`` falls back to the
    global ``ModelConfig`` value, so a per-field entry stores only its deltas."""

    hidden: Optional[int] = None  # noqa: UP045 -- OmegaConf needs Optional, not X | None
    layers: Optional[int] = None  # noqa: UP045
    fourier_feats: Optional[int] = None  # noqa: UP045
    fourier_scale: Optional[float] = None  # noqa: UP045


@dataclass
class ModelConfig:
    """Fourier-feature MLP ensemble -- one network per physical field."""

    hidden: int = MISSING
    layers: int = MISSING
    fourier_feats: int = MISSING
    fourier_scale: float = MISSING
    # Interface half-thickness in alpha = sigmoid(phi / alpha_eps). Annealing
    # this downwards is the interface-sharpening curriculum.
    alpha_eps: float = MISSING
    nodewise_activation: bool = True  # per-neuron adaptive tanh slopes
    fields: list[str] = MISSING
    # Optional per-field architecture overrides, keyed by field name. Absent =
    # every field uses the global architecture above (Stage-A behaviour).
    per_field: dict[str, FieldArch] = field(default_factory=dict)
    # Localized nucleation heat pulse (Stage B; requires the `T` field). Off by default
    # -> the energy residual is unchanged. When on, a brief localized pulse at the fixed
    # cavity seeds the bubble (see residuals.NucleationPulse): its location (x_pin, from
    # the data) and width/timing are fixed priors; only its magnitude is learnable.
    nucleation_pulse: bool = False
    pulse_t0: float = 0.1  # pulse active for the first this-fraction of the time window
    pulse_width: float = 0.05  # streamwise Gaussian sigma, as a fraction of the domain width
    # Hard root pin: transform the level set so the interface (alpha = 0.5) passes
    # through the dataset's measured bubble-root anchor at EVERY time, including
    # times beyond the training window -- phi = tanh(dist/pin_d_ref) * N. Off by
    # default -> phi is the raw network output, byte-for-byte. The anchor is
    # derived from the data at load time, never configured by hand.
    hard_pin: bool = False
    pin_d_ref: float = 0.1  # saturation scale of the pin, in non-dim x* units
    # Front geometry (R3): replace the free level-set net with a geometric
    # construction -- monotone nose, bounded width envelope, bounded centerline --
    # so a single connected, root-anchored capsule is the ONLY expressible shape.
    # Fixes the extrapolation shape collapse structurally (the free level set kept
    # volume and the pinned point but rounded/fragmented the SHAPE off-data).
    # Mutually exclusive with hard_pin (the geometry pins exactly by construction).
    front_geometry: bool = False
    # Let the bubble detach. The front geometry guarantees one connected capsule
    # with a never-retreating nose -- which is what fixed R3's extrapolation shape
    # collapse, and is also exactly what makes pinch-off inexpressible. With this
    # on, the radius is signed (so a station can stop being vapour) and the nose
    # rate is unconstrained. Requires `front_geometry`; off by default, so both
    # guarantees stand unless a run asks for the trade.
    allow_pinch: bool = False
    # Evolving width: deform the profile along a LEARNED DIRECTION at a rate tied
    # to the bubble's own elongation, instead of letting it be a raw function of
    # time. A tanh net in t saturates past the trained range and drifts back to
    # its bias, so the shape RELAXES off-data -- measured, the neck deepens
    # +0.075/frame through the last supervised frame, rolls off to +0.013, then
    # REVERSES and un-necks at -0.028/frame. The bubble heals; the real one
    # detaches. This is the treatment R3 already gave the nose (s = s0 + integral
    # of a rate, which continues at its last learned rate); the width never got
    # it. Requires `front_geometry`; off by default.
    evolving_width: bool = False
    # Cap freedom: let the END CAPS have a shape. Without this the caps are exact
    # circles -- past the nose apex the spine parameter clamps, so the field is
    # `r_nose - |x - b|` at every step of training, and `_cap_frame` hands the
    # Young-Laplace residual `kappa = 1/r` as a GIVEN. The cap is the one region
    # the data cannot reach (the circle cannot comply) and the physics cannot
    # reshape (its curvature is asserted), which is the standing hypothesis for a
    # nose jump error stuck at 0.257 against a < 0.10 target. The circular-cap
    # justification holds only for a STATIC interface at uniform pressure; an
    # advancing bubble in a Hele-Shaw gap has a flattened nose, which is what the
    # Bretherton term already models transversely.
    # The radius gains a bounded angular modulation whose gate vanishes at the
    # apex (so the root pin and the monotone nose stay EXACT) and at the seam (so
    # the cap still meets the body, and the modulation is identically zero on the
    # body -- which is what keeps the alpha field and the explicit front samples
    # describing the same shape). Between them the cap's curvature, including at
    # the tip, is free. Requires `front_geometry`; off by default.
    cap_freedom: bool = False
    # How far the cap may depart from its circle, as a fraction of the local
    # radius. Bounded on purpose: unbounded, a cap can fold or self-intersect.
    cap_delta: float = 0.2
    # Pressure-driven shape: the width profile's low modes are SOLVED from the
    # Young-Laplace condition rather than learned and then penalised for
    # disagreeing with it. The soft penalty is measured -- raising
    # `training.weights.laplace` halves the jump error and costs ~0.015 IoU --
    # and this exists to find out whether DETERMINING the shape does better than
    # trading against it. Requires `sharp_interface` (there is no pressure jump to
    # solve without it). Off by default.
    pressure_shape: bool = False
    # How far the solve may move the width profile, as a fraction of the learned
    # width. The rail that stops a wrong pressure producing a nonsense bubble --
    # and the depth-averaged pressure is KNOWN to be wrong at the bubble's sides,
    # which is exactly where it now steers the geometry.
    shape_delta: float = 0.3
    # Sharp-interface physics (R4): drive the shape with the conditions that hold
    # AT a fluid interface -- the Young-Laplace jump and the kinematic condition,
    # sampled on the explicit front -- instead of a bulk momentum residual a free
    # pressure field can absorb up to its curl. Requires `front_geometry` (there
    # is no front to sample without it) and the `p` field. Off by default -> the
    # equation set and every existing run are unchanged.
    sharp_interface: bool = False
    # Depletable superheat: bound the temperature below by SATURATION rather than
    # by the inferred inlet value. Liquid that has just boiled a bubble is colder
    # than the inlet -- the latent heat came out of it -- and that depletion is
    # the feedback which shuts evaporation off as the bubble blankets the wall.
    # Measured with the bound in place, the field collapsed onto its floor and
    # stayed there (std 4e-07 across the whole domain at every frame), so
    # evaporation became a fixed multiple of interfacial area and could only
    # grow. `theta_in` then anchors the inlet instead of flooring the field.
    # Requires the 'T' field; off by default.
    depletable_superheat: bool = False
    # Film pressure: the depth-averaged model's stand-in for a pressure it cannot
    # carry. At the bubble's SIDES the capillary pressure is balanced by the
    # liquid in the Bretherton film between bubble and gap wall (~4.9 um in a
    # 150 um gap), not by the bulk -- but that film lies in the gap direction,
    # which depth-averaging integrates away, so one pressure per (x, y) cannot be
    # both. One inferred scalar offset, applied on the body only: measured, the
    # discrepancy is a near-constant 1.25-1.48 across every frame and across
    # independently trained runs. Requires `sharp_interface`; off by default.
    film_pressure: bool = False
    # The nucleation root: replace the ROOT cap's enforced circle with the
    # Hugelschaffer half-oval carrying ONE trained, time-varying, bounded
    # bluntness parameter w(t). The circle is the one region of the interface
    # measured beyond the model's reach: the masks disagree with it more and
    # more as the event runs (circle residual 12% -> 32%, taper drifting
    # blunter), the deviation is driven and time-varying (a static family
    # cannot follow it), and the attachment physics predicts exactly its sign
    # (a dry patch on the heated plate lowers the demanded curvature over the
    # root). w = 0 IS the circle, the apex crossing is exact for every w (the
    # pin survives structurally), the seams keep meeting the body, and the
    # nose cap stays circular -- freedom is spent only where the signal is.
    # Requires `front_geometry`; mutually exclusive with `cap_freedom` (two
    # shape systems on one cap). Off by default -> byte-identical.
    nucleation_root: bool = False
    # Bound on the Hugelschaffer bluntness |w| as a fraction of the root cap
    # radius. Must stay inside 1: at |w| = r the family's denominator reaches
    # zero at the apex and the curvature diverges. 0.6 spans apex radii from
    # 0.53 r (sharpened) to 8.5 r (blunted) -- well past the measured drift.
    root_delta: float = 0.6
    # The nose shape: replace the NOSE cap's enforced circle with the same
    # Hugelschaffer half-oval that freed the root, mirrored, carrying its OWN
    # trained, time-varying, bounded bluntness w_nose(t). The measured head is
    # non-circular and only late (nose-window bluntness 0.61 -> 1.83 through
    # the event, CROSSING 1 -- an overhanging, mushroomed head no circle can
    # draw), the model's nose error appears exactly there (RMS 0.02 -> 0.19),
    # and the trained pressure already demands a non-circular nose (curvature
    # gap 0.94 at the nose cap vs 0.23 on the body) -- a physics drive the
    # root never had. The family already expresses the overhang: for w > 0
    # the width's maximum sits interior to the seam. w = 0 IS the circle, the
    # apex crossing is exact for every w (the monotone nose survives
    # structurally, so fv_apex composes unchanged), and the seams keep
    # meeting the body. Requires `front_geometry`; COMPOSABLE with
    # `nucleation_root` (different caps); mutually exclusive with
    # `cap_freedom` (the same cap). Off by default -> byte-identical.
    # BENCHED NEGATIVE (T6, 12 runs, 2026-08): the mechanism is alive (71-100%
    # budget in every arm) but the depth-averaged jump demands MORE curvature
    # at the nose (gap +0.94) and drives the freed cap to SHARPEN (w -> -delta)
    # -- the physics-only arm proves it without any supervision -- while the
    # measured head wants the opposite (the overhang). The fight degrades the
    # front in every arm tried (full / halved / sdf-only supervision), so the
    # flag stays off until the nose's jump physics is corrected; freeing the
    # shape again before that re-runs the same collision.
    nose_shape: bool = False
    # Bound on the Hugelschaffer nose bluntness |w_nose| as a fraction of the
    # nose cap radius; the same rationale (and value) as root_delta.
    nose_delta: float = 0.6
    # The attachment pressure (nucleation-root R2): over the dry contact patch
    # on the heated plate the gap-direction capillary term is NOT the
    # both-plates-wetted 2/H* -- contact-angle cosines ADD across the gap
    # (Park & Homsy 1984 for the wetted limit; Lu, Glasner, Bertozzi & Kim
    # 2007 for the contacted one), so the attached side carries
    # (1 + cos theta_app)/H*: LOWER capillary pressure, demanding a BLUNTER
    # root -- the measured sign. theta_app is ONE trained scalar whose init
    # and bounds come from the per-fluid wetting config; the patch footprint
    # a(x,t) is a trained monotone extent about the anchor (the nose-rate
    # treatment: it never shrinks while the bubble grows). Requires
    # `nucleation_root` (the DOF the pressure drives) and `sharp_interface`
    # (the jump the blend enters). Off by default -> the jump is unchanged.
    root_attachment: bool = False
    # Receding Bretherton branch (nucleation-root R0): a receding meniscus's cap
    # is FLATTER than static -- kappa R = 1 + beta (3Ca)^{2/3} with beta_rear =
    # -1.13 -- and without this flag every receding section is clamped TO the
    # static 2/H*, an O(10%) curvature overstatement on exactly the receding
    # root body (the trained capsule recedes at essentially every station; only
    # the nose advances). One term on an existing function, benchable alone
    # (the pi/4-fix precedent). Requires `sharp_interface` (the jump condition
    # is what consumes the gap curvature in training); off by default -> the
    # advancing-only correction is byte-for-byte what it was.
    receding_cap: bool = False
    # Liquid film: the thin layer the bubble leaves against the gap walls, as a
    # 1-D field delta(x, t) evaluated on the explicit front. Deposition follows the
    # Aussillous-Quere saturating law on the LOCAL normal speed -- deliberately
    # NOT the boundary-layer sqrt(nu t), which returns nearly the same value for
    # all four working fluids (their kinematic viscosities almost coincide) and
    # would build in exactly the fluid-blindness the film exists to remove.
    # Requires `sharp_interface` (the film is scored on the front the trainer
    # only samples in sharp mode, and its later stages feed the jump condition).
    # Off by default -> no film net, no film term, byte-identical.
    liquid_film: bool = False
    # Quadrature stations for the film's depletion ODE, spread between the
    # measured root and the nose at each front time. Deterministic midpoints,
    # not a random draw: the film is one smooth 1-D profile per time, so what
    # matters is covering the footprint, and a fixed grid is reproducible
    # across resume.
    film_stations: int = 32
    # Front samples per time, per body profile and per end cap, for the interface
    # conditions. Module-level rather than swept: the front is a smooth 1-D curve,
    # so this only has to resolve it, not fit anything.
    front_body_samples: int = 64
    front_cap_samples: int = 16
    # Times per step the front is sampled at, spread over the training window.
    front_times: int = 16
    # --- Domain and boundary choices (trustworthy-fields WP2). Every default keeps
    # existing runs byte-for-byte; configs/model/trusted.yaml switches them on. ---
    # Where the side walls are. "crop": the imaged ROI's edges, which sit a few
    # pixels INSIDE the channel walls (today). "physical": the walls preprocessing
    # detected (Domain.y_wall_lo/hi), so the wall samples, the inlet span, the
    # uniform collocation and the capsule's width bound reach the channel.
    walls_at: str = "crop"  # "crop" | "physical"
    # The side-wall velocity condition. A first-order Darcy law admits only
    # no-penetration (v = 0): adding no-slip gives zero Cauchy data on the wall,
    # which forces a harmonic pressure to be constant everywhere; the published
    # runs lost their inlet flux within one channel width. No-slip is admissible
    # with the Brinkman closure, whose in-plane viscous term carries the O(h)
    # wall layer (Brinkman 1949; Zeng, Yortsos & Salin 2003).
    wall_condition: str = "no_slip"  # "no_slip" | "no_penetration"
    # The depth-averaged momentum law in sharp mode: Darcy, or Darcy plus the
    # Brinkman viscous term (1/Re) mu~ lap(u).
    momentum_closure: str = "darcy"  # "darcy" | "brinkman"
    # Depth-averaged inertia in the Brinkman law, rho~ (u_t + 6/5 u.grad u),
    # making it the depth-averaged Navier-Stokes of a Hele-Shaw cell (Gondret &
    # Rabaud 1997). Without it the law balances pressure against friction alone,
    # valid only when Re (h/L)^2 / 12 << 1; on Series-1-w300 that is about 4.5,
    # and measured at the advancing head the omitted inertia matched the drag
    # and reached the in-plane viscous term in the side gaps. Brinkman only.
    momentum_inertia: bool = False
    # The kinematic condition imposed exactly instead of as a penalty: the
    # liquid's velocity across the interface equals the interface's own normal
    # speed on the whole of it, at every instant, by construction. The velocity
    # network is corrected along the interface normal by a factor that is one on
    # the interface and fades over kinematic_blend_length (L_ref units) on both
    # sides; the tangential velocity is left free (a gas interface is
    # shear-free). Normal and speed come from the geometry with its parameters
    # detached, so the correction never moves the outline. Held off through the
    # warm-up (training.stage_b_warmup_steps) and on from the first physics
    # step, so the warm-up stays the outline fit. Needs the front geometry.
    # The evaporation term m/rho_l (about 0.1 mm/s on Series-1-w300, under 0.1%
    # of the head's speed) is not included.
    hard_kinematics: bool = False
    kinematic_blend_length: float = 0.04
    # Which side of the interface the exact condition corrects. "liquid": the
    # liquid only, the vapour's velocity left to its own equations (inside, the
    # vapour leaves the interface at V_n - m/rho_v, not V_n). "both": the blend
    # on both sides, as the first exact-condition run (trust-hk4-s0) had it,
    # whose continuity failed most just inside the advancing head.
    kinematic_side: str = "both"  # "both" | "liquid"
    # The outlet at x_max: nothing (today), or zero streamwise velocity gradient.
    # The channel continues beyond the field of view and its pressure there is
    # unknown, so no pressure condition is imposed.
    outlet_condition: str = "none"  # "none" | "zero_gradient"
    # The outlet's pressure, Pa above atmospheric (gauge), imposed on the liquid
    # pressure across the outlet. With the zero-gradient velocity it makes the
    # outflow stress-free at a known pressure, and fixes the pressure's level,
    # which the vapour pressure otherwise carries alone. None imposes nothing.
    outlet_pressure_Pa: Optional[float] = None  # noqa: UP045
    # The inlet velocity across the channel's width. "plug": the mean everywhere,
    # which disagrees with no-slip side walls at the inlet corners. "developed":
    # the fully developed profile of the depth-averaged Brinkman law, flat in the
    # middle and falling to zero at the walls over h / sqrt(12), with the same
    # mean (the flow rate).
    inlet_profile: str = "plug"  # "plug" | "developed"
    # How the pressure field is represented (trustworthy-fields WP3). "single":
    # one network for both phases (today), which has to fake the Young-Laplace
    # discontinuity with steep gradients the momentum residual then punishes.
    # "two_phase": the uniform vapour pressure p_v(t) inside the bubble and the
    # liquid network outside, blended by the (detached) indicator, so the jump
    # is built in: the Laplace condition reads the liquid branch on the front,
    # the momentum law is scored in the liquid, and the vapour velocity is made
    # curl-free (the Darcy law with a uniform vapour pressure). Requires
    # `sharp_interface` (p_v exists only there).
    pressure_representation: str = "single"  # "single" | "two_phase"
    # --- The thermal model (trustworthy-fields WP4). ---
    # "legacy": the diffuse energy equation with the |grad alpha| latent sink, the
    # kinetic (Schrage-type) closure through a learned resistance, and the inlet
    # floor on theta (today). "sharp": the liquid-only energy equation with the
    # interface AT saturation, inlet, initial and side-wall conditions from the
    # experiment, and the evaporation flux taken from the liquid-side temperature
    # gradient on the explicit front, so one mass flux drives both the latent
    # heat and the vapour volume. Requires the front geometry, the sharp
    # interface and the T field.
    thermal_model: str = "legacy"  # "legacy" | "sharp"
    # The interface temperature under the sharp model: saturation (conduction-
    # limited evaporation, no fitted resistance). The kinetic route lives in the
    # legacy model for now.
    interface_temperature: str = "saturation"  # "saturation" | "kinetic"
    # How theta is built under the sharp model: "gated" multiplies the T network
    # by the (smoothed) distance into the liquid, so theta is exactly zero in
    # the vapour and on the front and may go below saturation in the liquid.
    theta_parameterization: str = "gated"  # "gated" | "free"
    # Smoothing width of that distance gate, in L_ref units: a numerical choice,
    # kept well below the thermal layer it must not blur (~0.02-0.04 here).
    theta_gate_width: float = 0.005
    # The shape of that gate. "linear": theta = d N, the distance itself, so far
    # from the bubble theta is the network times a distance that SHRINKS as the
    # front approaches; unless the network compensates exactly, the liquid
    # ahead of an advancing nose cools for no physical reason (measured on the
    # trusted recipe: 21 to 69 percent of the downstream temperature's fall was
    # this geometric decay, and the liquid that should stay at 58 C drifted to
    # saturation). "saturating": theta = tanh(d / theta_gate_length) N, still
    # zero on the front, but equal to the network beyond a few gate lengths, so
    # the far field no longer depends on where the bubble is.
    theta_gate: str = "linear"  # "linear" | "saturating"
    # The saturating gate's length in L_ref units: the thermal layer's scale
    # sqrt(t*/Pe), about 0.02 to 0.04 here; the network still shapes the layer.
    theta_gate_length: float = 0.04
    # Where the continuity source comes from. "free": the s network, pulled toward
    # the thermal flux two-way by the evap_flux closure. "thermal": s IS the
    # interface flux spread over the band, no s network in the loop.
    source_mode: str = "free"  # "free" | "thermal"
    # Width of the Gaussian that spreads the front flux into the source band, in
    # L_ref units. None = alpha_eps, the interface half-thickness.
    source_kernel_width: Optional[float] = None  # noqa: UP045
    # Film evaporation under the bubble, the growth mechanism of confined flow
    # boiling: the series' wall flux under the footprint evaporates the liquid
    # film at the wet fraction and creates vapour there, a volumetric source
    # footprint_source_star * f_wet * alpha added to the thermal source. The same
    # flux heats the liquid where there is liquid (the energy equation's
    # (1 - alpha) gate), so one wall flux is split by phase. Measured on
    # Series-1, the wall flux under the footprint alone pays for the growth from
    # about 1.5 ms on; the first full-chain benchmark, which let only the
    # outline's conduction flux create vapour, found the no-evaporation solution.
    # Requires the sharp thermal model. Off by default.
    footprint_evaporation: bool = False
    # Mirror symmetry about the channel centreline (trustworthy-fields WP5):
    # every field network is averaged with its own reflection (u, p, T, s and
    # the interface even, v odd) and the capsule's centreline is pinned to the
    # axis, so the symmetry a symmetric channel implies holds exactly. A safety
    # net for AFTER the physics passes its tests: switched on earlier it would
    # hide the non-identifiability the audit exists to expose. Requires the
    # front geometry (the axis is the run's wall span, which it carries).
    mirror_symmetry: bool = False


@dataclass
class LossWeights:
    """Initial multipliers; ``vof``/``div``/``bc`` (and Stage-B ``mom``/``energy``)
    are rebalanced during training. Stage-B terms default to 1.0 so Stage-A
    configs compose unchanged -- they are inert while their equations are off."""

    data: float = MISSING
    vof: float = MISSING
    div: float = MISSING
    src: float = MISSING
    bc: float = MISSING
    mom: float = 1.0  # Stage B: momentum
    energy: float = 1.0  # Stage B: energy
    evap: float = 1.0  # Stage B: evaporation mass-closure penalty
    darcy: float = 1.0  # R4: depth-averaged momentum (replaces `mom` in sharp mode)
    brinkman: float = 1.0  # WP2: Darcy + Brinkman momentum (replaces `darcy` when chosen)
    vapour_potential: float = 1.0  # WP3: curl-free vapour velocity under the two-phase pressure
    energy_liquid: float = 1.0  # WP4: liquid energy equation of the sharp thermal model
    theta_bc: float = 1.0  # WP4: inlet, initial and side-wall temperature conditions
    evap_flux: float = 1.0  # WP4: free source pulled to the interface flux, two-way
    kinematic: float = 1.0  # R4: kinematic condition on the explicit front
    laplace: float = 1.0  # R4: Young-Laplace jump across the explicit front
    film: float = 1.0  # liquid film: deposition where the meniscus advances
    film_depletion: float = 1.0  # liquid film: evaporative thinning behind the front
    film_source: float = 1.0  # liquid film: its evaporation as the mass source


@dataclass
class TrainingConfig:
    steps: int = MISSING
    lr: float = MISSING
    lr_halflife: int = MISSING  # steps between learning-rate halvings

    n_data: int = MISSING  # supervised points per step
    n_coll: int = MISSING  # PDE collocation points per step
    n_bc: int = MISSING  # boundary points per step
    # Collapse each collocation batch's TIMES onto this many stratified instants
    # instead of one per point. A shape rebuilt from a field costs once per
    # distinct time, so an unbinned batch makes that construction unaffordable.
    # Stratified, so coverage of the window is kept and the Monte-Carlo estimate's
    # variance falls rather than its bias rising. 0 = the independent per-point
    # draw, unchanged.
    collocation_time_bins: int = 0

    # Frame withheld from supervision entirely (0-based) and used as the
    # honest generalisation test. Set to -1 to train on every frame. This is the
    # legacy single-frame knob; ``val_fraction`` below is the general axis and the
    # two compose (both frames are held out).
    holdout_frame: int = MISSING

    # Validation axis A: a per-dataset *fraction* of event frames held out of
    # supervision as an in-distribution validation set. 0.0 (the default) keeps
    # every existing run byte-for-byte unchanged; the Solver dials this to 0.2 for
    # new joint runs. ``val_strategy`` picks which frames: "tail" holds the last
    # frames (extrapolation, the honest test for a growth series); "scatter" holds
    # evenly spaced frames (interpolation, easier). Held-out frames are the union
    # of this split, the ``holdout_frame``, and (already-dropped) excluded frames.
    # ``val_strategy`` is a plain ``str`` (validated at use in ``BubbleDataset``),
    # not a ``Literal["tail", "scatter"]``: OmegaConf rejects ``Literal`` annotations
    # in structured configs, so ``str`` + a runtime check is the supported pattern.
    # NB the *aggregate* val IoU is surfaced only for joint runs (``evaluate_joint``);
    # a single-dataset CLI run honours ``val_fraction`` in training but ``evaluate``
    # does not report a separate ``iou_val`` for it.
    # Let the evaporation mass closure push the DRIVE up to the source, not only
    # pull the source down to the drive. The shipped closure detaches its whole
    # target, so it is one-way and structurally cannot raise the temperature at
    # any weight -- measured, at weight 10 it closed an 82x gap by shrinking the
    # source 6x while theta did not move. That matters because the energy
    # equation is too weakly forced to determine theta (see `q_wall_star`), and
    # the source is the only evaporation quantity the data constrains, through
    # the bubble's growth. Only `theta` is freed; the interfacial area stays
    # detached, so the closure still cannot flatten the interface to cheat.
    evap_closure_two_way: bool = False
    # Interface sharpening: anneal `model.alpha_eps` geometrically down to
    # `alpha_eps_final` over this many steps (0 = off, the interface thickness
    # holds where it is configured). alpha = sigmoid(phi / alpha_eps) spreads the
    # transition over roughly 4*alpha_eps, so at the Stage-A default of 0.05 the
    # blur is ~0.2 in y* -- the same size as the measured mid-bubble neck. A
    # feature cannot be resolved through a blur its own width.
    alpha_eps_anneal_steps: int = 0
    alpha_eps_final: float = 0.0
    val_fraction: float = 0.0
    val_strategy: str = "tail"  # "tail" (extrapolation) | "scatter" (interpolation)

    weights: LossWeights = MISSING
    rebalance_every: int = MISSING  # gradient-norm loss rebalancing period
    # Stage B: steps over which the evaporation mass-closure ramps in while the
    # off-interface source penalty decays (soft -> hard curriculum). 0 disables
    # the schedule (the penalty and closure keep their static weights).
    curriculum_steps: int = 0
    # Stage B: steps to train the Stage-A objective alone before the Stage-B
    # physics (momentum, energy, evaporation) engages -- an in-run warm start, so
    # those terms act on an interface that has already converged instead of on a
    # random field (which collapses it). 0 disables the gate (Stage-B on from
    # step 1); a resumed run whose step count is past the gate is unaffected.
    stage_b_warmup_steps: int = 0
    # Write the checkpoint every this many steps as well as at the end (0 = at
    # the end only), so a run cut short on a rented machine resumes from its
    # last save instead of from nothing. A resume reads the saved step count.
    checkpoint_every: int = 0
    # Clip the total gradient's norm to this before each optimiser step (0 = no
    # clipping). The objective is untouched; only a spike's step is bounded. The
    # full-chain benchmark lost one seed in three to one such spike: the
    # rebalancer had raised the energy term's weight to 80 (3.5 on the seeds that
    # held), and the next 500 steps took every residual up four orders.
    grad_clip: float = 0.0
    # Cap on every rebalanced weight as a multiple of the data weight (0 = the
    # fixed 1e3 ceiling alone). The gradient-norm rebalancer ratchets a term
    # whose gradient is small up to the ceiling; the first full-chain benchmark
    # ended with Brinkman at 1000 and energy at 253 against data at 10, and the
    # images lost the vote: the outline drifted off the masks while every
    # residual improved. The images are the measurement, so no physics term may
    # outweigh them by more than this.
    rebalance_max_ratio: float = 0.0
    # The outline guard: every this many steps after the warm-up the per-frame
    # IoU against the masks is measured (0 = off), and a run whose mean IoU has
    # fallen more than outline_guard_drop below the warm-up's own fit is stopped
    # there, checkpoint written, instead of finishing hours later as a failure.
    # The warm-up's fit is recorded at the step the physics engages.
    outline_guard_every: int = 0
    outline_guard_drop: float = 0.05
    # Draw the front times and the mass-balance instants afresh every step
    # (stratified: one instant per equal stratum of the window) instead of
    # holding them on fixed grids. Measured on the first 12000-step seed of the
    # trusted recipe: the flux defect was 0.1 at the eight fixed mass-balance
    # instants and 1.5 to 2.5 everywhere between them, and the Young-Laplace
    # residual 0.1 to 0.3 at the sixteen fixed front times and 0.5 to 1.0
    # between them. The fields had learned the quadrature, not the physics.
    # Off keeps the fixed grids (existing runs byte-for-byte).
    quadrature_jitter: bool = False
    # Two-stage training: at the end of the warm-up, the step the physics
    # engages, the bubble's geometry is frozen at the warm-up's fit of the
    # images, and the physics stage trains only the hidden fields (u, v, p, T,
    # the vapour pressure, the wet fraction) against the equations. The
    # measured outline becomes an input to the flow problem, not an objective
    # the flow competes with: the kinematic and Young-Laplace conditions read
    # the frozen front as a moving boundary. With nothing left for the physics
    # to pull off the masks, the rebalancer equalises the physics terms among
    # themselves (the median of their weighted gradient norms is the
    # reference) instead of holding them to the data term, which the measured
    # 12000-step seeds showed keeping the momentum weight below 1 against the
    # data's 10. The terms that act on the geometry alone (the measured front
    # velocity, the root and nose supervision) are not evaluated once it is
    # frozen. Needs a warm-up (stage_b_warmup_steps > 0) and the front geometry.
    freeze_geometry: bool = False
    # How the physics terms are weighted once the geometry is frozen. "fixed":
    # each at its configured weight, never rebalanced. "equalised": rebalanced
    # to equal gradient contributions (the median of the weighted norms is the
    # reference). The first two-stage seed ran equalised and turned the
    # momentum weight down to 0.26 (its gradient is the largest), and its
    # Brinkman residual doubled against the coupled run's.
    frozen_weighting: str = "fixed"  # "fixed" | "equalised"
    # The temperature equation trains the temperature only: the velocity in its
    # advection term is detached, so the equation's error cannot bend the flow
    # (see residuals.energy_liquid_residual). Off keeps the two-way gradient.
    one_way_advection: bool = False
    # Sequential temperature solve: after this step the outline and the flow are
    # held where they are and only the temperature net trains, on the energy
    # equation and the thermal start and inlet conditions alone, from a fresh
    # optimiser whose learning-rate clock restarts at this step. Zero (default)
    # keeps the coupled field stage to the end. Needs freeze_geometry and a step
    # past the warm-up.
    temperature_stage_from: int = 0
    # Reuse the warm-up: a fresh run starts from the checkpoint at the end of
    # stage_b_warmup_steps that an earlier run with the same warm-up inputs
    # cached, or trains and caches it (naviernet.warmup_cache). The key is the
    # processed archive's content, the config minus what acts only after the
    # warm-up, and the code the warm-up runs.
    warmup_cache: bool = False
    # The learning rate's halving period in the physics stage, in steps: the
    # schedule's clock runs lr_halflife / physics_lr_halflife times faster after
    # the warm-up, so a physics stage of fewer, larger steps (more points per
    # step) sweeps the same rates as the full-length one. Zero keeps one clock.
    physics_lr_halflife: int = 0

    # --- Accuracy techniques (opt-in; every default is a no-op, so an unchanged
    # config trains byte-for-byte as before). See
    # .claude/plans/pinn-accuracy-causality-adaptive-weighting-plan.md ---
    # Problem 1 -- temporal causal weighting (arXiv:2203.07404): weight the PDE
    # residual by time so the model learns causally (early times first), which
    # fixes the late-frame extrapolation bias. False -> uniform-in-time (today).
    causal_weighting: bool = False
    causal_time_chunks: int = 16  # number of ordered time bins the residual is split into
    # ε annealing schedule -- steeper causal enforcement as training proceeds.
    causal_eps_schedule: list[float] = field(
        default_factory=lambda: [1e-2, 1e-1, 1.0, 10.0, 100.0]
    )
    # Causal variant. "weight": soft per-time exp weights (uses causal_eps_schedule).
    # "march": a time-marching curriculum -- enforce the collocation residual only up
    # to a horizon that expands from the earliest bin to the whole domain, so late
    # times are trained at FULL weight once the front reaches them (causal_eps_schedule
    # is unused in this mode). See arXiv:2203.07404 §time-marching.
    causal_mode: str = "weight"  # "weight" | "march"
    # Fraction of the run over which the marching horizon expands to the full time
    # domain; it stays at the full domain afterwards (so late frames get real
    # full-weight training). Only used when causal_mode="march".
    causal_march_full_frac: float = 0.5
    # Convergence gate on the eps schedule (the rule in arXiv:2203.07404 that the
    # fixed per-step annealing above lacks): eps advances to the next schedule value
    # only once EVERY time chunk's causal weight exceeds this threshold at the
    # current eps. None keeps the fixed annealing. Measured without the gate on the
    # representative run: Stage-B chunk losses of O(1-30) drove every weight but
    # the first chunk's to exactly zero, so the PDE residuals received no gradient
    # after ~0.5 ms of the event for the whole physics stage.
    causal_gate_min_weight: Optional[float] = None  # noqa: UP045 -- OmegaConf needs Optional
    # Per-time-chunk residual diagnostics: at every logged step, each collocation
    # term's plain mean over each time chunk is appended to the run's record, so a
    # residual left unenforced at late times is visible in the run's own history
    # (and in metrics.json). Off keeps the history byte-for-byte as it was.
    log_chunk_residuals: bool = False
    # Problem 2 -- loss-weighting scheme. "gradnorm" is the gradient-norm rebalancer
    # (unbounded; ratchets to the 1e3 cap on long runs); "rba" is bounded residual-
    # based attention (arXiv:2307.00379). Default keeps existing behaviour until
    # Phase 2 makes "rba" the default.
    weighting: str = "gradnorm"  # "gradnorm" | "rba"
    rba_gamma: float = 0.999  # RBA EMA decay on per-point attention
    rba_eta: float = 0.01  # RBA update rate
    # Problem 3 -- residual-adaptive collocation (RAR): periodically resample
    # collocation points toward high-residual regions (interface, late times).
    # False -> uniform sampling (today).
    adaptive_collocation: bool = False
    resample_every: int = 500  # steps between collocation resamples
    resample_fraction: float = 0.5  # fraction of the batch drawn from high-residual regions

    # Kinematic growth constraints (R2): physics-only scalar functionals on a fixed
    # quadrature grid over the LATE time window, targeting the diagnosed
    # extrapolation failure (the dilatation source collapses at the supervision
    # boundary and predicted dV/dt goes negative while the true bubble keeps
    # growing). Off by default -> the objective is unchanged. The terms sit outside
    # causal weighting, RBA, and the rebalancer (they are functionals, not
    # pointwise residuals), normalized by the dataset's supervised-tail growth rate.
    kinematics: bool = False
    kin_weight_mono: float = 1.0  # monotone volume: ReLU(margin - dQ/dt)^2
    kin_weight_balance: float = 1.0  # volume-vs-source balance: (dQ/dt - ∫α·s)^2
    kin_weight_evap: float = 1.0  # evap floor: ReLU(floor·E - ∫α·s)^2 (needs T)
    kin_margin_frac: float = 0.3  # mono margin, as a fraction of the reference rate
    kin_evap_floor: float = 0.5  # require the source to carry ≥ this × the evap drive
    kin_time_frac: float = 0.4  # quadrature window = last this-fraction of [t_min, t_max]
    kin_times: int = 12  # fixed time slices in the window
    kin_grid: int = 48  # fixed spatial quadrature grid, per side
    kin_ramp: int = 300  # steps to ramp balance/evap in (the source field is noise early)

    # Measured front velocity: supervise the front's own normal speed against the
    # velocity measured from consecutive masks, and the nose apex against its
    # measured 2-D displacement. The shape is already supervised at every training
    # frame; its RATE is not -- and under the front-geometry parameterisation the
    # rate is exactly what carries the profile past the last supervised frame.
    # Requires `model.front_geometry` (there is no explicit front, and no
    # `normal_speed`, without it). Off by default -> the objective is unchanged.
    front_velocity: bool = False
    # 10, not 1. These terms sit outside the gradient-norm rebalancer by design
    # (they are functionals, not pointwise residuals), so nothing corrects a
    # weight that is too small to act -- and at 1.0 their combined gradient norm
    # measures 0.094 against the data term's 1.66, which is 1/18th. Benched, a
    # run at weight 1 scored 0.871/0.929 against a baseline 0.872/0.928: the same
    # run. A flag whose default makes it inert is a switch that does nothing.
    fv_weight: float = 10.0  # binned normal-velocity profile along the front
    fv_apex_weight: float = 10.0  # nose-apex 2-D displacement (a true (vx, vy))
    # Front samples averaged into each profile bin -- the knob is stated this way
    # round deliberately. Averaging within a bin is the ONLY thing keeping
    # single-pixel mask noise out of the learned rate, and a bin count says
    # nothing about how much averaging it buys: at 16 bins the default 16 cap
    # samples land one per bin, which is no averaging at all. Each segment gets
    # its own bin count derived from this, so body and caps are both averaged.
    fv_samples_per_bin: int = 4
    # Gaussian blur (in pixels) on the signed distance fields before differencing.
    # A binary mask quantises the interface onto the pixel grid and its distance
    # field inherits that staircase; both frames are blurred identically, so the
    # interface is not shifted.
    fv_smooth_px: float = 1.5

    # Root-window supervision (nucleation-root R1): drive the Hugelschaffer
    # bluntness w(t) with the measured mask's signed-distance transform sampled
    # AT the predicted root-cap contour, plus the per-frame scalar bluntness
    # target. Distance- and ratio-based, never pixel-only: the root's mismatch
    # band is thinner than the interface blur and cap pixels are a sliver of
    # the batch, which is how cap_freedom starved twice (the undriven-knob
    # trap). Requires `model.nucleation_root` (nothing else reads the signal).
    # Off by default -> the objective is unchanged.
    root_supervision: bool = False
    # Like fv_weight, these sit outside the rebalancer, so nothing corrects a
    # weight too small to act -- a flag whose default makes it inert is a
    # switch that does nothing (the front-velocity lesson). 10 matches the
    # fv precedent for a front functional; the bluntness term's differences
    # are O(0.1) so its square needs the larger factor to register. The R0/R1
    # probe's mechanism gate (w(t) budget use) is what validates these.
    root_sdf_weight: float = 10.0
    root_bluntness_weight: float = 25.0
    # The root sampling budget: cap samples per frame for the supervision's own
    # front call -- denser than the physics' front_cap_samples, because these
    # samples are the only place the root signal enters.
    root_cap_samples: int = 48

    # Nose-window supervision: drive the Hugelschaffer nose bluntness
    # w_nose(t) with the mask's signed-distance transform sampled AT the
    # predicted nose-cap contour, plus the per-frame scalar bluntness target
    # (which legitimately EXCEEDS 1 late -- the measured overhang). The exact
    # mirror of root_supervision, for the exact reason: this nose is where
    # cap_freedom starved twice on pixel losses (the undriven-knob trap), and
    # the root's distance+ratio channel measurably fixed the same failure.
    # Requires `model.nose_shape` (nothing else reads the signal). Off by
    # default -> the objective is unchanged.
    nose_supervision: bool = False
    # The root's proven weights, carried over: these sit outside the
    # rebalancer, so nothing corrects a weight too small to act; the bench's
    # mechanism gate (w_nose budget use) is what validates them.
    nose_sdf_weight: float = 10.0
    nose_bluntness_weight: float = 25.0
    # The nose sampling budget, mirroring root_cap_samples.
    nose_cap_samples: int = 48

    # Cross-section mass balance (trustworthy-fields WP2): on a fixed quadrature,
    # Q(x,t) - Q_in - int_0^x int s dy dx' = 0 at every station and time, so the
    # axial volume flux is the inlet flux plus the vapour created upstream. The
    # local residual div u = s says this only where it is sampled; the published
    # runs lost the whole inlet flux within one channel width while their local
    # residual looked moderate. Normalised by Q_in; outside causal weighting, RBA
    # and the rebalancer like the other functionals. Off by default.
    mass_balance: bool = False
    mass_balance_weight: float = 1.0
    mass_balance_stations: int = 32  # x-stations of the cumulative source integral
    mass_balance_times: int = 8  # fixed time slices across the window
    # Growth closure: at every event frame the vapour the source makes,
    # int s dA, equals the measured growth of the bubble's area. The outline
    # flux's part is detached, so the term trains the footprint's wetting and
    # never the temperature. Needs the thermal source with footprint
    # evaporation. Zero weight (default) leaves it off; outside causal
    # weighting, RBA and the rebalancer like the mass balance.
    growth_closure_weight: float = 0.0
    growth_closure_stride: int = 4  # every n-th pixel of the covered floor
    # Let the closure's outline part (the flux the temperature carries) keep its
    # gradient, so the measured growth also trains the temperature. Off: it is
    # read detached, and the closure trains the footprint's wetting alone.
    growth_closure_outline_gradient: bool = False
    mass_balance_grid: int = 64  # transverse quadrature points, wall to wall

    seed: int = 0
    device: str = "cpu"
    log_every: int = 200


@dataclass
class AcceptanceConfig:
    """Thresholds a hidden field must meet before it is trusted (WP6): seed
    agreement, mirror symmetry, residuals on the audit grid, the cross-section
    flux balance, the heat budget, and the outline against the masks. Each is a
    fraction, a correlation, or a length on L_ref."""

    seed_corr_min: float = 0.95  # min cross-seed correlation of each field
    mirror_max: float = 0.05  # max symmetry-breaking fraction of a hidden field
    residual_rms_max: float = 0.05  # max RMS of any collocation residual
    flux_tol: float = 0.05  # max |Q - Q_in - created upstream| / Q_in at any station
    heat_budget_tol: float = 0.30  # max spread between the latent-power estimates
    # The outline: the per-frame IoU against the masks may fall neither below
    # this floor nor more than iou_drop_max below the warm-up's own fit of that
    # frame, and the nose may sit no farther than nose_tol (L_ref units) from
    # the measured nose at any frame. A run that fails here is rejected whatever
    # its residuals say: the images are the measurement.
    iou_min: float = 0.85
    iou_drop_max: float = 0.05
    nose_tol: float = 0.1
    # Where the frozen (or fitted) shape blocks the physics: the Young-Laplace
    # mismatch on the front, expressed as the shape change it would take to
    # remove it, in camera pixels. The 90th percentile over the front and the
    # event may not exceed this: within it the measured outline and the
    # physics agree to the measurement's own resolution.
    jump_px_max: float = 2.0


@dataclass
class EvaluationConfig:
    stride: int = 4  # pixel stride for gridded prediction
    n_traj_points: int = 61  # timesteps on the continuous nose trajectory
    threshold: float = 0.5  # alpha level defining the interface
    # The acceptance audit written beside metrics.json by `evaluate` (WP6): every
    # event frame on an audit_nx x audit_ny grid centred on the channel.
    audit: bool = True
    audit_nx: int = 129
    audit_ny: int = 49
    # Instants scanned across the whole event window (not only the camera
    # frames) for the residuals, the flux balance and the jump condition, on a
    # grid half the audit's in each direction. A field can satisfy the physics
    # at the instants it was trained on and violate it between them; this is
    # what sees it. 0 skips the scan.
    audit_scan_times: int = 61
    acceptance: AcceptanceConfig = field(default_factory=AcceptanceConfig)


@dataclass
class VideoConfig:
    n_timesteps: int = 100
    fps: int = 22
    width: int = 1600
    hold_frames: int = 14  # duplicated tail frames so the video ends on a beat
    background_dark_thresh: int = 110  # for the inpainted background plate


@dataclass
class PathsConfig:
    """All paths derive from ``root`` plus the dataset and run names.

    ``root`` stays relative by default and Hydra is configured with
    ``job.chdir=false``, so the working directory never changes and commands
    behave identically whether run via the CLI or from a notebook.
    """

    root: str = "."
    raw_dir: str = "${paths.root}/data/raw/${dataset}"
    processed_dir: str = "${paths.root}/data/processed/${dataset}"
    output_dir: str = "${paths.root}/outputs/${run_name}"


@dataclass
class Config:
    """Root configuration object handed to every stage."""

    # Which raw dataset to read, and the name of the output directory. Keeping
    # them separate lets several runs (architectures, seeds, weightings) share
    # one preprocessed dataset without recomputing or overwriting it.
    dataset: str = "highest_t"
    run_name: str = "${dataset}"

    # Joint (transfer-learning) training reads several datasets at once. Empty
    # means "just `dataset`", so every single-dataset run is the 1-element case
    # and existing configs are unchanged. Use `resolved_datasets(cfg)` to read it.
    datasets: list[str] = field(default_factory=list)

    # Validation axis B: whole datasets kept OUT of training and scored on every
    # frame as a transfer test ("can it predict a condition it never trained on?").
    # A subset of ``datasets`` -- never all of them, or nothing is left to train.
    # Possible only because the model is conditioned on dimensionless groups, so a
    # held-out dataset still has a valid conditioning vector. Empty = train on all.
    heldout_datasets: list[str] = field(default_factory=list)

    stage: str = "all"

    experiment: ExperimentConfig = MISSING
    fluid: FluidConfig = MISSING
    imaging: ImagingConfig = MISSING
    scales: ScalesConfig = MISSING
    model: ModelConfig = MISSING
    training: TrainingConfig = MISSING

    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"experiment": "highest_t"},
            {"fluid": "fc72"},
            {"imaging": "default"},
            {"scales": "default"},
            {"model": "stage_a"},
            {"training": "stage_a"},
        ]
    )


def resolved_datasets(cfg) -> list[str]:
    """The datasets a run trains on: ``cfg.datasets`` if set, else ``[cfg.dataset]``.

    Order-preserving and deduplicated, so callers get a clean list whether the run
    is single- or multi-dataset. This is the one place the two config fields are
    reconciled, so every stage sees the same resolved list.
    """
    names = list(cfg.datasets) if cfg.datasets else [cfg.dataset]
    return list(dict.fromkeys(names))


def validate_split(names: list[str], heldout: list[str]) -> list[str]:
    """The training datasets for a run: ``names`` minus the held-out conditions
    (validation axis B), validated.

    Every held-out name must be one of ``names``, and at least one dataset must
    remain to train on. Both failures raise ``ValueError`` -- an invalid split is a
    config error, not a silent no-op. Order-preserving. This is the single source of
    truth for the split rules, shared by the CLI (:func:`training_datasets`) and the
    API's launch validation so the two can never drift.
    """
    heldout = list(dict.fromkeys(heldout))
    unknown = [name for name in heldout if name not in names]
    if unknown:
        raise ValueError(f"held-out datasets are not in the run: {unknown}")

    training = [name for name in names if name not in set(heldout)]
    if not training:
        raise ValueError("cannot hold out every dataset; no training datasets remain")
    return training


def training_datasets(cfg) -> list[str]:
    """The datasets actually supervised: ``resolved_datasets(cfg)`` minus the
    held-out conditions. Thin wrapper over :func:`validate_split`."""
    return validate_split(resolved_datasets(cfg), list(cfg.heldout_datasets))


def register_configs() -> None:
    """Register the schema so ``configs/*.yaml`` is validated against it."""
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Config)
    cs.store(group="experiment", name="base_experiment", node=ExperimentConfig)
    cs.store(group="fluid", name="base_fluid", node=FluidConfig)
    cs.store(group="imaging", name="base_imaging", node=ImagingConfig)
    cs.store(group="scales", name="base_scales", node=ScalesConfig)
    cs.store(group="model", name="base_model", node=ModelConfig)
    cs.store(group="training", name="base_training", node=TrainingConfig)

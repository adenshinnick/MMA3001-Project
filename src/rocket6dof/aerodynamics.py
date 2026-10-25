"""Aerodynamic model: normal force, centre of pressure, drag and moments.

Normal force and centre of pressure (CP)
----------------------------------------
Barrowman's method (Barrowman, 1967), in the form used by OpenRocket
(Niskanen, 2009). Each component produces a normal-force slope C_N_alpha
[1/rad] acting at its own CP:

- Nose cone: C_N_alpha = 2 for any shape. Its CP is at L - V/A_base from the
  tip (V = nose volume, A_base = base area). For an ellipsoid this is L/3.
- Body tube: no normal force at small angles (its cross-section is constant).
- Fins: a lifting-surface formula with a compressibility factor
  beta = sqrt(1 - M^2) and a body-interference factor 1 + r/(s + r).

The rocket's CP is the C_N_alpha-weighted average of the component CPs.

Drag
----
A drag build-up in the style of OpenRocket: skin friction on the wetted
area, pressure drag of the blunt fin edges, and base drag. The friction
coefficient is the larger of the smooth-turbulent and roughness-limited
values. These formulas are for subsonic flight, so drag is only computed
up to Mach 0.8 (``MACH_MAX``); above that the code stops with an error
instead of extrapolating.

Frames and signs
----------------
Body x points out through the nose. Stations (positions along the rocket)
are measured aft from the nose tip. The air-relative velocity in body axes
is ``v_air = R(q)^T (v - wind)``.
"""

from dataclasses import dataclass
from functools import cached_property
from math import atan2, cos, log, pi, sin, sqrt

import numpy as np
from scipy.interpolate import PchipInterpolator

from rocket6dof.quaternions import cross3

MACH_MAX = 0.8          # upper limit of the subsonic formulas used here [-]
NOSE_SHAPES = ("ellipsoid", "conical", "ogive")


@dataclass
class Geometry:
    """External shape of a single-stage rocket with one fin set.

    Attributes
    ----------
    diameter : float
        Body diameter, also the reference length [m].
    length : float
        Total length, nose tip to base [m].
    nose_shape : str
        ``"ellipsoid"``, ``"conical"`` or ``"ogive"`` (tangent ogive).
    nose_length : float
        Nose cone length [m].
    fin_count : int
        Number of fins (3 or more).
    root_chord, tip_chord, span, sweep : float
        Fin root chord, tip chord, span (root to tip) and leading-edge sweep
        length (root leading edge to tip leading edge, along the axis) [m].
    fin_thickness : float
        Fin thickness [m].
    fin_station : float
        Station of the fin root leading edge [m].
    roughness : float
        Surface roughness height [m] (OpenRocket "regular paint" = 60e-6).
    lug_count : int
        Number of launch lugs (0 allowed).
    lug_radius, lug_length : float
        Launch lug outer radius and length [m].

    Raises
    ------
    ValueError
        If any dimension is not physical; the message names the field.
    """

    diameter: float
    length: float
    nose_shape: str
    nose_length: float
    fin_count: int
    root_chord: float
    tip_chord: float
    span: float
    sweep: float
    fin_thickness: float
    fin_station: float
    roughness: float = 60e-6
    lug_count: int = 0
    lug_radius: float = 0.0
    lug_length: float = 0.0

    def __post_init__(self):
        positive = ["diameter", "length", "nose_length", "root_chord", "span",
                    "fin_thickness", "roughness"]
        for name in positive:
            if not getattr(self, name) > 0.0:
                raise ValueError(f"geometry field '{name}' must be > 0, got {getattr(self, name)}")
        for name in ["tip_chord", "sweep", "lug_radius", "lug_length"]:
            if not getattr(self, name) >= 0.0:
                raise ValueError(f"geometry field '{name}' must be >= 0, got {getattr(self, name)}")
        if self.nose_shape not in NOSE_SHAPES:
            raise ValueError(f"geometry field 'nose_shape' must be one of {NOSE_SHAPES}, "
                             f"got '{self.nose_shape}'")
        if int(self.fin_count) != self.fin_count or self.fin_count < 3:
            raise ValueError(f"geometry field 'fin_count' must be a whole number >= 3, "
                             f"got {self.fin_count}")
        if not self.nose_length < self.length:
            raise ValueError("geometry field 'nose_length' must be shorter than 'length'")
        if not self.fin_station + self.root_chord <= self.length + 1e-9:
            raise ValueError("fins extend past the base: check 'fin_station' and 'root_chord'")

    # Derived quantities. cached_property computes each one once, the first
    # time it is used, so the right-hand side f(t, x) does not redo them.

    @cached_property
    def radius(self):
        """Body radius [m]."""
        return 0.5 * self.diameter

    @cached_property
    def reference_area(self):
        """Reference area, the body cross-section pi r^2 [m^2]."""
        return pi * self.radius**2

    @cached_property
    def fin_area(self):
        """Planform area of one fin [m^2]."""
        return 0.5 * (self.root_chord + self.tip_chord) * self.span

    @cached_property
    def fin_mac(self):
        """Mean aerodynamic chord of one trapezoidal fin [m]."""
        a, b = self.root_chord, self.tip_chord
        return (2.0 / 3.0) * (a + b - a * b / (a + b))

    @cached_property
    def fin_midchord_cos(self):
        """Cosine of the mid-chord sweep angle [-]."""
        dx = self.sweep + 0.5 * self.tip_chord - 0.5 * self.root_chord
        return self.span / sqrt(self.span**2 + dx**2)

    @cached_property
    def fin_le_cos2(self):
        """cos^2 of the leading-edge sweep angle [-]."""
        return self.span**2 / (self.span**2 + self.sweep**2)

    @cached_property
    def nose_cp(self):
        """Station of the nose cone's CP [m] (Barrowman: L - V/A_base)."""
        fraction = {"ellipsoid": 1.0 / 3.0, "conical": 2.0 / 3.0, "ogive": 0.466}
        return fraction[self.nose_shape] * self.nose_length

    @cached_property
    def fin_cp(self):
        """Station of the fin set's CP [m] (Barrowman's subsonic formula)."""
        a, b, m = self.root_chord, self.tip_chord, self.sweep
        x_from_root_le = m * (a + 2.0 * b) / (3.0 * (a + b)) + (a + b - a * b / (a + b)) / 6.0
        return self.fin_station + x_from_root_le

    @cached_property
    def wetted_area_body(self):
        """Wetted area of the nose cone plus body tube [m^2], by numerical integration."""
        L, R = self.nose_length, self.radius
        x = np.linspace(0.0, L, 2001)
        r = nose_radius(self.nose_shape, x, L, R)
        drdx = np.gradient(r, x)
        nose = np.trapezoid(2.0 * pi * r * np.sqrt(1.0 + drdx**2), x)
        return float(nose) + 2.0 * pi * R * (self.length - L)

    @cached_property
    def fineness_ratio(self):
        """Length divided by diameter [-]."""
        return self.length / self.diameter

    @cached_property
    def nose_joint_sin2(self):
        """sin^2 of the angle where the nose meets the body (0 for smooth shapes)."""
        if self.nose_shape == "conical":
            return self.radius**2 / (self.radius**2 + self.nose_length**2)
        return 0.0


def nose_radius(shape, x, length, radius):
    """Radius of a nose cone at distance ``x`` from the tip [m].

    Parameters
    ----------
    shape : str
        One of ``NOSE_SHAPES``.
    x : numpy.ndarray
        Distances from the tip, 0 to ``length`` [m].
    length, radius : float
        Nose length and base radius [m].

    Returns
    -------
    numpy.ndarray
    """
    if shape == "ellipsoid":
        return radius * np.sqrt(np.clip(1.0 - ((length - x) / length) ** 2, 0.0, None))
    if shape == "conical":
        return radius * x / length
    if shape == "ogive":
        rho = (radius**2 + length**2) / (2.0 * radius)   # radius of the ogive arc
        return np.sqrt(np.clip(rho**2 - (length - x) ** 2, 0.0, None)) + radius - rho
    raise ValueError(f"unknown nose shape '{shape}'")


def _check_mach(mach):
    if not 0.0 <= mach <= MACH_MAX:
        raise ValueError(f"Mach {mach:.3f} is outside the aerodynamic model's range "
                         f"0 to {MACH_MAX}")


def fin_normal_force_slope(geom, mach):
    """Normal-force slope of the whole fin set, C_N_alpha [1/rad].

    One fin (OpenRocket form of Diederich's formula):

        C_N_alpha_1 = 2 pi (s^2 / A_ref) / (1 + sqrt(1 + (beta s^2 / (A_fin cos G))^2))

    with s the span, A_fin one fin's area, G the mid-chord sweep angle and
    beta = sqrt(1 - M^2). For 3 or more equally spaced fins the set gives
    N/2 times one fin. The body adds lift to the fins: factor 1 + r/(s + r).
    At M = 0 this equals Barrowman's original fin formula.

    Parameters
    ----------
    geom : Geometry
    mach : float
        Mach number, 0 to ``MACH_MAX``.

    Returns
    -------
    float
    """
    _check_mach(mach)
    beta = sqrt(1.0 - mach**2)
    s = geom.span
    one_fin = (2.0 * pi * s**2 / geom.reference_area) / (
        1.0 + sqrt(1.0 + (beta * s**2 / (geom.fin_area * geom.fin_midchord_cos)) ** 2))
    interference = 1.0 + geom.radius / (s + geom.radius)
    return 0.5 * geom.fin_count * one_fin * interference


def normal_force_slope_and_cp(geom, mach):
    """Total normal-force slope [1/rad] and CP station [m] of the rocket.

    Parameters
    ----------
    geom : Geometry
    mach : float
        Mach number, 0 to ``MACH_MAX``.

    Returns
    -------
    cn_alpha : float
        Normal-force slope of the whole rocket, per radian.
    x_cp : float
        Station of the centre of pressure, aft of the nose tip [m].
    """
    cn_nose = 2.0
    cn_fins = fin_normal_force_slope(geom, mach)
    cn_alpha = cn_nose + cn_fins
    x_cp = (cn_nose * geom.nose_cp + cn_fins * geom.fin_cp) / cn_alpha
    return cn_alpha, x_cp


def stability_margin(x_cp, x_cg, diameter):
    """Static stability margin in calibres: (x_cp - x_cg) / d [-].

    Positive means the CP is behind the CG, so the rocket turns into the wind
    (statically stable). About 1 calibre or more is the usual rule of thumb.
    """
    return (x_cp - x_cg) / diameter


def critical_reynolds(roughness, length):
    """Reynolds number above which friction is set by surface roughness [-]."""
    return 51.0 * (roughness / length) ** -1.039


def skin_friction_coefficient(reynolds, mach, roughness, length):
    """Skin-friction coefficient C_f, compressibility-corrected [-].

    Fully turbulent flow is assumed (Niskanen 2009, eqs 3.79-3.84):

    - smooth turbulent: C_f = 1 / (1.50 ln Re - 5.6)^2, or 1.48e-2 if Re < 1e4
    - roughness-limited: C_f = 0.032 (roughness / length)^0.2
    - subsonic compressibility factor (1 - 0.1 M^2) on both

    The larger of the two values is used, as OpenRocket does.

    Parameters
    ----------
    reynolds : float
        Reynolds number based on rocket length [-], >= 0.
    mach : float
        Mach number [-].
    roughness : float
        Surface roughness height [m].
    length : float
        Rocket length [m].

    Returns
    -------
    float
    """
    if reynolds < 1.0e4:
        cf_smooth = 1.48e-2
    else:
        cf_smooth = 1.0 / (1.50 * log(reynolds) - 5.6) ** 2
    cf_rough = 0.032 * (roughness / length) ** 0.2
    return max(cf_smooth, cf_rough) * (1.0 - 0.1 * mach**2)


def drag_breakdown(geom, mach, reynolds):
    """Zero-angle-of-attack drag coefficient, split into its parts.

    All coefficients use the body cross-section as reference area.

    Parameters
    ----------
    geom : Geometry
    mach : float
        Mach number, 0 to ``MACH_MAX``.
    reynolds : float
        Reynolds number based on rocket length [-].

    Returns
    -------
    dict
        ``"friction"``, ``"fin_pressure"``, ``"nose_pressure"``, ``"base"``
        and ``"total"`` [-].
    """
    _check_mach(mach)
    A = geom.reference_area
    cf = skin_friction_coefficient(reynolds, mach, geom.roughness, geom.length)

    # Skin friction. The body term is scaled up for its thickness (fineness
    # ratio f_B) and the fin term for theirs (thickness t over chord).
    body = (1.0 + 1.0 / (2.0 * geom.fineness_ratio)) * geom.wetted_area_body
    fins = (1.0 + 2.0 * geom.fin_thickness / geom.fin_mac) * 2.0 * geom.fin_count * geom.fin_area
    lugs = geom.lug_count * 2.0 * pi * geom.lug_radius * geom.lug_length
    friction = cf * (body + fins + lugs) / A

    # Square fin edges: the leading edge stops the flow (stagnation pressure,
    # reduced by sweep) and the trailing edge leaves a low-pressure wake.
    q_stag = 1.0 + mach**2 / 4.0 + mach**4 / 40.0     # stagnation / dynamic pressure
    base_cd = 0.12 + 0.13 * mach**2                    # base-drag coefficient, M < 1
    leading = 0.85 * q_stag * geom.fin_le_cos2
    fin_frontal = geom.fin_count * geom.fin_thickness * geom.span
    fin_pressure = (leading + base_cd) * fin_frontal / A

    nose_pressure = 0.8 * geom.nose_joint_sin2         # 0 for smooth nose shapes
    base = base_cd                                     # whole base area = A here

    total = friction + fin_pressure + nose_pressure + base
    return {"friction": friction, "fin_pressure": fin_pressure,
            "nose_pressure": nose_pressure, "base": base, "total": total}


class BuildupDrag:
    """Drag coefficient from the component build-up, C_D(M, Re).

    Parameters
    ----------
    geom : Geometry
    """

    def __init__(self, geom):
        self.geom = geom

    def __call__(self, mach, reynolds):
        """Total drag coefficient [-]."""
        return drag_breakdown(self.geom, mach, reynolds)["total"]


class ConstantDrag:
    """A constant drag coefficient: the simplest (baseline) drag model.

    Parameters
    ----------
    cd : float
        Drag coefficient [-], > 0.
    """

    def __init__(self, cd):
        if not cd > 0.0:
            raise ValueError(f"drag coefficient must be > 0, got {cd}")
        self.cd = float(cd)

    def __call__(self, mach, reynolds):
        """The constant drag coefficient (Mach and Reynolds are ignored)."""
        return self.cd


class DragTable:
    """Drag coefficient tabulated against Mach number.

    Values between table points come from monotone cubic (PCHIP)
    interpolation, which follows the data without overshooting. Mach numbers
    outside the table raise an error: nothing is extrapolated.

    Parameters
    ----------
    mach : array_like
        Mach numbers, strictly increasing.
    cd : array_like
        Drag coefficient at each Mach number, all > 0.

    Raises
    ------
    ValueError
        If the table is too short, not increasing, or has non-positive values.
    """

    def __init__(self, mach, cd):
        mach = np.asarray(mach, dtype=float)
        cd = np.asarray(cd, dtype=float)
        if mach.ndim != 1 or mach.size < 2 or mach.size != cd.size:
            raise ValueError("need matching 1-D arrays with at least 2 points")
        if np.any(np.diff(mach) <= 0.0):
            raise ValueError("Mach numbers must be strictly increasing")
        if np.any(cd <= 0.0) or not np.all(np.isfinite(cd)):
            raise ValueError("drag coefficients must be finite and > 0")
        self.mach = mach
        self.cd = cd
        self._interp = PchipInterpolator(mach, cd, extrapolate=False)

    def __call__(self, mach, reynolds=None):
        """Drag coefficient at ``mach`` [-] (``reynolds`` is ignored)."""
        if not self.mach[0] <= mach <= self.mach[-1]:
            raise ValueError(f"Mach {mach:.3f} is outside the drag table "
                             f"({self.mach[0]:.3f} to {self.mach[-1]:.3f})")
        return float(self._interp(mach))


def aerodynamic_loads(geom, drag_model, v_air, rho, sound_speed, viscosity, x_cg, omega):
    """Aerodynamic force and moment on the rocket, in body axes.

    Parameters
    ----------
    geom : Geometry
    drag_model : callable
        ``drag_model(mach, reynolds)`` returning C_D.
    v_air : numpy.ndarray
        Velocity of the rocket relative to the air, body axes [m/s].
    rho : float
        Air density [kg/m^3].
    sound_speed : float
        Speed of sound [m/s].
    viscosity : float
        Dynamic viscosity of air [Pa s].
    x_cg : float
        Station of the current CG [m].
    omega : numpy.ndarray
        Body angular velocity [rad/s].

    Returns
    -------
    force : numpy.ndarray
        Aerodynamic force, body axes [N].
    moment : numpy.ndarray
        Aerodynamic moment about the CG, body axes [N m].
    mach : float
        Mach number [-].

    Notes
    -----
    - Drag acts opposite ``v_air`` and passes through the CG.
    - Normal force C_N = C_N_alpha sin(alpha) acts at the CP, at right angles
      to the body axis, opposite the cross-flow. Linear (Barrowman) theory,
      so it is only accurate for small angles of attack (below about 10 deg).
    - Pitch damping: rotating at rate w adds a local cross-flow w * d at
      distance d from the CG, so each component (nose, fins) contributes
      -q A C_N_alpha_k d_k^2 w / V.
    - No roll moments: the fins have no cant.
    """
    speed = sqrt(v_air[0] ** 2 + v_air[1] ** 2 + v_air[2] ** 2)
    zero = np.zeros(3)
    if speed < 1e-9:
        return zero, zero.copy(), 0.0

    mach = speed / sound_speed
    reynolds = rho * speed * geom.length / viscosity
    q_area = 0.5 * rho * speed**2 * geom.reference_area   # dynamic pressure x area [N]

    force = -q_area * drag_model(mach, reynolds) * v_air / speed

    cross = sqrt(v_air[1] ** 2 + v_air[2] ** 2)            # cross-flow speed [m/s]
    cn_fins = fin_normal_force_slope(geom, mach)
    moment = np.zeros(3)
    if cross > 1e-12:
        alpha = atan2(cross, v_air[0])                     # total angle of attack [rad]
        cn_alpha = 2.0 + cn_fins
        x_cp = (2.0 * geom.nose_cp + cn_fins * geom.fin_cp) / cn_alpha
        normal = q_area * cn_alpha * sin(alpha)            # size of normal force [N]
        f_normal = np.array([0.0, -normal * v_air[1] / cross, -normal * v_air[2] / cross])
        force = force + f_normal
        arm = np.array([x_cg - x_cp, 0.0, 0.0])            # CG to CP, body axes [m]
        moment = cross3(arm, f_normal)

    # Pitch and yaw damping (body y and z rates only).
    damping = (q_area / speed) * (2.0 * (geom.nose_cp - x_cg) ** 2
                                  + cn_fins * (geom.fin_cp - x_cg) ** 2)
    moment[1] -= damping * omega[1]
    moment[2] -= damping * omega[2]
    return force, moment, mach

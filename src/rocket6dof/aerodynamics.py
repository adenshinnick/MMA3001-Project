"""Aerodynamic model: drag, normal force, centre of pressure (CP) and moments.

- Drag: a drag coefficient that is either constant, or rises with Mach number
  as C_D = C_D0 + k M^2 (fitted to OpenRocket's data by scripts/fit_drag.py).
- Normal force and CP: Barrowman's method. The nose gives C_N_alpha = 2 at
  L/3 from the tip (ellipsoid nose); the fins add a lifting-surface term with
  the compressibility factor sqrt(1 - M^2).

Body x points out through the nose; stations are measured aft from the nose tip.
Mach numbers above MACH_MAX stop the run, because the formulas are subsonic.
"""

from dataclasses import dataclass
from math import atan2, pi, sin, sqrt

import numpy as np

from rocket6dof.quaternions import cross3

MACH_MAX = 0.8          # upper limit of the subsonic formulas used here [-]


@dataclass
class Geometry:
    """External shape of the rocket: ellipsoid nose, body tube, one fin set.

    Attributes
    ----------
    diameter : float
        Body diameter, also the reference length [m].
    length : float
        Total length, nose tip to base [m].
    nose_length : float
        Ellipsoid nose cone length [m].
    fin_count : int
        Number of fins (3 or more).
    root_chord, tip_chord, span, sweep : float
        Fin root chord, tip chord, span, and leading-edge sweep length [m].
    fin_station : float
        Station of the fin root leading edge [m].

    Raises
    ------
    ValueError
        If a dimension is not physical; the message names the field.
    """

    diameter: float
    length: float
    nose_length: float
    fin_count: int
    root_chord: float
    tip_chord: float
    span: float
    sweep: float
    fin_station: float

    def __post_init__(self):
        for name in ["diameter", "length", "nose_length", "root_chord", "span"]:
            if not getattr(self, name) > 0.0:
                raise ValueError(f"geometry field '{name}' must be > 0, got {getattr(self, name)}")
        for name in ["tip_chord", "sweep"]:
            if not getattr(self, name) >= 0.0:
                raise ValueError(f"geometry field '{name}' must be >= 0, got {getattr(self, name)}")
        if int(self.fin_count) != self.fin_count or self.fin_count < 3:
            raise ValueError(f"geometry field 'fin_count' must be a whole number >= 3, got {self.fin_count}")
        if not self.nose_length < self.length:
            raise ValueError("geometry field 'nose_length' must be shorter than 'length'")
        if not self.fin_station + self.root_chord <= self.length + 1e-9:
            raise ValueError("fins extend past the base: check 'fin_station' and 'root_chord'")

    @property
    def radius(self):
        """Body radius [m]."""
        return 0.5 * self.diameter

    @property
    def reference_area(self):
        """Reference area, the body cross-section pi r^2 [m^2]."""
        return pi * self.radius**2

    @property
    def nose_cp(self):
        """Station of the nose CP [m]: L - V/A_base, which is L/3 for an ellipsoid."""
        return self.nose_length / 3.0

    @property
    def fin_cp(self):
        """Station of the fin set's CP [m] (Barrowman)."""
        a, b, m = self.root_chord, self.tip_chord, self.sweep
        x_from_root_le = m * (a + 2.0 * b) / (3.0 * (a + b)) + (a + b - a * b / (a + b)) / 6.0
        return self.fin_station + x_from_root_le


def _check_mach(mach):
    if not 0.0 <= mach <= MACH_MAX:
        raise ValueError(f"Mach {mach:.3f} is outside the aerodynamic model's range 0 to {MACH_MAX}")


def fin_normal_force_slope(geom, mach):
    """Normal-force slope of the fin set, C_N_alpha [1/rad].

    One fin: 2 pi (s^2/A_ref) / (1 + sqrt(1 + (beta s^2 / (A_fin cos G))^2)),
    with s the span, A_fin one fin's area, G the mid-chord sweep angle and
    beta = sqrt(1 - M^2). Three or more fins give N/2 times one fin, and the
    body adds the interference factor 1 + r/(s + r). At M = 0 this is
    Barrowman's fin formula.

    Parameters
    ----------
    geom : Geometry
    mach : float
        Mach number, 0 to MACH_MAX.

    Returns
    -------
    float
    """
    _check_mach(mach)
    s, a, b = geom.span, geom.root_chord, geom.tip_chord
    fin_area = 0.5 * (a + b) * s                                   # one trapezoidal fin
    # Cosine of the sweep angle of the line joining the mid-points of the chords.
    cos_midchord = s / sqrt(s**2 + (geom.sweep + 0.5 * b - 0.5 * a) ** 2)
    beta = sqrt(1.0 - mach**2)        # compressibility correction: grows the force near Mach 1
    one_fin = (2.0 * pi * s**2 / geom.reference_area) / (
        1.0 + sqrt(1.0 + (beta * s**2 / (fin_area * cos_midchord)) ** 2))
    interference = 1.0 + geom.radius / (s + geom.radius)
    return 0.5 * geom.fin_count * one_fin * interference


def normal_force_slope_and_cp(geom, mach):
    """Total normal-force slope [1/rad] and CP station [m] of the rocket.

    The CP is the C_N_alpha-weighted average of the nose and fin CPs.
    """
    cn_fins = fin_normal_force_slope(geom, mach)
    cn_alpha = 2.0 + cn_fins          # Barrowman: any nose contributes 2 per radian
    x_cp = (2.0 * geom.nose_cp + cn_fins * geom.fin_cp) / cn_alpha
    return cn_alpha, x_cp


def stability_margin(x_cp, x_cg, diameter):
    """Static stability margin (x_cp - x_cg) / d in calibres; positive is stable."""
    return (x_cp - x_cg) / diameter


class ConstantDrag:
    """A constant drag coefficient: the baseline drag model."""

    def __init__(self, cd):
        if not cd > 0.0:
            raise ValueError(f"drag coefficient must be > 0, got {cd}")
        self.cd = float(cd)

    def __call__(self, mach):
        return self.cd


class MachDrag:
    """Drag coefficient rising with Mach number: C_D = cd0 + k M^2.

    cd0 is the drag coefficient at low speed, and k M^2 is the rise in drag
    as the air starts to compress near the speed of sound. The two numbers
    are fitted to OpenRocket's flight data by ``scripts/fit_drag.py``
    (linear regression of C_D against M^2). Valid up to Mach 0.8.
    """

    def __init__(self, cd0, k):
        if not cd0 > 0.0 or not k >= 0.0:
            raise ValueError(f"need cd0 > 0 and k >= 0, got {cd0}, {k}")
        self.cd0, self.k = float(cd0), float(k)

    def __call__(self, mach):
        _check_mach(mach)
        return self.cd0 + self.k * mach**2


def aerodynamic_loads(geom, drag_model, v_air, rho, sound_speed, x_cg, omega):
    """Aerodynamic force and moment on the rocket, in body axes.

    Parameters
    ----------
    geom : Geometry
    drag_model : callable
        ``drag_model(mach)`` returning C_D.
    v_air : numpy.ndarray
        Velocity of the rocket relative to the air, body axes [m/s].
    rho, sound_speed : float
        Air density [kg/m^3] and speed of sound [m/s].
    x_cg : float
        Station of the current CG [m].
    omega : numpy.ndarray
        Body angular velocity [rad/s].

    Returns
    -------
    force, moment : numpy.ndarray
        Force [N] and moment about the CG [N m], body axes.
    mach : float
        Mach number [-].

    Notes
    -----
    Drag acts opposite ``v_air``. The normal force q A C_N_alpha sin(alpha)
    acts at the CP, opposing the cross-flow (linear theory: small angles of
    attack only). Rotation adds a cross-flow w*d at distance d from the CG,
    which gives the pitch damping term.
    """
    speed = sqrt(v_air[0] ** 2 + v_air[1] ** 2 + v_air[2] ** 2)
    if speed < 1e-9:                  # no airflow, so no aerodynamic force
        return np.zeros(3), np.zeros(3), 0.0

    mach = speed / sound_speed
    q_area = 0.5 * rho * speed**2 * geom.reference_area     # dynamic pressure x area [N]
    force = -q_area * drag_model(mach) * v_air / speed      # drag, opposite the airflow
    moment = np.zeros(3)

    cn_fins = fin_normal_force_slope(geom, mach)
    # The body x axis is the rocket's axis, so the y and z parts of v_air are
    # the air flowing across the rocket. Without them there is no normal force.
    cross = sqrt(v_air[1] ** 2 + v_air[2] ** 2)              # cross-flow speed [m/s]
    if cross > 1e-12:
        alpha = atan2(cross, v_air[0])                       # angle of attack [rad]
        cn_alpha = 2.0 + cn_fins
        x_cp = (2.0 * geom.nose_cp + cn_fins * geom.fin_cp) / cn_alpha
        normal = q_area * cn_alpha * sin(alpha)              # size of the normal force [N]
        # The normal force pushes against the cross-flow, perpendicular to the axis.
        f_normal = np.array([0.0, -normal * v_air[1] / cross, -normal * v_air[2] / cross])
        force = force + f_normal
        moment = cross3(np.array([x_cg - x_cp, 0.0, 0.0]), f_normal)   # arm from CG to CP

    # Pitch and yaw damping from the nose and fins: a rotating rocket moves
    # each part sideways through the air, which creates a force opposing the
    # rotation. Parts further from the CG contribute more (distance squared).
    damping = (q_area / speed) * (2.0 * (geom.nose_cp - x_cg) ** 2
                                  + cn_fins * (geom.fin_cp - x_cg) ** 2)
    moment[1] -= damping * omega[1]
    moment[2] -= damping * omega[2]
    return force, moment, mach

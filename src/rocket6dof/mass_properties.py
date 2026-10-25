"""Mass, centre of gravity and inertia of a rocket built from simple parts.

OpenRocket reports the rocket's total mass and centre of gravity (CG), but
the CSV export used here does not include its moments of inertia. The 6-DOF
equations need the inertia tensor, so this module estimates it from a list
of simple shapes (tubes, a hollow nose cone, flat fins, point masses).

Every part is described by its mass, the station of its own CG, and its
moments of inertia about its own CG. ``combine`` then adds the parts using
the parallel-axis theorem:

    I_total = sum( I_own + m * d^2 )

where d is the distance from the part's CG to the combined CG.

Conventions
-----------
- Stations are measured aft from the nose tip along the rocket axis [m].
- ``I_axial`` is about the long (body x, roll) axis [kg m^2].
- ``I_transverse`` is about an axis through the CG at right angles to the
  long axis (pitch and yaw are equal for a rocket with 3 or more fins).
"""

from dataclasses import dataclass
from math import pi

import numpy as np


@dataclass
class Part:
    """One part of the rocket.

    Attributes
    ----------
    name : str
        Label used in printed tables.
    mass : float
        Mass [kg].
    x_cg : float
        Station of the part's own CG [m].
    I_axial : float
        Moment of inertia about the long axis, through the part's CG [kg m^2].
    I_transverse : float
        Moment of inertia about a transverse axis through the part's CG [kg m^2].
    """

    name: str
    mass: float
    x_cg: float
    I_axial: float = 0.0
    I_transverse: float = 0.0


def tube(name, density, length, r_outer, r_inner, x_front):
    """A hollow (or, with ``r_inner = 0``, solid) circular cylinder.

    Parameters
    ----------
    name : str
        Label.
    density : float
        Material density [kg/m^3].
    length : float
        Length along the rocket axis [m].
    r_outer, r_inner : float
        Outer and inner radius [m]. Use ``r_inner = 0`` for a solid disc.
    x_front : float
        Station of the front face [m].

    Returns
    -------
    Part

    Raises
    ------
    ValueError
        If the dimensions are not physical.
    """
    if not (length > 0.0 and r_outer > r_inner >= 0.0 and density > 0.0):
        raise ValueError(f"{name}: need length > 0, r_outer > r_inner >= 0, density > 0")
    mass = density * pi * (r_outer**2 - r_inner**2) * length
    r2 = r_outer**2 + r_inner**2
    return Part(
        name,
        mass,
        x_front + 0.5 * length,
        I_axial=0.5 * mass * r2,                        # standard result for a thick tube
        I_transverse=mass * (3.0 * r2 + length**2) / 12.0,
    )


def ellipsoid_nose(name, density, length, radius, wall, n=4001):
    """A hollow ellipsoidal nose cone, integrated numerically in thin slices.

    The outer radius at distance x from the tip is
    ``r(x) = R * sqrt(1 - ((L - x) / L)^2)`` and the inner radius is
    ``max(r - wall, 0)``. Each slice is a thin disc (or ring), so its mass
    and inertia are known exactly; the trapezoidal rule adds the slices up.

    Parameters
    ----------
    name : str
        Label.
    density : float
        Material density [kg/m^3].
    length : float
        Nose length, tip to shoulder [m].
    radius : float
        Base radius [m].
    wall : float
        Wall thickness [m].
    n : int
        Number of slice boundaries for the trapezoidal rule.

    Returns
    -------
    Part
    """
    x = np.linspace(0.0, length, n)
    r = radius * np.sqrt(np.clip(1.0 - ((length - x) / length) ** 2, 0.0, None))
    r_in = np.clip(r - wall, 0.0, None)
    area = pi * (r**2 - r_in**2)                       # slice cross-section [m^2]
    mass = density * np.trapezoid(area, x)
    x_cg = density * np.trapezoid(area * x, x) / mass
    # Thin disc/ring: dI_axial = 0.5 dm (r^2 + r_in^2); dI_diameter = 0.25 dm (r^2 + r_in^2).
    I_axial = density * np.trapezoid(0.5 * pi * (r**4 - r_in**4), x)
    I_about_tip = density * np.trapezoid(0.25 * pi * (r**4 - r_in**4) + area * x**2, x)
    return Part(name, float(mass), float(x_cg), float(I_axial), float(I_about_tip - mass * x_cg**2))


def trapezoidal_fins(name, density, count, root, tip, span, sweep, thickness,
                     x_root_le, body_radius):
    """A set of flat trapezoidal fins, treated as thin plates.

    Parameters
    ----------
    name : str
        Label.
    density : float
        Material density [kg/m^3].
    count : int
        Number of fins (3 or more).
    root, tip, span, sweep : float
        Root chord, tip chord, span and leading-edge sweep length [m].
    thickness : float
        Fin thickness [m].
    x_root_le : float
        Station of the root leading edge [m].
    body_radius : float
        Radius of the body tube the fins are attached to [m].

    Returns
    -------
    Part
    """
    area = 0.5 * (root + tip) * span                   # one fin, one side [m^2]
    mass = density * thickness * area * count
    # Centroid of the trapezoid: along the axis from the root leading edge,
    # and outwards from the root.
    x_c = (root**2 + root * tip + tip**2 + sweep * (root + 2.0 * tip)) / (3.0 * (root + tip))
    y_c = span * (root + 2.0 * tip) / (3.0 * (root + tip))
    r_c = body_radius + y_c                            # distance of fin centroid from the axis
    # Point-mass approximation about the axis (fins are thin and far out).
    I_axial = mass * r_c**2
    # About a transverse axis: half the fins' radial spread counts on average
    # (sum of sin^2 over 3+ equally spaced fins is count/2), plus their chordwise spread.
    I_transverse = mass * (0.5 * r_c**2 + (root**2 + tip**2) / 24.0)
    return Part(name, mass, x_root_le + x_c, I_axial, I_transverse)


def combine(parts):
    """Combine parts into total mass, CG station and inertia about the CG.

    Parameters
    ----------
    parts : list of Part

    Returns
    -------
    mass : float
        Total mass [kg].
    x_cg : float
        Station of the combined CG [m].
    I_axial : float
        Roll inertia about the long axis [kg m^2].
    I_transverse : float
        Pitch/yaw inertia about the combined CG [kg m^2].

    Raises
    ------
    ValueError
        If the list is empty or the total mass is not positive.
    """
    if not parts:
        raise ValueError("need at least one part")
    mass = sum(p.mass for p in parts)
    if not mass > 0.0:
        raise ValueError("total mass must be positive")
    x_cg = sum(p.mass * p.x_cg for p in parts) / mass
    I_axial = sum(p.I_axial for p in parts)
    # Parallel-axis theorem: move each part's transverse inertia to the combined CG.
    I_transverse = sum(p.I_transverse + p.mass * (p.x_cg - x_cg) ** 2 for p in parts)
    return mass, x_cg, I_axial, I_transverse

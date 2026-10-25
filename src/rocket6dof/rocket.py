"""Rocket description: the data the equations of motion need.

A ``Rocket`` holds the dry (motor-less) mass properties, the motor, and
optionally the external geometry (for aerodynamics) and a parachute.

Mass properties during the burn
-------------------------------
If the dry CG station and the motor's position are given, the CG and the
inertia tensor change as propellant burns. The motor is treated as a
uniform solid cylinder whose mass falls with the impulse delivered:

    m(t)    = m_dry + m_motor(t)
    x_cg(t) = (m_dry x_dry + m_motor(t) x_motor) / m(t)
    J(t)    = J_dry + J_motor(t) + parallel-axis terms to x_cg(t)

The rate of change of inertia (the J-dot term) and jet damping are
neglected in Euler's equation; see ``rigid_body.py``.
"""

from dataclasses import dataclass
from math import pi

import numpy as np

from rocket6dof.rigid_body import validate_inertia


@dataclass
class Parachute:
    """A recovery parachute, deployed at apogee.

    Attributes
    ----------
    diameter : float
        Canopy diameter [m].
    drag_coefficient : float
        Drag coefficient based on canopy area [-].
    """

    diameter: float
    drag_coefficient: float

    def __post_init__(self):
        if not self.diameter > 0.0:
            raise ValueError(f"parachute 'diameter' must be > 0, got {self.diameter}")
        if not self.drag_coefficient > 0.0:
            raise ValueError(f"parachute 'drag_coefficient' must be > 0, got {self.drag_coefficient}")

    @property
    def area(self):
        """Canopy reference area pi d^2 / 4 [m^2]."""
        return 0.25 * pi * self.diameter**2


class Rocket:
    """A single stage rocket.

    Parameters
    ----------
    dry_mass : float
        Mass of the rocket without its motor [kg].
    inertia : array_like
        Inertia tensor of the dry rocket about its own CG, body axes [kg m^2].
    motor : Motor
        The rocket's motor.
    geometry : Geometry, optional
        External shape. If omitted, no aerodynamic forces act.
    drag_model : callable, optional
        ``drag_model(mach, reynolds)`` returning C_D. Needed with ``geometry``.
    dry_cg : float, optional
        Station of the dry CG, aft of the nose tip [m].
    motor_station : float, optional
        Station of the motor's front end [m]. With ``dry_cg``, makes the CG
        and inertia vary during the burn. Needs the motor's length.
    parachute : Parachute, optional
        Deployed at apogee if given.
    name : str
        Label for printouts.

    Raises
    ------
    ValueError
        If the dry mass is not positive, the inertia tensor is invalid, or
        the optional data are inconsistent.
    """

    def __init__(self, dry_mass, inertia, motor, geometry=None, drag_model=None,
                 dry_cg=None, motor_station=None, parachute=None, name="rocket"):
        if not dry_mass > 0.0:
            raise ValueError(f"Dry mass must be positive, got {dry_mass}")
        if geometry is not None and drag_model is None:
            raise ValueError("a rocket with geometry also needs a drag_model")
        if geometry is not None and dry_cg is None:
            raise ValueError("a rocket with geometry also needs dry_cg (for the moment arm)")
        if motor_station is not None:
            if dry_cg is None:
                raise ValueError("motor_station needs dry_cg as well")
            if motor.length is None or motor.diameter is None:
                raise ValueError("motor_station needs the motor's length and diameter")
        self.dry_mass = float(dry_mass)
        self.inertia = validate_inertia(inertia)
        self.motor = motor
        self.geometry = geometry
        self.drag_model = drag_model
        self.dry_cg = dry_cg
        self.motor_station = motor_station
        self.parachute = parachute
        self.name = name

    def mass(self, t):
        """Total mass at time t after ignition [kg]."""
        return self.dry_mass + self.motor.mass(t)

    def cg(self, t):
        """Station of the CG at time t [m] (the dry CG if the motor position is unknown)."""
        if self.motor_station is None:
            return self.dry_cg
        x_motor = self.motor_station + 0.5 * self.motor.length
        m_motor = self.motor.mass(t)
        return (self.dry_mass * self.dry_cg + m_motor * x_motor) / (self.dry_mass + m_motor)

    def inertia_at(self, t):
        """Inertia tensor about the current CG, body axes [kg m^2].

        Constant (the dry value) unless ``dry_cg`` and ``motor_station`` were given.
        """
        if self.motor_station is None:
            return self.inertia
        m_motor = self.motor.mass(t)
        r = 0.5 * self.motor.diameter
        L = self.motor.length
        x_motor = self.motor_station + 0.5 * L
        x_cg = self.cg(t)
        # Uniform solid cylinder for the motor, about its own CG.
        J_motor = np.diag([0.5 * m_motor * r**2,
                           m_motor * (3.0 * r**2 + L**2) / 12.0,
                           m_motor * (3.0 * r**2 + L**2) / 12.0])
        # Parallel-axis theorem for an offset d along body x: adds m d^2 to the
        # two transverse moments and nothing to the axial one.
        shift = np.diag([0.0, 1.0, 1.0])
        d_dry = self.dry_cg - x_cg
        d_motor = x_motor - x_cg
        return (self.inertia + self.dry_mass * d_dry**2 * shift
                + J_motor + m_motor * d_motor**2 * shift)

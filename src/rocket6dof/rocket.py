"""Rocket description: mass properties, motor, geometry and parachute.

The CG moves forward as the propellant burns; the inertia tensor is kept
constant (the inertia-rate and jet-damping terms are neglected).
"""

from dataclasses import dataclass
from math import pi

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
        Inertia tensor about the CG, body axes [kg m^2] (held constant).
    motor : Motor
        The rocket's motor.
    geometry : Geometry, optional
        External shape. If omitted, no aerodynamic forces act.
    drag_model : callable, optional
        ``drag_model(mach)`` returning C_D. Needed with ``geometry``.
    dry_cg : float, optional
        Station of the dry CG, aft of the nose tip [m]. Needed with ``geometry``.
    motor_station : float, optional
        Station of the motor's front end [m]. With ``dry_cg``, the CG moves as
        propellant burns (needs the motor's length).
    parachute : Parachute, optional
        Deployed at apogee if given.
    name : str
        Label for printouts.

    Raises
    ------
    ValueError
        If the dry mass is not positive, the inertia tensor is invalid, or the
        optional data are inconsistent.
    """

    def __init__(self, dry_mass, inertia, motor, geometry=None, drag_model=None,
                 dry_cg=None, motor_station=None, parachute=None, name="rocket"):
        if not dry_mass > 0.0:
            raise ValueError(f"Dry mass must be positive, got {dry_mass}")
        if geometry is not None and (drag_model is None or dry_cg is None):
            raise ValueError("a rocket with geometry also needs a drag_model and dry_cg")
        if motor_station is not None and (dry_cg is None or motor.length is None):
            raise ValueError("motor_station needs dry_cg and the motor's length")
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
        """Station of the CG at time t [m]: mass-weighted average of dry rocket and motor."""
        if self.motor_station is None:
            return self.dry_cg
        x_motor = self.motor_station + 0.5 * self.motor.length    # middle of the motor
        m_motor = self.motor.mass(t)
        return (self.dry_mass * self.dry_cg + m_motor * x_motor) / (self.dry_mass + m_motor)

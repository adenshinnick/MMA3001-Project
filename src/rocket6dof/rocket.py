"""Rocket description: the data the equations of motion need"""

from rocket6dof.rigid_body import validate_inertia

class Rocket:
    """ A single stage rocket.
    
    Parameters
    ----------
    dry_mass : float
        Mass of the rocket without its motor [kg]'
    inertia: array_like
        Inertia tensor about the centre of mass, body axes [kg m^2]
    motor: Motor
        The rocket's Motor
        
    Raises
    ------
    ValueError
    If the dry mass is not positive, or the inertia tensor is invalid
    """
    def __init__(self,dry_mass, inertia, motor):
        if not dry_mass> 0.0:
            raise ValueError(f"Dry mass must be positive, got {dry_mass}")
        self.dry_mass= float(dry_mass)
        self.inertia=validate_inertia(inertia)
        self.motor = motor

    def mass(self, t):
        """Total mass at time t after ignition [kg]"""
        return self.dry_mass + self.motor.mass(t)
"""Wind models: horizontal wind velocity as a function of height.

Earth frame: x east, y north, z up. Wind *direction* follows the
meteorological convention: the direction the wind blows FROM, in degrees
clockwise from north. A 270 deg wind blows from the west, towards the east.

The aerodynamic forces use the velocity relative to the air,
``v_air = v - wind(z)``.
"""

from math import cos, radians, sin

import numpy as np


class Wind:
    """Horizontal wind, constant or following a power-law profile.

    The power law ``w(z) = w_ref (z / z_ref)^exponent`` is a common model of
    the wind near the ground (the exponent is about 1/7 over open country).
    ``exponent = 0`` gives a wind that is the same at every height.

    Parameters
    ----------
    speed : float
        Wind speed at the reference height [m/s], >= 0.
    direction_deg : float
        Direction the wind blows from, clockwise from north [deg].
    exponent : float
        Power-law exponent [-], >= 0.
    reference_height : float
        Height where ``speed`` applies [m], > 0.

    Raises
    ------
    ValueError
        If the speed or exponent is negative, or the reference height is not
        positive.
    """

    def __init__(self, speed, direction_deg=270.0, exponent=0.0, reference_height=10.0):
        if not speed >= 0.0:
            raise ValueError(f"wind speed must be >= 0, got {speed}")
        if not exponent >= 0.0:
            raise ValueError(f"wind exponent must be >= 0, got {exponent}")
        if not reference_height > 0.0:
            raise ValueError(f"reference height must be > 0, got {reference_height}")
        self.speed = float(speed)
        self.direction_deg = float(direction_deg)
        self.exponent = float(exponent)
        self.reference_height = float(reference_height)
        # Unit vector the wind blows TOWARDS (opposite to where it comes from).
        theta = radians(direction_deg)
        self._towards = np.array([-sin(theta), -cos(theta), 0.0])

    def __call__(self, z):
        """Wind velocity at height ``z`` above the pad, Earth frame [m/s]."""
        if self.exponent == 0.0 or self.speed == 0.0:
            return self.speed * self._towards
        # Below 0.1 m the profile is held at its 0.1 m value (it tends to 0 at the ground).
        scale = (max(z, 0.1) / self.reference_height) ** self.exponent
        return self.speed * scale * self._towards


NO_WIND = Wind(0.0)

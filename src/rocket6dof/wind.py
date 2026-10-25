"""Constant horizontal wind.

Direction is where the wind blows FROM, clockwise from north (270 = from the
west, so it blows towards +x, east). Aerodynamic forces use v_air = v - wind.
"""

from math import cos, radians, sin

import numpy as np


class Wind:
    """A steady horizontal wind, the same at every height.

    Parameters
    ----------
    speed : float
        Wind speed [m/s], >= 0.
    direction_deg : float
        Direction the wind blows from, clockwise from north [deg].

    Raises
    ------
    ValueError
        If the speed is negative.
    """

    def __init__(self, speed, direction_deg=270.0):
        if not speed >= 0.0:
            raise ValueError(f"wind speed must be >= 0, got {speed}")
        self.speed = float(speed)
        self.direction_deg = float(direction_deg)
        theta = radians(direction_deg)       # compass angle, clockwise from north
        # Unit vector the wind blows TOWARDS (opposite to where it comes from).
        self.velocity = self.speed * np.array([-sin(theta), -cos(theta), 0.0])

    def __call__(self, z):
        """Wind velocity at height z, Earth frame [m/s]."""
        return self.velocity

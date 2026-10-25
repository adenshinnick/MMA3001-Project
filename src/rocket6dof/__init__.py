"""rocket6dof: a six-degree-of-freedom flight simulator for a single-stage rocket.

Written for the MMA3001 individual project. Reads a rocket description
(``data/rocket/l1_rocket.toml``) and launch conditions, and predicts the flight:
apogee, speeds, stability margin and landing point. See the README for usage.

Conventions: SI units; Earth frame x east, y north, z up; body x out through the
nose; quaternions scalar-first [w, x, y, z], rotating body vectors into the
Earth frame.
"""

__docformat__ = "numpy"
__version__ = "1.0.0"

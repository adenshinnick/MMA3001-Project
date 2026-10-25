"""rocket6dof: a six-degree-of-freedom flight simulator for a single-stage rocket.

Written for the MMA3001 individual project. Given a rocket description file
(geometry, mass properties, motor thrust curve, parachute) and launch
conditions (rail length and angle, wind, site altitude), it predicts the
flight: apogee, speeds, stability margin, and landing point.

Modules, in the order the physics builds up
-------------------------------------------
- `rocket6dof.integrators`: fixed-step RK4 and explicit Euler.
- `rocket6dof.atmosphere`: International Standard Atmosphere and air viscosity.
- `rocket6dof.motor`: thrust curve (RASP ``.eng`` files) and motor mass.
- `rocket6dof.quaternions`: attitude as a unit quaternion.
- `rocket6dof.euler_angles`: Euler angles, kept as the baseline for comparison.
- `rocket6dof.rigid_body`: Euler's rotation equation.
- `rocket6dof.mass_properties`: mass, CG and inertia from simple parts.
- `rocket6dof.aerodynamics`: Barrowman normal force and CP, drag build-up, moments.
- `rocket6dof.wind`: wind profiles.
- `rocket6dof.rocket`: the ``Rocket`` (mass, CG and inertia during the burn).
- `rocket6dof.config`: reading a rocket from a TOML file, with input checks.
- `rocket6dof.dynamics`: the 13-state equations of motion.
- `rocket6dof.simulation`: the flight from pad to landing, with events.
- `rocket6dof.benchmark`: reading OpenRocket exports and comparing against them.
- `rocket6dof.analysis`: convergence, integrator, sensitivity and Monte Carlo studies.

Quick start
-----------
>>> from rocket6dof.config import load_rocket
>>> from rocket6dof.simulation import launch_attitude, simulate, summarise
>>> rocket = load_rocket("data/rocket/l1_rocket.toml")
>>> result = simulate(rocket, launch_attitude(90.0), rail_length=1.0, h=0.02, h_descent=0.1)
>>> round(summarise(result)["apogee"])  # doctest: +SKIP
1015

Conventions
-----------
SI units throughout. Earth frame: x east, y north, z up (flat, non-rotating
Earth). Body frame: x out through the nose. Quaternions are scalar-first
[w, x, y, z] and rotate body vectors into the Earth frame. Stations along
the rocket are measured aft from the nose tip.
"""

__docformat__ = "numpy"
__version__ = "1.0.0"

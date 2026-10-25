"""First flight: simulate a demo rocket and plot its trajectory. 

There is no aerodynamic drag yet, so the apogee is far higher than a real flight. 
This checks the simulator from end to end, not the rocket
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rocket6dof.motor import Motor
from rocket6dof.quaternions import quat_from_axis_angle
from rocket6dof.rocket import Rocket
from rocket6dof.simulation import simulate

# A made-up thrust curve shaped like a small H-class motor (not real data)
motor = Motor(
    "DEMO",
    [0.05, 0.2, 1.0, 1.6, 1.8],
    [180.0, 150.0, 140.0, 120.0, 0.0],
    propellant_mass=0.09,
    total_mass=0.2,
)
rocket = Rocket(dry_mass=1.2, inertia=np.diag([0.002,0.1,0.1]), motor=motor)
q_launch=quat_from_axis_angle([0.0,1.0,0.0], -np.radians(85.0)) #85 degree rail

result=simulate(rocket, q_launch, rail_length=1.5, h=0.01)
t,x = result["t"],result["x"]
events = result["events"]

print(f"Status:          {result['status']}")
print(f"Rail-exit speed: {events['rail_exit']['speed']:.1f} m/s")
print(f"Burnout:         {events['burnout']['t']:.2f} s")
print(f"Apogee:          {events['apogee']['position'][2]:.0f} m "
      f"at {events['apogee']['t']:.1f} s")
print(f"Landing:         {events['landing']['t']:.1f} s, "
      f"{np.linalg.norm(events['landing']['position'][:2]):.0f} m from the pad")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(t, x[:, 2])
ax1.axvline(events["burnout"]["t"], linestyle="--", color="grey", label="Burnout")
ax1.plot(events["apogee"]["t"], events["apogee"]["position"][2], "o", label="Apogee")
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Altitude [m]")
ax1.legend()

ax2.plot(x[:, 0], x[:, 2])
ax2.set_xlabel("Distance east [m]")
ax2.set_ylabel("Altitude [m]")
ax2.set_aspect("equal")

fig.suptitle("Demo flight, no drag")
fig.tight_layout()

out_dir = Path("reports/figures")
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "first_flight.png", dpi=150)
plt.show()
""" Solid rocket motor model: thrust and mass as functions of time.

Thrust comes from a measured thrust curve, linearly interpolated between
data points. Mass falls in proportion to the impulse delivered so far
"""
import numpy as np

class Motor:
    """ A solid rocket motor described by a measured thrust curve

    Parameters
    ----------
    name : str
        Motor designation, for example ``"H128W"``.
    times : array_like
        Times of the thrust data points [s], strictly increasing, starting
        at 0 or later. If the first time is after 0, a point (0 s, 0 N) is
        added at the start.
    thrusts : array_like
        Thrust at each time [N]. Must be 0 or more, and end at 0.
    propellant_mass : float
        Propellant mass at ignition [kg].
    total_mass : float
        Motor mass at ignition, casing plus propellant [kg].
    diameter : float, optional
        Motor diameter [m]. Needed only for the motor's moment of inertia.
    length : float, optional
        Motor length [m]. Needed for the motor's CG and moment of inertia.

    Raises
    ------
    ValueError
        If the thrust curve or masses are invalid.
    """
    def __init__(self, name, times, thrusts, propellant_mass, total_mass,
                 diameter=None, length=None):
        times = np.asarray(times, dtype = float)
        thrusts = np.asarray(thrusts, dtype=float)

        if times.size != thrusts.size or times.size < 1:
            raise ValueError("times and thrusts must be non-empty, and the same length ")
        if times[0] < 0.0:
            raise ValueError("Time must be non negative")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("Thrust curve times must be strictly increasing")
        if np.any(thrusts<0.0):
            raise ValueError("Thrust values must be higher than or equal to 0")
        if thrusts[-1] != 0.0:
            raise ValueError("The thrust curve must end at 0N (Burnout)")
        if not 0.0 <propellant_mass < total_mass:
            raise ValueError("need 0 < propellant mass < total mass")

        #the curve implicitly starts from rest at ignition
        if times[0] > 0.0:
            times = np.insert(times,0,0.0)
            thrusts = np.insert(thrusts,0,0.0)

        self.name = name
        self.times = times
        self.thrusts = thrusts
        self.propellant_mass = propellant_mass
        self.total_mass = total_mass
        self.burn_time = times[-1]
        self.diameter = diameter
        self.length = length

        #Impulse delivered by the end of each data point, by the trapezoidal rule
        #This is exact when assuming thrust is linear between points
        segment_impulse = 0.5 * (thrusts[1:]+thrusts[:-1])*np.diff(times)
        self.cumulative_impulse = np.concatenate(([0.0],np.cumsum(segment_impulse)))
        self.total_impulse = self.cumulative_impulse[-1]

        if self.total_impulse <=0:
            raise ValueError("The thrust curve delivers no impulse.")
        
    def thrust(self,t):
        """Thrust at time ''t'' after ignition [N]; 0 outside of burn."""
        return float(np.interp(t,self.times,self.thrusts, left=0.0,right=0.0))

    def impulse(self,t):
        """Impulse delivered from ignition to time ''t'' [N s]. """
        if t<= 0.0:
            return 0.0
        if t>= self.burn_time:
            return self.total_impulse
        # Index of the last data point at or before t.
        k = np.searchsorted(self.times, t, side="right") - 1
        # Completed segments, plus the partial trapezoid from times [k] to t.
        partial = 0.5 * (self.thrusts[k] + self.thrust(t)) * (t - self.times[k])
        return float(self.cumulative_impulse[k] + partial)

    def mass(self,t):
        """ Motor mass at any time 't' [kg], assuming propellant burns
        in proportion to the impulse delivered"""
        burned_fraction = self.impulse(t)/self.total_impulse
        return self.total_mass - self.propellant_mass*burned_fraction

def load_eng(path):
    """Read a motor from a RASP ``.eng`` thrust-curve file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the ``.eng`` file.

    Returns
    -------
    Motor
        The motor described by the file.

    Raises
    ------
    ValueError
        If the header is missing or incomplete, or the data are invalid.
    """
    header = None
    times = []
    thrusts = []

    with open(path) as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith (";"):
                continue # skip blank lines and comments
            parts = line.split()
            if header is None:
                header = parts # the first real line is the header
            else:
                times.append(float(parts[0]))
                thrusts.append(float(parts[1]))
    if header is None or len(header) < 6:
        raise ValueError(f"{path}: missing or incomplete header line. ")

    name = header[0]
    diameter = float(header[1]) / 1000.0  # the header gives millimetres
    length = float(header[2]) / 1000.0
    propellant_mass= float(header[4])
    total_mass = float(header[5])
    return Motor(name, times, thrusts, propellant_mass, total_mass, diameter, length)

    
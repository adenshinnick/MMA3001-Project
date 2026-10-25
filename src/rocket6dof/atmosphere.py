"""International Standard Atmosphere (ISA) Model
Gives air temperature, pressure, density and speed of sound as functions of altitude, from just below sea level to 20km. 
The simulator uses the density for aerodynamic forces and the speed of sound for the Mach number. 
"""

import math

#ISA constants
T0 = 288.15          # sea-level temperature [K]
P0 = 101325.0        # sea-level pressure [Pa]
G0 = 9.80665         # standard gravity [m/s^2]
R_AIR = 287.05287    # specific gas constant for dry air [J/(kg K)]
GAMMA = 1.4          # ratio of specific heats for air [-]
LAPSE_RATE = 0.0065  # temperature drop per metre in the troposphere [K/m]

H_TROPOPAUSE = 11000.0  # boundary between troposphere and stratosphere [m]
H_MIN = -610.0          # lowest altitude in the ISA tables [m]
H_MAX = 20000.0         # top of the layers modelled here [m]

#conditions at the tropopause, from the troposphere equations
T_tropopause = T0- LAPSE_RATE*H_TROPOPAUSE
P_tropopause = P0*(T_tropopause/T0)**(G0/(LAPSE_RATE*R_AIR))

def isa(h):
    """Return ISA air properties at a given altitude
    Parameters
    ----------
    h : float
        Altitude above mean sea level [m]. Must be between H_MIN and H_MAX. 
        Treated as geopotential altitude, below 11km this differs from geometric 
        altitude by under 20m, which changes density by about 0.2%.
    
    Returns
    -------
    T : float
        Temperature [K].
    p : float
        Pressure [Pa].
    rho : float
        Density [kg/m^3].
    a : float
        Speed of sound [m/s].

    Raises
    ------
    ValueError
        If ``h`` is outside the modelled range, or is NaN.
    """

    if not H_MIN <= h <= H_MAX:
        raise ValueError(
            f"altitude {h} m is outside the modelled range "
            f"{H_MIN} to {H_MAX} m"
        )

    if h<= H_TROPOPAUSE:
        #troposphere: temperature falls linearly with height
        T= T0- LAPSE_RATE*h
        p = P0*(T/T0) ** (G0/(LAPSE_RATE*R_AIR))
    else:
        #Lower stratosphere constant temperature, so pressure falls
        #exponentially with height
        T= T_tropopause
        p = P_tropopause * math.exp(-G0 * (h-H_TROPOPAUSE)/(R_AIR*T))

    rho =p/(R_AIR*T)     #Ideal Gas Law
    a = math.sqrt(GAMMA*R_AIR*T)  # Speed of Sound

    return T, p, rho, a

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ---------------------
# Parameterizations
# ---------------------

def alpha(T, P):
    return 3.622e-5 * np.exp(-2.377e-5 * T - 0.0106 * P)

def density(T, P):
    return 2870. - 0.082 * T + 162. * (P**0.58)

def depths(P,T):
    return P / (density(T, P)*g*1e-6)

def C_p(T, P):
    return 627. + 0.411 * T - 0.211 * P

def latent_heat_fusion(P):
    # Linear interpolation between given values
    P0, P1 = 0.0, 136.0
    H0, H1 = 6e5, 9e6  # J/kg
    return H0 + (H1 - H0)*((P - P0)/(P1 - P0))

# Knots (depth in km)
z_knots = np.array([0.0, 410.0, 670.0, 2900.0])

g = 9.81

# Solidus parameters
T_s_knots = np.array([1273.0, 2323.0, 2473.0, 3985.0])  # Kelvin
b_s_knots = np.array([-1e-3, -7e-3, -1e-3, 3e-4])       # K/km^2

# Liquidus parameters
T_l_knots = np.array([1973.0, 2423.0, 2723.0, 5375.0]) # Kelvin
b_l_knots = np.array([-1e-3, -7e-3, -2.5e-3, 5e-4])    # K/km^2

def H(x):
    """Heaviside step function."""
    return np.where(x >= 0, 1.0, 0.0)

def S(Ti, Tip1, bi, bip1, zi, zip1, z):
    """
    Cubic spline segment function S as defined in the text.
    """
    hi = zip1 - zi
    term1 = (bip1 * (z - zi)**3) / (6.0 * hi)
    term2 = (bi * (zip1 - z)**3) / (6.0 * hi)
    term3 = ( (Tip1 / hi) - (bip1 * hi / 6.0) ) * (z - zi)
    term4 = ( (Ti / hi) - (bi * hi / 6.0) ) * (zip1 - z)
    return term1 + term2 + term3 + term4

def solidus(z):
    """
    Solidus temperature Ts(z) defined by cubic splines.
    z in km, Ts in Kelvin.
    """
    Ts_val = np.zeros_like(z)  # Initialize array of zeros with same shape as input
    for i in range(3):
        zi = z_knots[i]
        zip1 = z_knots[i+1]
        mask = (z >= zi) & (z < zip1)  # Create boolean mask for this interval
        if np.any(mask):  # If any points fall in this interval
            Ts_val[mask] = S(T_s_knots[i], T_s_knots[i+1], 
                           b_s_knots[i], b_s_knots[i+1], 
                           zi, zip1, z[mask])
    return Ts_val

def liquidus(z):
    """
    Liquidus temperature Tl(z) defined by cubic splines.
    z in km, Tl in Kelvin.
    """
    Tl_val = np.zeros_like(z)
    for i in range(3):
        zi = z_knots[i]
        zip1 = z_knots[i+1]
        mask = (z >= zi) & (z < zip1)
        if np.any(mask):
            Tl_val[mask] = S(T_l_knots[i], T_l_knots[i+1], 
                           b_l_knots[i], b_l_knots[i+1], 
                           zi, zip1, z[mask])
    return Tl_val

def melt_fraction(T, P):
    Ts = solidus(depths(P,T))
    Tl = liquidus(depths(P,T))
    if T < Ts:
        return 0.0
    elif T > Tl:
        return 1.0
    else:
        return (T - Ts)/(Tl - Ts)

def dphi_dP_isentropic(T, P):
    """Calculate (∂φ/∂P)s more accurately by considering the change in melt fraction
    along an isentropic path."""
    dP = 1e-4  # small pressure increment
    
    # Calculate current temperature gradient
    dT = dT_dP(P, T, include_latent=False) * dP
    
    # Calculate melt fraction at current and next point along isentrope
    phi1 = melt_fraction(T, P)
    phi2 = melt_fraction(T + dT, P + dP)
    
    return (phi2 - phi1) / dP

def dT_dP(P, T, include_latent=True):
    """Calculate (∂T/∂P)s according to equations (5) and (9)"""
    # Calculate basic isentropic gradient (equation 5)
    a = alpha(T, P)
    rho = density(T, P)
    cp = C_p(T, P)
    dT_dP_isentropic_zero = (a * T) / (rho * cp)   

    # Check if we're in the partially molten region
    Ts = solidus(depths(P,T))
    Tl = liquidus(depths(P,T))

    if include_latent and T > Ts and T < Tl:
        # Add latent heat term (equation 9)
        Hf = latent_heat_fusion(P)
        dphidP = dphi_dP_isentropic(T, P)
        return dT_dP_isentropic_zero + (Hf/cp)*dphidP
    else:
        return dT_dP_isentropic_zero

# ---------------------
# Function to compute the adiabat for a given surface T
# ---------------------
def compute_adiabat(T_surface, P_start=0.0, P_end=136.0, n_points=200):
    P_eval = np.linspace(P_start, P_end, n_points)
    sol = solve_ivp(lambda P, T: dT_dP(P, T),
                    [P_start, P_end],
                    [T_surface],
                    t_eval=P_eval,
                    method='RK45',
                    rtol=1e-8,  # Add tighter tolerances
                    atol=1e-8,
                    vectorized=True)
    return sol.t, sol.y[0]

# ---------------------
# Time evolution scenario
# ---------------------
# Let's assume a simple scenario:
# The initial surface temperature at t=0 is high (e.g. 2200 K),
# and it decreases linearly over time to 1800 K after some arbitrary "time".
# We'll just pick a few snapshots in "time" to illustrate.

times = np.linspace(0, 1e6, 10)  # 5 time snapshots (in arbitrary units, say years)
T_initial = 5000.0
T_final = 1500.0
def surface_temperature_at_time(t):
    # Simple linear cooling:
    return T_initial + (T_final - T_initial)*(t - times[0])/(times[-1] - times[0])

fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

for t in times:
    T_surf = surface_temperature_at_time(t)
    P_profile, T_profile = compute_adiabat(T_surf)
    z_profile = depths(P_profile,T_profile)
    ax.plot(T_profile, z_profile, label=f'time={t:.2e} yrs')

ax.plot(solidus(z_profile), z_profile, color = 'black')
ax.plot(liquidus(z_profile), z_profile, color = 'black', ls='--')

ax2.plot(solidus(depths(P_profile,T_profile)),depths(P_profile,T_profile), color = 'black')
ax2.plot(liquidus(depths(P_profile,T_profile)),depths(P_profile,T_profile), color = 'black', ls='--')

ax.set_xlabel('Temperature (K)')
ax.invert_yaxis()
ax.set_xlim(1000,5000)
ax2.invert_yaxis()
#ax.legend()
ax.grid(True)
plt.show()

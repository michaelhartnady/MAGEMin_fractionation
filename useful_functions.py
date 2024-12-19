import numpy as np



#% --------------Mantle Solidus and Liquidus from Korenega et al. 2024-------------------------------------------------

# Knots (depth in km)
z_knots = np.array([0.0, 410.0, 670.0, 2900.0])

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

#% ---------------------------------------------------------------


def calculate_adiabatic_profile(depth_array, T_surface=2500, g=9.81, 
                              alpha=3e-5, Cp=1200, rho=3300):
    """
    Calculate adiabatic temperature profile in a magma ocean
    
    Parameters:
    -----------
    depth_array : array
        Array of depths in meters
    T_surface : float
        Surface temperature in Kelvin (default 2773.15 K = 2500°C)
    g : float
        Gravitational acceleration in m/s^2
    alpha : float
        Thermal expansion coefficient in K^-1
    Cp : float
        Specific heat capacity in J/kg/K
    rho : float
        Density in kg/m^3
        
    Returns:
    --------
    array
        Temperature profile (K) at each depth point
    """
    # Calculate adiabatic gradient (dT/dz)
    T_surface_K = T_surface
    dT_dz = (alpha * g * T_surface_K) / Cp
    
    # Calculate temperature at each depth point
    # T(z) = T_surface * exp(alpha * g * z / Cp)
    T_profile = T_surface_K * np.exp((alpha * g * depth_array) / Cp)
    
    return T_profile-273.15

def plot_adiabat(depth_array, T_profile):
    """
    Plot the adiabatic temperature profile
    
    Parameters:
    -----------
    depth_array : array
        Array of depths in meters
    T_profile : array
        Array of temperatures in Kelvin
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 6))
    plt.plot(T_profile, depth_array/1000, 'b-')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Depth (km)')
    plt.title('Magma Ocean Adiabatic Profile')
    plt.gca().invert_yaxis()  # Invert y-axis to show depth increasing downward
    plt.grid(True)
    plt.show()

def cool_adiabatic_profile(depth_array, T_profile, time_steps, cooling_rate_surface=30):
    """
    Model the cooling of an adiabatic profile with depth-dependent cooling rate
    
    Parameters:
    -----------
    depth_array : array
        Array of depths in meters
    T_profile : array
        Initial temperature profile in Kelvin
    time_steps : int
        Number of time steps to evolve
    cooling_rate_surface : float
        Cooling rate at the surface in K/timestep
        
    Returns:
    --------
    array
        Time evolution of temperature profiles (time_steps x depths)
    """
    # Initialize array to store temperature evolution
    T_evolution = np.zeros((time_steps, len(depth_array)))
    T_evolution[0] = T_profile
    
    # Calculate depth-dependent cooling rate
    # Cooling rate decreases exponentially with depth
    depth_factor = np.exp(-depth_array / 500e3)  # characteristic depth of 500 km
    cooling_rates = cooling_rate_surface * depth_factor
    
    # Evolution through time
    for t in range(1, time_steps):
        T_evolution[t] = T_evolution[t-1] - cooling_rates
        
        # Ensure temperatures don't go below reasonable values (e.g., 1000 K)
        T_evolution[t] = np.maximum(T_evolution[t], 1000)
    
    return T_evolution

def plot_cooling_evolution(depth_array, T_evolution, time_steps_to_plot=None):
    """
    Plot the evolution of the temperature profile during cooling
    
    Parameters:
    -----------
    depth_array : array
        Array of depths in meters
    T_evolution : array
        Time evolution of temperature profiles
    time_steps_to_plot : list, optional
        List of time steps to plot (default plots every 20% of total time)
    """
    import matplotlib.pyplot as plt
    
    if time_steps_to_plot is None:
        time_steps_to_plot = np.linspace(0, len(T_evolution)-1, 6).astype(int)
    
    plt.figure(figsize=(10, 8))
    for t in time_steps_to_plot:
        plt.plot(T_evolution[t], depth_array/1000, 
                label=f'Time step {t}')
    
    plt.xlabel('Temperature (K)')
    plt.ylabel('Depth (km)')
    plt.title('Cooling Evolution of Magma Ocean')
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.legend()
    plt.show()

# Example usage:
depths = np.linspace(0, 1000e3, 1000)  # 0 to 1000 km depth
initial_temps = calculate_adiabatic_profile(depths)
T_evolution = cool_adiabatic_profile(depths, initial_temps, time_steps=100)
plot_cooling_evolution(depths, T_evolution)
import numpy as np
import matplotlib.pyplot as plt

def latent_heat_fusion(P):
    # Linear interpolation between given values
    P0, P1 = 0.0, 136.0
    H0, H1 = 6e5, 9e6  # J/kg
    return H0 + (H1 - H0)*((P - P0)/(P1 - P0))
    #return 4e6

def calculate_optimal_dp(T, P):
    """
    Calculate optimal pressure step size based on characteristic scales
    
    Returns:
        dP: optimal pressure step in GPa
    """
    # Get current properties
    alpha, rho, Cp = calculate_material_properties(T, P)
    z = pressure_to_depth(P)
    Ts, Tl = calculate_phase_temperatures(z)
    
    # Calculate characteristic scales
    V_characteristic = 1e9 * (alpha * T) / (rho * Cp)  # Adiabatic gradient (K/GPa)
    
    if T > Ts and T < Tl:
        # In two-phase region, consider latent heat effects
        Hf = latent_heat_fusion(P)
        L_characteristic = (Tl - Ts)  # Temperature difference across phase transition
        
        # CFL condition: dP ≤ L_characteristic / max(V_characteristic, Hf/(rho*Cp))
        dP = 0.1 * min(L_characteristic / abs(V_characteristic), 
                       L_characteristic / abs(Hf/(rho*Cp)))
    else:
        # Outside two-phase region, only consider adiabatic gradient
        L_characteristic = 100  # Typical temperature change over 1 GPa
        dP = 0.1 * L_characteristic / abs(V_characteristic)
    
    # Ensure dP doesn't get too small or too large
    dP = max(min(dP, 0.1), 0.01)
    
    return dP

def pressure_to_depth(P):
    """
    Convert pressure to depth using P = ρgz
    
    Args:
        P: Pressure in GPa
    Returns:
        z: Depth in km
    
    Note: 
        - P is in GPa (10^9 Pa)
        - g is ~9.81 m/s^2
        - ρ is ~3300 kg/m^3 (average mantle density)
        - Need to convert final depth to km
    """
    g = 9.81  # m/s^2
    rho_avg = 3300  # kg/m^3
    
    # Convert GPa to Pa (multiply by 10^9)
    P_pa = P * 1e9
    
    # Calculate depth in meters
    z_m = P_pa / (rho_avg * g)
    
    # Convert to km
    z_km = z_m / 1000
    
    return z_km

def depth_to_pressure(z):
    """
    Convert depth to pressure using P = ρgz
    
    Args:
        z: Depth in km
    Returns:
        P: Pressure in GPa
    
    Note:
        - z is in km (need to convert to m)
        - g is ~9.81 m/s^2
        - ρ is ~3300 kg/m^3 (average mantle density)
        - Need to convert final pressure to GPa
    """
    g = 9.81  # m/s^2
    rho_avg = 3300  # kg/m^3
    
    # Convert km to m
    z_m = z * 1000
    
    # Calculate pressure in Pa
    P_pa = rho_avg * g * z_m
    
    # Convert to GPa
    P_gpa = P_pa / 1e9
    
    return P_gpa

def calculate_phase_temperatures(z):
    """
    Calculate solidus and liquidus temperatures at given depth z (in km)
    using cubic spline interpolation as defined in equations (1) and (2)
    """
    # Knot points (depths in km)
    z_knots = np.array([0, 410, 670, 2900])
    
    # Temperature values at knots for solidus (K)
    # Corrected from paper: T_s^i = {1273, 2323, 2473, 3985}
    T_s = np.array([1273, 2323, 2473, 3985])
    # Temperature values at knots for liquidus (K)
    # Corrected from paper: T_l^i = {1973, 2423, 2723, 5375}
    T_l = np.array([1973, 2423, 2723, 5375])
    
    # Second derivatives at knots (bi values)
    # Corrected from paper: b_s^i = {-10^-3, -7×10^-3, -10^-3, 3×10^-4}
    b_s = np.array([-1e-3, -7e-3, -1e-3, 3e-4])
    # Corrected from paper: b_l^i = {-10^-3, -7×10^-3, -2.5×10^-3, 5×10^-4}
    b_l = np.array([-1e-3, -7e-3, -2.5e-3, 5e-4])
    
    # Find the appropriate interval or use end intervals for extrapolation
    if z <= z_knots[0]:
        # Use first interval for extrapolation
        i = 0
    elif z >= z_knots[-1]:
        # Use last interval for extrapolation
        i = len(z_knots) - 2
    else:
        # Find the appropriate interval for interpolation
        for i in range(len(z_knots)-1):
            if z_knots[i] <= z <= z_knots[i+1]:
                break
    
    # Calculate cubic spline for solidus
    T_s_z = cubic_spline(z, z_knots[i], z_knots[i+1], 
                        T_s[i], T_s[i+1], b_s[i], b_s[i+1])
    
    # Calculate cubic spline for liquidus
    T_l_z = cubic_spline(z, z_knots[i], z_knots[i+1],
                        T_l[i], T_l[i+1], b_l[i], b_l[i+1])
    
    return T_s_z, T_l_z



def cubic_spline(z, zi, zi1, Ti, Ti1, bi, bi1):
    """
    Implement the cubic spline function as defined in equation (3)
    """
    hi = zi1 - zi
    term1 = (bi1 * (z - zi)**3) / (6 * hi)
    term2 = (bi * (zi1 - z)**3) / (6 * hi)
    term3 = ((Ti1/hi) - (bi1*hi)/6) * (z - zi)
    term4 = ((Ti/hi) - (bi*hi)/6) * (zi1 - z)
    
    return term1 + term2 + term3 + term4

def calculate_melt_fraction(z):
    """
    Calculate f_40 as described in the paper
    Linear interpolation between specified points
    """
    if z < 0 or z > 2900:
        return None
    
    # Define interpolation points
    depths = [0, 410, 670, 2900]
    f_values = [0.4, 0.4, 0.6, 0.845]
    
    # Linear interpolation
    for i in range(len(depths)-1):
        if depths[i] <= z <= depths[i+1]:
            f = f_values[i] + (f_values[i+1] - f_values[i]) * \
                (z - depths[i]) / (depths[i+1] - depths[i])
            return f
    
    return None

def melt_fraction(T, P):
    """
    Calculate melt fraction with piecewise linear distribution:
    - Linear from 0 to 0.4 between Ts and T40
    - Linear from 0.4 to 1.0 between T40 and Tl
    """
    z = pressure_to_depth(P)
    Ts, Tl = calculate_phase_temperatures(z)
    T40 = calculate_T40(z)
    
    # Check if any of our values are None
    if T40 is None or Ts is None or Tl is None:
        return 0.0  # or handle this case as appropriate
    
    if T <= Ts:
        return 0.0
    elif T >= Tl:
        return 1.0
    elif T <= T40:
        # Linear interpolation between Ts (f=0) and T40 (f=0.4)
        return 0.4 * (T - Ts) / (T40 - Ts)
    else:
        # Linear interpolation between T40 (f=0.4) and Tl (f=1.0)
        return 0.4 + 0.6 * (T - T40) / (Tl - T40)

def calculate_material_properties(T, P, d_rho_frac = 0.015):
    """
    Calculate material properties α, ρ, and Cp as functions of T and P
    using equations (6), (7), and (8)
    
    Args:
        T: Temperature in Kelvin
        P: Pressure in GPa
    Returns:
        α: Thermal expansivity in K^-1
        ρ: Density in kg/m^3
        Cp: Specific heat in J/kg/K
    """
    
    Ts,Tl = calculate_phase_temperatures(pressure_to_depth(P))
    Hf = latent_heat_fusion(P)
    if T > Ts and T < Tl:
        alpha = (3.622e-5 * np.exp(-2.377e-5 * T - 0.0106 * P)) #+ d_rho_frac * (1/(Tl-Ts))
        Cp = (627 + 0.411 * T - 0.211 * P) #+ Hf/(Tl-Ts)
        print(f"T: {T : .0f}, P: {P : .1f}, alpha: {alpha: .6f}, Cp: {Cp: .3f}")
    else:
        alpha = 3.622e-5 * np.exp(-2.377e-5 * T - 0.0106 * P)
        Cp = 627 + 0.411 * T - 0.211 * P
        print(f"T: {T : .0f}, P: {P : .1f}, alpha: {alpha: .6f}, Cp: {Cp: .3f}")

    # Equation (7)
    rho = 2870 - 0.082 * T + 162 * P**0.58

    #print(f"T: {T : .0f}, P: {P : .1f}, alpha: {alpha: .6f}, rho: {rho: .2f}, Cp: {Cp: .2f}")
    return alpha, rho, Cp

def calculate_adiabatic_gradient(T, P):
    """
    Calculate the adiabatic temperature gradient (dT/dP)_S
    using equation (5)
    
    Args:
        T: Temperature in Kelvin
        P: Pressure in GPa
    Returns:
        dT/dP in K/GPa
    """
    z = pressure_to_depth(P)
    Ts,Tl = calculate_phase_temperatures(z)
    alpha, rho, Cp = calculate_material_properties(T, P)
    
    # Base adiabatic term
    adiabatic_term = 1e9 * (alpha * T) / (rho * Cp)
    
    # Only add latent heat effect if we're actually in the two-phase region
    if T >= Ts and T <= Tl:  # Removed buffer, use inclusive bounds
        dP = 0.01  # Use smaller step for derivative calculation
        dT = adiabatic_term * dP
        print(f"gradient : {dT/dP * (1/10):.2f}")
        # Calculate melt fraction change
        phi1 = melt_fraction(T, P)
        phi2 = melt_fraction(T, P + dP)
        dF_dP = (phi2 - phi1) / dP
        
        Hf = latent_heat_fusion(P)
        
        # Scale down latent heat term significantly
        scaling_factor = 0.112
        latent_term = -(Hf/Cp) * dF_dP * scaling_factor
        
        #print(f"At T={T:.1f}, P={P:.1f}:")
        #print(f"  Ts={Ts:.1f}, Tl={Tl:.1f}")
        #print(f"  Adiabatic term: {adiabatic_term:.2f} K/GPa")
        #print(f"  Latent term: {latent_term:.2f} K/GPa")
        
        return adiabatic_term + latent_term
    
    return adiabatic_term

def calculate_adiabat(T_surface, P_range, dP=0.1):
    """
    Calculate an adiabatic temperature profile starting from a surface temperature
    
    Args:
        T_surface: Surface temperature in Kelvin
        P_range: Tuple of (min_pressure, max_pressure) in GPa
        dP: Pressure step size in GPa
    Returns:
        P_points: Array of pressure points
        T_points: Array of temperature points
    """
    P_min, P_max = P_range
    P_points = [P_min]
    T_points = [T_surface]
    
    while P_points[-1] < P_max:
        P = P_points[-1]
        T = T_points[-1]
        
        # Use smaller steps near phase boundaries
        z = pressure_to_depth(P)
        Ts, Tl = calculate_phase_temperatures(z)
        if abs(T - Ts) < 50 or abs(T - Tl) < 50:  # Near phase boundary
            current_dP = 0.01
        else:
            current_dP = 0.1 if dP is None else dP
            
        # Ensure we don't overshoot
        if P + current_dP > P_max:
            current_dP = P_max - P
            
        # RK4 integration
        k1 = calculate_adiabatic_gradient(T, P)
        k2 = calculate_adiabatic_gradient(T + 0.5*current_dP*k1, P + 0.5*current_dP)
        k3 = calculate_adiabatic_gradient(T + 0.5*current_dP*k2, P + 0.5*current_dP)
        k4 = calculate_adiabatic_gradient(T + current_dP*k3, P + current_dP)
        
        dT = (current_dP/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        P_points.append(P + current_dP)
        T_points.append(T + dT)
    
    return np.array(P_points), np.array(T_points)



# Test at surface
print(f"Surface (0 km): Ts={calculate_phase_temperatures(0)[0]}, Tl={calculate_phase_temperatures(0)[1]}")
# Test at 410 km
print(f"410 km: Ts={calculate_phase_temperatures(410)[0]}, Tl={calculate_phase_temperatures(410)[1]}")
# Test at 670 km
print(f"670 km: Ts={calculate_phase_temperatures(670)[0]}, Tl={calculate_phase_temperatures(670)[1]}")
# Test at 2900 km
print(f"2900 km: Ts={calculate_phase_temperatures(2900)[0]}, Tl={calculate_phase_temperatures(2900)[1]}")

def calculate_T40(z):
    """
    Calculate T40 temperature as defined in equation (4)
    """
    T_s, T_l = calculate_phase_temperatures(z)
    f40 = calculate_melt_fraction(z)
    
    if T_s is None or T_l is None or f40 is None:
        return None
    
    return T_s + f40 * (T_l - T_s)

def plot_phase_diagram(depth_range=(0, 1500), temp_range=(1000, 4000)):
    """
    Create a phase diagram plot similar to the paper figure
    """
    # Create depth points
    depths = np.linspace(depth_range[0], depth_range[1], 1000)
    
    # Calculate corresponding pressures
    pressures = depth_to_pressure(depths)
    
    # Calculate solidus and liquidus
    Ts_values = []
    Tl_values = []
    T40_values = []
    for z in depths:
        Ts, Tl = calculate_phase_temperatures(z)
        T40 = calculate_T40(z)
        Ts_values.append(Ts)
        Tl_values.append(Tl)
        T40_values.append(T40)

    # Convert lists to numpy arrays
    Ts_values = np.array(Ts_values)
    Tl_values = np.array(Tl_values)
    T40_values = np.array(T40_values)

    # Add adiabats for different surface temperatures
    surface_temperatures = [1300,1500,1700,1900,2100,2300,2500,2700,2900]
    colors = ['purple', 'blue', 'cyan', 'green', 'yellowgreen', 
              'yellow', 'orange', 'red']
    P_range = (0, 100)  # Pressure range in GPa

    # Create plot
    figure = plt.figure(figsize=(6, 6))
    ax1 = figure.add_subplot(111)
    
    # Create second y-axis
    ax2 = ax1.twinx()
    
    # Plot adiabats
    for T_surf, c in zip(surface_temperatures, colors):
        P_points, T_points = calculate_adiabat(T_surf, P_range)
        adiabat_depths = pressure_to_depth(P_points)
        ax1.plot(T_points, adiabat_depths, '-', color=c, alpha=0.5)
    
    # Plot phase boundaries
    ax1.plot(Ts_values, depths, 'k-', label='solidus')
    ax1.plot(Tl_values, depths, 'k-', label='liquidus')
    ax1.plot(T40_values, depths, '--', color='gray', label='T40')

    # Customize primary axis (depths)
    ax1.invert_yaxis()  # Depth increases downward
    ax1.set_xlim(temp_range)
    ax1.set_ylabel('Depth [km]')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Calculate pressure ticks based on depth ticks
    depth_ticks = ax1.get_yticks()
    pressure_ticks = depth_to_pressure(depth_ticks)


    # Move temperature labels to top
    ax3 = ax1.twiny()
    ax3.set_xlim(ax1.get_xlim())
    ax3.set_xlabel('Temperature [K]')
    
    # Remove bottom temperature labels
    ax1.set_xlabel('')
    ax1.xaxis.set_ticks_position('top')
    ax1.xaxis.set_label_position('top')
    
    # Add legend
    ax1.legend(loc='lower right')
    
    # Adjust layout to prevent label overlap
    #plt.tight_layout()

    return figure

def main():
    # Example usage
    # Calculate temperatures at a specific depth
    depth = 500  # depth in km
    T_s, T_l = calculate_phase_temperatures(depth)
    T40 = calculate_T40(depth)
    print(f"At depth {depth} km:")
    print(f"Solidus temperature: {T_s:.1f} K")
    print(f"Liquidus temperature: {T_l:.1f} K")
    print(f"T40 temperature: {T40:.1f} K")
    
    # Create and show the phase diagram
    fig = plot_phase_diagram()
    plt.show()

if __name__ == "__main__":
    main()

import numpy as np
from molmass import Formula
import pandas as pd
import matplotlib.pyplot as plt

import juliacall
from juliacall import Main as jl, convert as jlconvert

# Explicitly add MAGEMin_C to the current environment
#jl.seval('import Pkg; Pkg.add("MAGEMin_C")') run this line to install the package in your local julia virtual environment
jl.seval('using MAGEMin_C')
# Create the module reference
MAGEMin_C = jl.MAGEMin_C

# ----------- Define all System Parameters-----------

data    = MAGEMin_C.Initialize_MAGEMin("ig", verbose=False)
P,T     = 15.0, 1600.0
oxides = ['SiO2', 'Al2O3', 'CaO',  'MgO', 'FeO', 'K2O', 'Na2O', 'TiO2', 'O' , 'Cr2O3' , 'H2O']
values = [39.7918,1.999,2.9166, 49.2832, 5.5007, 0.0055, 0.0418, 0.1104, 0.0825, 0.1706,  0.0]
Xoxides = jlconvert(jl.Vector[jl.String], oxides)
X       = jlconvert(jl.Vector[jl.Float64], values)
sys_in  = "mol"
out     = MAGEMin_C.single_point_minimization(P, T, data, X=X, Xoxides=Xoxides, sys_in=sys_in)

# ----------- Functions for calculating Adiabats-----------

def composition_to_molarMass(composition, oxides=oxides):
    model_composition = dict(zip(oxides,composition)) # oxides, wtfractions 
    total_moles = sum((val / Formula(ox).mass) for ox, val  in model_composition.items())
    Molar_mass = 1 / total_moles
    return Molar_mass

def check_melt_frac(out):
    if 'liq' in out.ph:
        idx = out.ph.index('liq')
        return out.ph_frac[idx]
    else:
        return 0
    
def get_alpha(out):
    alpha = out.alpha
    return alpha

def get_density(out):
    rho = out.rho
    return rho

def get_Cp(out):
    ''' Converts heat capacity in kJ/K to J/kg/K assuming total mass is 100g'''
    Cp = out.cp
    return (Cp * 1e3) / 0.1

def get_latent_heat(out):
    ''' Calculates latent heat of fusion in J/kg'''
    model_composition = dict(zip(out.oxides,out.bulk)) # oxides, mole fractions   
    molar_mass_g = out.M_sys
    total_moles = sum((100*val / Formula(ox).mass) for ox, val  in model_composition.items())
    melt = []
    solids = []

    if check_melt_frac(out) > 0:
        print(f"Melt present, calculating H & dF/dP")

        for phase, type in zip(out.ph, out.ph_type):
            if type == 1:
                for index in range(out.n_SS): 
                    if phase == 'liq':
                        melt.append(out.SS_vec[index].enthalpy)
                    else:
                        solids.append(out.SS_vec[index].enthalpy)

            elif type == 0:
                for index in range(out.n_PP):
                    solids.append(out.PP_vec[index].enthalpy)
            else:
                pass
        else:
            melt.append(0)
            solids.append(0)
            
    enthalpy_melt = sum(melt)
    enthalpy_solids = sum(solids)
    print(f"Enthalpy melt : {enthalpy_melt}, Enthalpy solids : {enthalpy_solids}, H_fusion : {enthalpy_melt - enthalpy_solids}")
    return (enthalpy_melt - enthalpy_solids) / 0.1


def calculate_material_properties(P, T):
    """
    Args:
        T: Temperature in Kelvin
        P: Pressure in GPa
    Returns:
        α: Thermal expansivity in K^-1
        ρ: Density in kg/m^3
        Cp: Specific heat in J/kg/K
    """

    print(f"Calculating material properties at {P: .2f} Kbar, {T: .1f} C")
    out     = MAGEMin_C.single_point_minimization(jlconvert(jl.Float64, P), jlconvert(jl.Float64, T-273.15), data, X=X, Xoxides=Xoxides, sys_in=sys_in)
    alpha = get_alpha(out)
    rho = get_density(out)
    Cp = get_Cp(out)
    melt_frac = check_melt_frac(out)
    latent_heat = get_latent_heat(out)
    return alpha, rho, Cp, melt_frac, latent_heat

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
    #z = pressure_to_depth(P)
    alpha, rho, Cp, melt_frac, latent_heat = calculate_material_properties(P, T)
    
    # Base adiabatic term 1e8 converts GPa to kbar
    adiabatic_term = 1e8 * (alpha * T) / (rho * Cp)
    # Only add latent heat effect if we're actually in the two-phase region
    if melt_frac > 0: 
        print(f"Melt present at {P: .2f} Kbar, {T: .1f} C, calculating H & dF/dP")
        dP = 0.1  # Use smaller step for derivative calculation
        dT = adiabatic_term * dP

        out_dP = MAGEMin_C.single_point_minimization(jlconvert(jl.Float64, P + dP), jlconvert(jl.Float64, T-273.15), data, X=X, Xoxides=Xoxides, sys_in=sys_in)
       #print(f"gradient : {dT/dP * (1/10):.2f}")
        # Calculate melt fraction change
        F_p = melt_frac
        F_dp = check_melt_frac(out_dP)
        dF_dP = (F_dp - F_p) / dP
        
        print(latent_heat)
        Hf = latent_heat
        
        # Scale down latent heat term significantly
        scaling_factor = 1
        latent_term = -(Hf/Cp) * dF_dP * scaling_factor
        
        #print(f"At T={T:.1f}, P={P:.1f}:")
        #print(f"  Adiabatic term: {adiabatic_term:.2f} K/GPa")
        
        print(f"Adiabatic term : {adiabatic_term:.2f} K/kbar")
        print(f"  Latent term: {latent_term:.2f} K/kbar")
        return adiabatic_term + latent_term
    
    return adiabatic_term

def calculate_adiabat(T_surface, P_range, dP=0.5):
    """
    Calculate an adiabatic temperature profile starting from a surface temperature
    
    Args:
        T_surface: Surface temperature in Kelvin
        P_range: Tuple of (min_pressure, max_pressure) in GPa
        dP: Pressure step size in Kbar
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
        #if abs(T - Ts) < 50 or abs(T - Tl) < 50:  # Near phase boundary
            #current_dP = 0.01
        #else:
         #   current_dP = 0.1 if dP is None else dP
            
        # Ensure we don't overshoot
        current_dP = dP
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


# ----------- Functions for finding phase boundaries-----------

def initial_search(P_array, T_array, data, X, Xoxides, sys_in):
    print(f"Searching at P : {P_array[0]} Kbar, between {T_array[0]} and {T_array[-1]} C")
    out_m = MAGEMin_C.multi_point_minimization(P_array, T_array, data, X=X, Xoxides=Xoxides, sys_in=sys_in)
    solidus_T_interval = None
    liquidus_T_interval = None
    
    # Track if we've seen any melt
    seen_melt = False
    
    for i in range(len(out_m)):
        melt_frac = check_melt_frac(out_m[i])
        
        # Update solidus interval logic
        if not seen_melt and melt_frac > 0:
            solidus_T_interval = (T_array[i-1], T_array[i])
            seen_melt = True
            
        # Update liquidus interval logic
        if seen_melt and melt_frac >= 1:
            liquidus_T_interval = (T_array[i-1], T_array[i])
            break
    
    # If we've seen melt but haven't found liquidus, check last point
    if seen_melt and liquidus_T_interval is None:
        if check_melt_frac(out_m[-1]) >= 1:
            liquidus_T_interval = (T_array[-2], T_array[-1])
            
    print(f"Found intervals - Solidus: {solidus_T_interval}, Liquidus: {liquidus_T_interval}")
    return solidus_T_interval, liquidus_T_interval

def binary_search_boundary(P, T_low, T_high, condition_func, boundary_name, tolerance=0.1):
    """
    Binary search to find temperature boundary within tolerance.
    
    Args:
        P: Pressure in GPa
        T_low: Lower temperature bound
        T_high: Upper temperature bound
        condition_func: Function that returns True above boundary
        tolerance: Temperature tolerance in K
    
    Returns:
        float: Temperature at boundary
    """
    iteration = 0
    while (T_high - T_low) > tolerance:
        iteration += 1
        #print(f"P : {P}, Boundary : {boundary_name}, iteration : {iteration}, T_low : {T_low}, T_high : {T_high}")
        T_mid = (T_low + T_high) / 2
        if condition_func(T_mid):
            T_high = T_mid
        else:
            T_low = T_mid
            
    return (T_low + T_high) / 2

def find_phase_boundaries(data, X, Xoxides, sys_in, P, T_range=(800, 2500), tolerance=1, n_points_initial=5):
    """
    Find solidus and liquidus temperatures at a given pressure using binary search.
    If return_all_points is True, returns arrays of all T points and their melt fractions.
    
    Args:
        P: Pressure in GPa or array of pressures
        Trange: Tuple of (min_temp, max_temp) in K
        tolerance: Temperature tolerance in K
        return_all_points: If True, returns full T and melt fraction arrays
    
    Returns:
        if return_all_points=False:
            tuple: (solidus_T, liquidus_T) in K, or (None, None) if not found
        if return_all_points=True:
            tuple: (temperatures, melt_fractions, solidus_T, liquidus_T)
    """
    T_array = jlconvert(jl.Vector[jl.Float64], np.linspace(T_range[0], T_range[1], n_points_initial))
    P_array = jlconvert(jl.Vector[jl.Float64], [P for p in range(len(T_array))])

    Ts_int, Tl_int = initial_search(P_array, T_array, data, X, Xoxides, sys_in)
    
    # Binary search for solidus
    solidus_T = binary_search_boundary(
        P, Ts_int[0], Ts_int[1],
        lambda t: check_melt_frac(MAGEMin_C.single_point_minimization(P, t, data, X=X, Xoxides=Xoxides, sys_in=sys_in)) > 0,
        'solidus',
        tolerance
    )
    
    # Binary search for liquidus
    liquidus_T = binary_search_boundary(
        P, Tl_int[0], Tl_int[1],
        lambda t: check_melt_frac(MAGEMin_C.single_point_minimization(P, t, data, X=X, Xoxides=Xoxides, sys_in=sys_in)) >= 1,
        'liquidus',
        tolerance
    )
    
    return solidus_T, liquidus_T


# ----------- Find and Plot Phase Boundaries Adiabats-----------
print(f"Finding phase boundaries")
pressures = np.linspace(0, 50, 10)  # 11 points from 5 to 15 Kbar
solidus_points = []
liquidus_points = []

for P in pressures:
    sol_T, liq_T = find_phase_boundaries(data, X, Xoxides, sys_in,P, T_range=(800, 2500), tolerance=0.1)
    if sol_T is not None:
        solidus_points.append((P, sol_T))
        liquidus_points.append((P, liq_T))

solidus_points = np.array(solidus_points)
liquidus_points = np.array(liquidus_points)

# Create plot
figure = plt.figure(figsize=(6, 6))
ax1 = figure.add_subplot(111)
ax1.plot(solidus_points[:,1], solidus_points[:,0],'b-', label='Solidus')
ax1.plot(liquidus_points[:,1], liquidus_points[:,0], 'r--', label='Liquidus')

# ----------- Plotting Adiabats-----------

# Add adiabats for different surface temperatures
surface_temperatures = [1400+273.15]
colors = ['purple', 'blue', 'cyan', 'green', 'yellowgreen', 
            'yellow', 'orange', 'red']
P_range = (0, 50)  # Pressure range in kbar



# Plot adiabats
for T_surf, c in zip(surface_temperatures, colors):
    P_points, T_points = calculate_adiabat(T_surf, P_range,dP=1)
    ax1.plot(T_points-273.15, P_points, '-', color=c, alpha=0.5)

# Customize primary axis (depths)
ax1.invert_yaxis()  # Depth increases downward
ax1.set_xlim(1000,2200)
ax1.set_ylabel('Pressure [kbar]')
ax1.set_xlabel('Temperature K')
ax1.grid(True, linestyle='--', alpha=0.3)

plt.show()

"""
===============
MAGEMin Adiabats
===============

the MAGEMin_adiabat_class module provides the MAGEMin adiabatic classes. At the moment it consists of
a single adiabatic class- a simple 1D adiabatic class.

"""

import numpy as np
from molmass import Formula
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import juliacall
from juliacall import Main as jl, convert as jlconvert

# Explicitly add MAGEMin_C to the current environment
#jl.seval('import Pkg; Pkg.add("MAGEMin_C")') run this line to install the package in your local julia virtual environment
#jl.seval('using MAGEMin_C')
# Create the module reference
#MAGEMin_C = jl.MAGEMin_C

class mantle_adiabat:
    def __init__(self,Xoxides, X, sys_in = 'mol', dataset = "ig"):
        #initiate MAGEMin_C
        print(f"Initializing MAGEMin_C with the following parameters:")
        print(f"Xoxides : {Xoxides}")
        print(f"X : {X}")
        print(f"sys_in : {sys_in}")
        print(f"dataset : {dataset}")
        jl.seval('using MAGEMin_C')
        self.MAGEMin_C = jl.MAGEMin_C
        self.data = self.MAGEMin_C.Initialize_MAGEMin(dataset, verbose=False)
        self.X = jlconvert(jl.Vector[jl.Float64], X)
        self.Xoxides = jlconvert(jl.Vector[jl.String], Xoxides)
        self.sys_in = sys_in
        self.solidus_points = None
        self.liquidus_points = None
        self.T_solidus = None
        self.T_liquidus = None
        self.adiabat_data = {}
    
    def check_melt_frac(self, out):
        if 'liq' in out.ph:
            idx = out.ph.index('liq')
            return out.ph_frac[idx]
        else:
            return 0
    
    def get_alpha(self, out):
        alpha = out.alpha
        return alpha

    def get_density(self, out):
        rho = out.rho
        return rho

    def get_Cp(self, out):
        ''' Converts heat capacity in kJ/K to J/kg/K assuming total mass is 100g'''
        Cp = out.cp
        return (Cp * 1e3) / 0.1
    
    def get_phases(self, out):
        return out.ph, out.ph_frac

    def get_latent_heat(self, out):
        melt = []
        solids = []

        if self.check_melt_frac(out) > 0:
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

        enthalpy_melt = sum(melt)
        enthalpy_solids = sum(solids)
        print(f"Enthalpy melt : {enthalpy_melt}, Enthalpy solids : {enthalpy_solids}, H_fusion : {enthalpy_melt - enthalpy_solids}")
        return (enthalpy_melt - enthalpy_solids) / 0.1


    def calculate_material_properties(self, out):
        """
        Args:
            out: Output from MAGEMin_C.single_point_minimization
        Returns:
            α: Thermal expansivity in K^-1
            ρ: Density in kg/m^3
            Cp: Specific heat in J/kg/K
        """
        alpha = self.get_alpha(out)
        rho = self.get_density(out)
        Cp = self.get_Cp(out)
        melt_frac = self.check_melt_frac(out)
        latent_heat = self.get_latent_heat(out)
        phases, phases_frac = self.get_phases(out)

        return alpha, rho, Cp, melt_frac, latent_heat, phases, phases_frac

    def calculate_adiabatic_gradient(self, T, P, dP=1, latent_scaling_factor=1):
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
        print(f"Calculating material properties at {P: .2f} Kbar, {T-273.15: .1f} C")
        out = self.MAGEMin_C.single_point_minimization(jlconvert(jl.Float64, P), jlconvert(jl.Float64, T-273.15), self.data, X=self.X, Xoxides=self.Xoxides, sys_in=self.sys_in)
        alpha, rho, Cp, melt_frac, latent_heat, phases, phases_frac = self.calculate_material_properties(out)
        
        
        # Base adiabatic term
        adiabatic_term = 1e8 * (alpha * T) / (rho * Cp) # factor of 1e8 converts Pa to Kbar
        
        # Only add latent heat effect if we're actually in the two-phase region
        properties = {'P': P, 'T': T-273.15, 'dP': dP, 'dT': adiabatic_term*dP,'alpha': alpha, 'rho': rho, 'Cp': Cp, 'Hf': latent_heat, 'F': melt_frac, 'phases': phases, 'phases_frac': phases_frac}
        if melt_frac > 0: 
            print(f"Melt present at {P: .2f} Kbar, {T-273.15: .1f} C, calculating H & dF/dP")
            #dP = 0.1  # Use smaller step for derivative calculation
            dT = adiabatic_term * dP

            out_dP = self.MAGEMin_C.single_point_minimization(jlconvert(jl.Float64, P + dP), jlconvert(jl.Float64, T-273.15), self.data, X=self.X, Xoxides=self.Xoxides, sys_in=self.sys_in)
        #print(f"gradient : {dT/dP * (1/10):.2f}")
            # Calculate melt fraction change
            F_p = melt_frac
            F_dp = self.check_melt_frac(out_dP)
            dF_dP = (F_dp - F_p) / dP
            
            Hf = latent_heat
            
            # Scale down latent heat term significantly
            scaling_factor = latent_scaling_factor
            latent_term = -(Hf/Cp) * dF_dP * scaling_factor
            
            #print(f"At T={T:.1f}, P={P:.1f}:")
            #print(f"  Adiabatic term: {adiabatic_term:.2f} K/GPa")
            #print(f"  Latent term: {latent_term:.2f} K/GPa")
            properties['dP'] = dP
            properties['dT'] = adiabatic_term*dP
            
            return (adiabatic_term + latent_term) , properties
        
        return adiabatic_term, properties

    def calculate_adiabat(self, T_surface, P_range, dP=1, latent_scaling_factor=1):
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
        self.adiabat_data[str(T_surface-273.15)] = {}
        current_dP = dP
        count = 0
        while P_points[-1] < P_max:
            P = P_points[-1]
            T = T_points[-1]
            
            if self.T_solidus is not None or self.T_liquidus is not None:
                # Use smaller steps near phase boundaries
                if abs(T - self.T_solidus(P)) < 10 or abs(T - self.T_liquidus(P)) < 10:  # Near phase boundary
                    current_dP = 0.1
                else:
                    current_dP = 0.5 if dP is None else dP
                    
            # Ensure we don't overshoot
            if P + current_dP > P_max:
                current_dP = P_max - P              
                
            # RK4 integration
            k1,_ = self.calculate_adiabatic_gradient(T, P, dP=current_dP, latent_scaling_factor=latent_scaling_factor)
            k2,_ = self.calculate_adiabatic_gradient(T + 0.5*current_dP*k1, P + 0.5*current_dP, dP=current_dP, latent_scaling_factor=latent_scaling_factor)
            k3,_ = self.calculate_adiabatic_gradient(T + 0.5*current_dP*k2, P + 0.5*current_dP, dP=current_dP, latent_scaling_factor=latent_scaling_factor)
            k4,props = self.calculate_adiabatic_gradient(T + current_dP*k3, P + current_dP, dP=current_dP, latent_scaling_factor=latent_scaling_factor)
            
            dT = (current_dP/6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            self.adiabat_data[str(T_surface-273.15)][count] = props
            P_points.append(P + current_dP)
            T_points.append(T + dT)
            count+=1

        return np.array(P_points), np.array(T_points)  
    
    # ----------- Functions for finding phase boundaries-----------

    def initial_search(self, P_array, T_array):
        print(f"Searching at P : {P_array[0]} Kbar, between {T_array[0]} and {T_array[-1]} C")
        out_m = self.MAGEMin_C.multi_point_minimization(P_array, T_array,  self.data, X=self.X, Xoxides=self.Xoxides, sys_in=self.sys_in)
        solidus_T_interval = None
        liquidus_T_interval = None
        
        # Track if we've seen any melt
        seen_melt = False
        
        for i in range(len(out_m)):
            melt_frac = self.check_melt_frac(out_m[i])
            
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
            if self.check_melt_frac(out_m[-1]) >= 1:
                liquidus_T_interval = (T_array[-2], T_array[-1])
                
        print(f"Found intervals - Solidus: {solidus_T_interval}, Liquidus: {liquidus_T_interval}")
        return solidus_T_interval, liquidus_T_interval

    def binary_search_boundary(self, P, T_low, T_high, condition_func, boundary_name, tolerance=0.1):
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

    def find_phase_boundaries(self, P, T_range=(800, 2500), tolerance=1, n_points_initial=5):
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

        Ts_int, Tl_int = self.initial_search(P_array, T_array)
        
        # Binary search for solidus
        solidus_T = self.binary_search_boundary(
            P, Ts_int[0], Ts_int[1],
            lambda t: self.check_melt_frac(self.MAGEMin_C.single_point_minimization(P, t, self.data, X=self.X, Xoxides=self.Xoxides, sys_in=self.sys_in)) > 0,
            'solidus',
            tolerance
        )
        
        # Binary search for liquidus
        liquidus_T = self.binary_search_boundary(
            P, Tl_int[0], Tl_int[1],
            lambda t: self.check_melt_frac(self.MAGEMin_C.single_point_minimization(P, t, self.data, X=self.X, Xoxides=self.Xoxides, sys_in=self.sys_in)) >= 1,
            'liquidus',
            tolerance
        )
        
        return solidus_T, liquidus_T
    
    def get_solidus_liquidus_curves(self, P_range, T_range,P_points=30,T_points_initial=5):
        # ----------- Find and Plot Phase Boundaries Adiabats-----------
        print(f"Finding phase boundaries")
        pressures = np.linspace(P_range[0], P_range[1], P_points)  # 11 points from 5 to 15 Kbar
        solidus_points = []
        liquidus_points = []

        for P in pressures:
            sol_T, liq_T = self.find_phase_boundaries(P, T_range=T_range, tolerance=0.1, n_points_initial=T_points_initial)
            if sol_T is not None:
                solidus_points.append((P, sol_T))
                liquidus_points.append((P, liq_T))

        self.solidus_points = np.array(solidus_points)
        self.liquidus_points = np.array(liquidus_points)

        # Create interpolation functions for phase boundaries as lambda functions
        self.T_solidus = lambda P: float(interp1d(self.solidus_points[:,0], self.solidus_points[:,1], kind='cubic')(P))
        self.T_liquidus = lambda P: float(interp1d(self.liquidus_points[:,0], self.liquidus_points[:,1], kind='cubic')(P))

        # Create finer pressure array for smooth curves (for plotting)
        P_fine = np.linspace(self.solidus_points[:,0].min(), self.solidus_points[:,0].max(), 200)
        T_solidus = [self.T_solidus(p) for p in P_fine]
        T_liquidus = [self.T_liquidus(p) for p in P_fine]

        print("Created solidus and liquidus curves")

        return P_fine, T_solidus, T_liquidus
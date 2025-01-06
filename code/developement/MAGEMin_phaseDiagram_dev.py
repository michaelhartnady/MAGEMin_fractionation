"""
===============
MAGEMin Phase Diagram
===============

the MAGEMin_phaseDiagram class provides the MAGEMin phase diagram classes. At the moment it consists of
a single phase diagram class used to find P,T points for the solidus and liquidus and parameterize the phase boundaries.

"""

import numpy as np
from molmass import Formula
import pandas as pd
from tqdm import tqdm  # Add this import
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import juliacall
from juliacall import Main as jl, convert as jlconvert

# Explicitly add MAGEMin_C to the current environment
#jl.seval('import Pkg; Pkg.add("MAGEMin_C")') run this line to install the package in your local julia virtual environment
#jl.seval('using MAGEMin_C')
# Create the module reference
#MAGEMin_C = jl.MAGEMin_C

class phaseDiagram:
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
    
    def check_melt_frac(self, out):
        if 'liq' in out.ph:
            idx = out.ph.index('liq')
            return out.ph_frac[idx]
        else:
            return 0

    def initial_search(self, P_array, T_array):
        print(f"Searching at P : {P_array[0]} Kbar, between {T_array[0]} and {T_array[-1]} C")
        n_points = len(T_array)
        out_m = self.MAGEMin_C.multi_point_minimization(P_array, T_array,  self.data, X=self.X, Xoxides=self.Xoxides, sys_in=self.sys_in)
        solidus_T_interval = None
        liquidus_T_interval = None

        melt_fracs = np.array([self.check_melt_frac(out_m[i]) for i in range(n_points)])
        # Find first occurrence of melt (solidus)
        melt_indices = np.where(melt_fracs > 0)[0]
        solidus_idx = melt_indices[0] if len(melt_indices) > 0 else None

        # Find first occurrence of complete melting (liquidus)
        liquidus_indices = np.where(melt_fracs >= 1)[0]
        liquidus_idx = liquidus_indices[0] if len(liquidus_indices) > 0 else None
        
        solidus_T_interval = None if solidus_idx is None else (T_array[solidus_idx-1], T_array[solidus_idx])
        liquidus_T_interval = None if liquidus_idx is None else (T_array[liquidus_idx-1], T_array[liquidus_idx])
        
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
        # Cache single point minimization function
        min_func = self.MAGEMin_C.single_point_minimization
        
        T_array = jlconvert(jl.Vector[jl.Float64], np.linspace(T_range[0], T_range[1], n_points_initial))
        P_array = jlconvert(jl.Vector[jl.Float64], np.full(n_points_initial, P))

        Ts_int, Tl_int = self.initial_search(P_array, T_array)

        if Ts_int is None:
            return None, None
        
        # Cache parameters for repeated calls
        params = {
            'X': self.X,
            'Xoxides': self.Xoxides,
            'sys_in': self.sys_in
        }
        
        # Define condition functions with cached parameters
        def solidus_condition(t):
            return self.check_melt_frac(min_func(P, t, self.data, **params)) > 0
    
        def liquidus_condition(t):
            return self.check_melt_frac(min_func(P, t, self.data, **params)) >= 1
    
        # Binary search for boundaries
        solidus_T = self.binary_search_boundary(P, Ts_int[0], Ts_int[1], 
                                          solidus_condition, 'solidus', tolerance)
        liquidus_T = self.binary_search_boundary(P, Tl_int[0], Tl_int[1], 
                                           liquidus_condition, 'liquidus', tolerance)
        
        return solidus_T, liquidus_T
    
    def parameterize_phase_boundaries(self, P_range, T_range,P_points=30,T_points_initial=5):
        # ----------- Find and Plot Phase Boundaries Adiabats-----------
        print(f"Finding phase boundaries")
        pressures = np.linspace(P_range[0], P_range[1], P_points)  # 11 points from 5 to 15 Kbar
      
        def process_pressure(P):
            try:
                sol_T, liq_T = self.find_phase_boundaries(P, T_range=T_range, 
                                                        tolerance=0.1, 
                                                        n_points_initial=T_points_initial)
                return (P, sol_T, liq_T) if sol_T is not None else None
            except Exception as e:
                print(f"Error at P={P}: {str(e)}")
                return None

        # Sequential processing with progress bar (for debugging)
        results = []
        for P in pressures:
            result = process_pressure(P)
            results.append(result)

        # Filter valid results and store
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            return None, None, None
        
        self.solidus_points = np.array([(r[0], r[1]) for r in valid_results])
        self.liquidus_points = np.array([(r[0], r[2]) for r in valid_results])

        # Create interpolation functions for phase boundaries as lambda functions
        self.T_solidus = lambda P: float(interp1d(self.solidus_points[:,0], self.solidus_points[:,1], kind='cubic')(P))
        self.T_liquidus = lambda P: float(interp1d(self.liquidus_points[:,0], self.liquidus_points[:,1], kind='cubic')(P))

        # Generate smooth curves for plotting
        P_fine = np.linspace(self.solidus_points[:,0].min(), 
                        self.solidus_points[:,0].max(), 200)
        T_solidus = np.array([self.T_solidus(p) for p in P_fine])
        T_liquidus = np.array([self.T_liquidus(p) for p in P_fine])

        print("Created solidus and liquidus curves")

        return P_fine, T_solidus, T_liquidus


oxides = ['SiO2', 'Al2O3', 'CaO',  'MgO', 'FeO', 'K2O', 'Na2O', 'TiO2', 'O' , 'Cr2O3' , 'H2O']
values = [39.7918,1.999,2.9166, 49.2832, 5.5007, 0.0055, 0.0418, 0.1104, 0.0825, 0.1706,  0.0]
sys_in  = "mol"


phaseDiagram = phaseDiagram(oxides, values, sys_in, dataset="ig")

P_fine, T_solidus, T_liquidus = phaseDiagram.parameterize_phase_boundaries(P_range=[0,100], T_range=[800,2500],P_points=20,T_points_initial=5)
phaseDiagram.plot_phase_boundaries(P_fine, T_solidus, T_liquidus)
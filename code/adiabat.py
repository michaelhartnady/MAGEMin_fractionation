
"""
===============
MAGEMin Adiabats
===============

the MAGEMin_adiabat_class module provides the MAGEMin adiabatic classes. At the moment it consists of
a single adiabatic class- a simple 1D adiabatic class.

"""

import numpy as np
from molmass import Formula
from scipy.interpolate import interp1d
from phaseDiagram import phaseDiagram
import juliacall
from juliacall import Main as jl, convert as jlconvert

# Explicitly add MAGEMin_C to the current environment
#jl.seval('import Pkg; Pkg.add("MAGEMin_C")') run this line to install the package in your local julia virtual environment
#jl.seval('using MAGEMin_C')
# Create the module reference
#MAGEMin_C = jl.MAGEMin_C

class mantle_adiabat(phaseDiagram):
    def __init__(self,Xoxides, X, sys_in = 'mol', dataset = "ig",T_solidus=None,T_liquidus=None,solidus_points=None,liquidus_points=None):
        super().__init__(Xoxides, X, sys_in = 'mol', dataset = "ig",T_solidus=None,T_liquidus=None,solidus_points=None,liquidus_points=None)
        self.adiabat_data = {}
    
    def composition_to_molarMass(self,composition):
        model_composition = dict(zip(self.Xoxides,composition)) # oxides, wtfractions 
        total_moles = sum((val / Formula(ox).mass) for ox, val  in model_composition.items())
        Molar_mass = 1 / total_moles
        return Molar_mass

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
    
    def get_data_dictionary(self,out):
        data_dictionary = {}    
        for i in range(len(out.ph)):
            data_dictionary[out.ph[i]] = {}
            data_dictionary[out.ph[i]]['weight_fraction'] = out.ph_frac_wt[i]
            data_dictionary[out.ph[i]]['enthalpy'] = out.SS_vec[i].enthalpy / 0.1
            data_dictionary[out.ph[i]]['entropy'] = out.SS_vec[i].entropy / 0.1
            data_dictionary[out.ph[i]]['alpha'] = out.SS_vec[i].alpha
            data_dictionary[out.ph[i]]['density'] = out.SS_vec[i].rho
            data_dictionary[out.ph[i]]['Cp'] = (out.SS_vec[i].cp * 1e3) / 0.1
            data_dictionary[out.ph[i]]['molar_mass'] = self.composition_to_molarMass(out.SS_vec[i].Comp_wt)
            data_dictionary[out.ph[i]]['specific_volume'] = 1 / data_dictionary[out.ph[i]]['density'] # m3 / kg
            data_dictionary[out.ph[i]]['volume'] = out.SS_vec[i].V * 1e-3 / data_dictionary[out.ph[i]]['molar_mass'] # m3
        return data_dictionary
    
    def get_enthalpy_data(self,out):

        solid_specific_entropy = 0
        liquid_specific_entropy = 0
        solid_specific_volume = 0
        liquid_specific_volume = 0

        for phase, type,idx in zip(out.ph, out.ph_type,out.ph_id):
                print(f"Phase: {phase}, Type: {type}, ID: {idx}")
                if type == 1:
                    if phase != 'liq':
                        print(f"Entropy of {phase} : {out.SS_vec[idx].entropy / 0.1} * {out.ph_frac_wt[idx]} = { (out.SS_vec[idx].entropy/0.1) * out.ph_frac_wt[idx]} J/K")
                        solid_specific_entropy += ((out.SS_vec[idx].entropy/0.1) * out.ph_frac_wt[idx])
                        solid_specific_volume += ((1/out.SS_vec[idx].rho) * out.ph_frac_wt[idx])
                    else:
                        print(f"Entropy of {phase} : {out.SS_vec[idx].entropy / 0.1} J/K")
                        liquid_specific_entropy = (out.SS_vec[idx].entropy/0.1)
                        liquid_specific_volume = ((1/out.SS_vec[idx].rho)* out.ph_frac_wt[idx])
                elif type == 0:
                    print(f"Entropy of {phase} : {out.SS_vec[idx].entropy / 0.1} * {out.ph_frac_wt[idx]} = { (out.SS_vec[idx].entropy/0.1) * out.ph_frac_wt[idx]} J/K")
                    solid_specific_entropy += ((out.PP_vec[idx].entropy/0.1) * out.ph_frac_wt[idx])
                    solid_specific_volume += ((1/out.PP_vec[idx].rho) * out.ph_frac_wt[idx])

        print(f"Entropy Difference: {liquid_specific_entropy} - {solid_specific_entropy} = {liquid_specific_entropy - solid_specific_entropy}")

        return solid_specific_entropy, liquid_specific_entropy, solid_specific_volume, liquid_specific_volume

    
    def calculate_enthalpy_of_fusion(self, current_out, next_out, T, P, dP):

        current_solid_specific_entropy, current_liquid_specific_entropy, current_solid_specific_volume, current_liquid_specific_volume = self.get_enthalpy_data(current_out)
        next_solid_specific_entropy, next_liquid_specific_entropy, next_solid_specific_volume, next_liquid_specific_volume = self.get_enthalpy_data(next_out)

        print(f"Entropy Difference: {current_liquid_specific_entropy} - {current_solid_specific_entropy} = {current_liquid_specific_entropy - current_solid_specific_entropy}")
       
        v_liquid_list = [current_liquid_specific_volume, next_liquid_specific_volume]
        v_solid_list = [current_solid_specific_volume, next_solid_specific_volume]
        
        pressure_volume_integral_term = 0
        P_list = [P,P+dP]
        n = len(P_list)
        v_liquid_list = [current_liquid_specific_volume, next_liquid_specific_volume]
        v_solid_list = [current_solid_specific_volume, next_solid_specific_volume]

        for i in range(n - 1):
            # 1) Average of (v_liquid - v_solid) at endpoints
            val_i     = v_liquid_list[i]     - v_solid_list[i]
            val_ip1   = v_liquid_list[i + 1] - v_solid_list[i + 1]
            avg_value = 0.5 * (val_i + val_ip1)

            # 2) Pressure difference in Pa (since P is given in GPa)
            dP = (P_list[i + 1] - P_list[i]) * 1.0e8  # 1 Kbar = 1e8 Pa

            # 3) Trapezoid contribution
            pressure_volume_integral_term += avg_value * dP
            
        #print(f"Integral Term: {pressure_volume_integral_term} J/kg")

        dS = current_liquid_specific_entropy - current_solid_specific_entropy
        dH = (T * dS) - pressure_volume_integral_term
        dH2 = (T * dS)
        
        return dH2

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
        phases, phases_frac = self.get_phases(out)

        return alpha, rho, Cp, melt_frac, phases, phases_frac

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
        #print(f"Calculating material properties at {P: .2f} Kbar, {T-273.15: .1f} C")
        out = self.MAGEMin_C.single_point_minimization(jlconvert(jl.Float64, P), jlconvert(jl.Float64, T-273.15), self.data, X=self.X, Xoxides=self.Xoxides, sys_in=self.sys_in)
        alpha, rho, Cp, melt_frac, phases, phases_frac = self.calculate_material_properties(out)
        
        
        # Base adiabatic term
        adiabatic_term = 1e8 * (alpha * T) / (rho * Cp) # factor of 1e8 converts Pa to Kbar
        
        # Only add latent heat effect if we're actually in the two-phase region
        properties = {'P': P, 'T': T-273.15, 'dP': dP, 'dT': adiabatic_term*dP,'alpha': alpha, 'rho': rho, 'Cp': Cp, 'latent heat': None, 'F': melt_frac, 'phases': phases, 'phases_frac': phases_frac}
        if melt_frac > 0 and melt_frac < 1: 
            print(f"Melt & Solids present at {P: .2f} Kbar, {T-273.15: .1f} C, calculating H & dF/dP")
            #dP = 0.1  # Use smaller step for derivative calculation
            dT = adiabatic_term * dP

            out_dP = self.MAGEMin_C.single_point_minimization(jlconvert(jl.Float64, P + dP), jlconvert(jl.Float64, T-273.15), self.data, X=self.X, Xoxides=self.Xoxides, sys_in=self.sys_in)

            Hf = self.calculate_enthalpy_of_fusion(out, out_dP, T, P, dP)
            print(f"Latent heat: {Hf} J/kg")
       
            # Calculate melt fraction change
            F_p = melt_frac
            F_dp = self.check_melt_frac(out_dP)
            dF_dP = (F_dp - F_p) / dP            
            # Scale down latent heat term significantly
            scaling_factor = latent_scaling_factor
            latent_term = -(Hf/Cp) * dF_dP * scaling_factor
            
            #print(f"At T={T:.1f}, P={P:.1f}:")
            #print(f"  Adiabatic term: {adiabatic_term:.2f} K/GPa")
            #print(f"  Latent term: {latent_term:.2f} K/GPa")
            properties['dP'] = dP
            properties['dT'] = adiabatic_term*dP
            properties['latent heat'] = Hf
            
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
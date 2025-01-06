from molmass import Formula
class ThermoBulk:
    
    chemical_systems = {'NCKFMASHTO': ['SiO2','TiO2', 'Al2O3','FeO','MgO','CaO','Na2O','K2O', 'O2','H2O'],
                    'NCKFMASHTOCr': ['SiO2','TiO2', 'Al2O3', 'Cr2O3','FeO', 'MgO','CaO','Na2O','K2O', 'O2','H2O'],
                    'MnNCKFMASHTOCr': ['SiO2','TiO2', 'Al2O3','Cr2O3','FeO','MnO', 'MgO','CaO','Na2O','K2O', 'O2','H2O'],
                    'KFMASH': ['SiO2','Al2O3','FeO','MgO','K2O','H2O'],
                    }
    
    def __init__(self):
        """
        Initialize the BulkComposition object with default values for chemical system,
        oxides, formula masses, bulk composition, mole fractions, and various modeling outputs.
        """
        self.chemical_system = None
        self.oxides = None
        self.formula_masses = None
        self.bulk_composition = {'SiO2' : None, 'TiO2' : None, 'Al2O3' : None, 'Cr2O3' : None, 'Fe2O3' : None, 'FeO' : None, 'MnO' : None, 'MgO' : None, 'CaO' : None, 'Na2O' : None, 'K2O' : None, 'P2O5' : None, 'H2O' : None, "O2": None}
        self.formula_masses = {oxide: Formula(oxide).mass for oxide in self.bulk_composition}
        self.mole_fractions = self.oxides
        self.XFe3 = None
        self.thermocalc = {}
        self.perplex = {}
        self.theriak_domino = {}
        
    def get_bulk_compositions(self, composition, chemical_system, XFe3=0.1, apatite_correction=True):
        """
        Set the bulk composition based on input composition and chemical system.

        Parameters:
        composition (pd.Series or dict): The input composition in weight percent oxides.
        chemical_system (str): The chemical system to use (e.g., 'NCKFMASHTOCr').
        XFe3 (float): The Fe3+/Fetotal ratio, default is 0.1.
        apatite_correction (bool): Whether to perform apatite correction, default is True.

        Returns:
        None
        """
        self.XFe3 = XFe3
        self.oxides = {oxide: None for oxide in self.chemical_systems[chemical_system] }
        
        composition = composition.to_dict()

        for oxide in self.bulk_composition.keys():
            try:
                self.bulk_composition[oxide] = composition[oxide]
                
                if oxide in self.oxides.keys():
                    self.oxides[oxide] = composition[oxide]
                else:
                    pass
            except:
                print(f"{oxide} was not provided")
        
        self.oxides["O2"] = self.calculate_O2(self.bulk_composition["FeO"])
        self.oxides["CaO"] = self.perform_apatite_correction(self.bulk_composition["CaO"],self.bulk_composition["P2O5"])
        self.mole_fractions =  {oxide : self.oxides[oxide]/self.formula_masses[oxide] for oxide in self.oxides.keys()}

        #self.cations_in_formula(list(self.oxides.keys()))

        self.thermocalc_bulk()
        self.perplex_bulk()
        self.theriakdomino_bulk()

    def perform_apatite_correction(self,CaO,P2O5):
        """
        Perform apatite correction on CaO.

        This method corrects the CaO content by removing the amount of CaO
        that would be incorporated into apatite. The correction is based on
        the stoichiometry of fluorapatite (Ca5(PO4)3F).

        Formula used:
        CaO_corrected = CaO_total - (3.33 * P2O5)

        Where:
        - 3.33 is the molar ratio of CaO to P2O5 in apatite (5 Ca : 3 P)
        - CaO and P2O5 are in molar quantities

        Parameters:
        CaO (float): The weight percent of CaO.
        P2O5 (float): The weight percent of P2O5.

        Returns:
        float: The corrected CaO value in weight percent.
        """
        return ((CaO/self.formula_masses["CaO"]) - 3.333*(P2O5/self.formula_masses["P2O5"])) * self.formula_masses["CaO"]
    
    def calculate_FeOT():
        return
    
    def anhydrous_normalization(self, input_composition):
        """
        Normalize the bulk composition to anhydrous conditions.

        Returns:
        dict: The adjusted bulk composition in mole percent oxides.
        """
        for oxide in input_composition.keys():
            if oxide == "H2O":
                input_composition[oxide] = 0
            else:
                pass
        
        total_moles = sum(input_composition.values())
        
        normalized_composition = {oxide : 100*(input_composition[oxide]/total_moles) for oxide in input_composition.keys()}
        print(normalized_composition)
        print(sum(normalized_composition.values()))

        return normalized_composition

    def calculate_O2(self,FeO):
        """
        Calculate the O2 content based on FeO and XFe3.

        This method calculates the amount of O2 needed to oxidize a portion of FeO to Fe2O3,
        based on the specified Fe3+/Fetotal ratio (XFe3).

        Formula used:
        O2 = FeO * (XFe3 / 4) * (M_O2 / M_FeO)

        Where:
        - FeO is the weight percent of FeO
        - XFe3 is the Fe3+/Fetotal ratio
        - M_O2 is the molar mass of O2
        - M_FeO is the molar mass of FeO
        - The factor 1/4 comes from the stoichiometry of the oxidation reaction:
        4FeO + O2 -> 2Fe2O3

        Parameters:
        FeO (float): The weight percent of FeO.

        Returns:
        float: The calculated O2 content in weight percent.
        """
        return FeO *(self.XFe3)/self.formula_masses["FeO"]/4*Formula("O2").mass

    def thermocalc_bulk(self):
        """
        Calculate the bulk composition for Thermocalc.

        Returns:
        dict: The Thermocalc bulk composition in mole percent oxides
        """
        thermocalc = self.mole_fractions.copy()
        thermocalc["O2"] = thermocalc["O2"] * 2
        thermocalc["O"] = thermocalc.pop("O2","O")
        total_moles =  sum(value for oxide, value in thermocalc.items() if oxide != 'H2O')
        H2O_mol_pct = 100 * self.mole_fractions["H2O"] / (total_moles+self.mole_fractions["H2O"])
        self.thermocalc = {oxide : round((100-H2O_mol_pct)*thermocalc[oxide]/total_moles,4) for oxide in thermocalc.keys()}
        print(self.thermocalc)
        print(sum(self.thermocalc.values()))
        return self.thermocalc
    
    def perplex_bulk(self):
        """
        Calculate the bulk composition for Perplex.

        Returns:
        dict: The Perplex bulk composition in mole percent oxides
        """
        perplex = self.mole_fractions.copy()
        total_moles = sum(value for oxide, value in perplex.items() if oxide != 'H2O')
        H2O_mol_pct = 100 * self.mole_fractions["H2O"] / (total_moles+self.mole_fractions["H2O"])   
        self.perplex = {oxide :  round((100-H2O_mol_pct)*perplex[oxide]/total_moles,4) for oxide in perplex.keys() }
        return self.perplex
    
    def cations_in_formula(self, oxide_list):
        """
        Calculate the number of cations in each oxide formula.

        Parameters:
        oxide_list (list): A list of oxide formulas.

        Returns:
        dict: A dictionary with oxides as keys and their cation counts as values.
        """
        cation_counts = {}
        for oxide in oxide_list:
            formula = Formula(oxide)
            composition = formula.composition()
            cation = next((elem for elem in composition if elem != 'O'), None)
            cation_counts[oxide] = composition[cation].count if cation else 1  # For O2 or other oxides without cations
        return cation_counts
    

    def theriakdomino_bulk(self):
        """"
        Calculate the bulk composition for Theriak-Domino.

        Returns:
        dict: The Theriak-Domino bulk composition in mole percent of cations.
        """
        theriak = self.mole_fractions.copy()
        theriak["O2"] = theriak["O2"] * 2
        theriak["O"] = theriak.pop("O2","O")

        cation_moles = {}
        total_cations = 0

        for oxide, mole_fraction in self.mole_fractions.items():
            formula = Formula(oxide)
            composition = formula.composition()
            
            # Find the cation (assuming it's the first non-oxygen element)
            cation = next((elem for elem in composition if elem != 'O'), None)
            
            if cation:
                cation_count = composition[cation].count
                cation_moles[cation] = cation_moles.get(cation, 0) + mole_fraction * cation_count
                total_cations += mole_fraction * cation_count
            else:
                # Handle special cases like O2
                if oxide == 'O2':
                    cation_moles['O'] = cation_moles.get('O', 0) + mole_fraction * 2
                    total_cations += mole_fraction * 2

        self.theriak_domino = {cation: round((moles / total_cations) * 100,4) for cation, moles in cation_moles.items()}
        
        return self.theriak_domino
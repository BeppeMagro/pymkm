import matplotlib.pyplot as plt
import warnings

from pymkm.mktable.core import MKTableParameters, MKTable
from pymkm.io.table_set import StoppingPowerTableSet
from pymkm.sftable.core import SFTableParameters, SFTable

"""
This script demonstrates how to:
  - Load stopping power tables from the default MSTAR source ("mstar_3_12").
  - Store input parameters for specific energies computation for the 3(+1) 
    available MK models (modified MK [Inaniwa et al. 2010], 
    stochastic MK [Inaniwa et al. 2018], 
    oxygen-effect-incorporated SMK, [Inaniwa et al. 2021, Inaniwa et al. 2023]).
  - Compute survival curves using the 3(+1) models under hypoxic coniditions
    for fixed LET value.
  - Plot survival curves.
"""

warnings.filterwarnings("ignore", category=UserWarning)

def main():

    ## Select input parameters for specific energy tables generation
    atomic_number = 6 # C
    source = "mstar_3_12" # Source code used to generate stopping power tables (available with pymkm: fluka_2020_0, geant4_11_3_0 or mstar_3_12)
    model_name = "Kiefer-Chatterjee" # Amorphous track structure model (Kiefer-Chatterjee or Scholz-Kraft)
    core_type = "energy-dependent" # Core radius model ('constant' or 'energy-dependent')
    LET = 500 # Chosen LET value for survival fraction computation (MeV/cm)

    mkm_parameters = { # HSG parameters declared in [Inaniwa et al. 2018]
        "domain_radius": 0.29, # μm
        "nucleus_radius": 3.9, # μm
        "alpha0": 0.150, # 1/Gy
        "beta0": 0.0593, # 1/Gy^2
        "z0": 55.0 # Gy
    } 

    smk_parameters = { # HSG parameters declared in [Inaniwa et al. 2018]
        "domain_radius": 0.28, # μm
        "nucleus_radius": 8.1, # μm
        "alpha0": 0.174, # 1/Gy
        "beta0": 0.0568, # 1/Gy^2
        "z0": 66.0 # Gy
    }

    osmk_parameters = { # HSG parameters declared in [Inaniwa et al. 2021/2023]
        "domain_radius": 0.23, # μm
        "nucleus_radius": 8.1, # μm
        "alphaL": 0.0, # 1/Gy
        "alphaS": 0.21, # 1/Gy
        "beta0": 0.043, # 1/Gy^2
        "z0": 88.0, # Gy
        "K": 3, # mmHg
        "zR": 28.0, # Gy
        "gamma": 1.30,
        "Rm": 2.9,
        "f_rd_max": 3.00,
        "f_z0_max": 3.53,
        "Rmax": 4.46
    }
    pO2 =  0.0 # total hypoxia (mmHg)

    ## Load stopping power tables
    print(f"\nGenerating stopping power tables for ion Z = {[atomic_number]} (using source '{source}')...")
    sp_table_set = StoppingPowerTableSet.from_default_source(source).filter_by_ions([atomic_number])

    ## Store input parameters for MK and SMK tables
    mk_params = MKTableParameters(
        domain_radius=mkm_parameters["domain_radius"],
        nucleus_radius=mkm_parameters["nucleus_radius"],
        z0=mkm_parameters["z0"],
        beta0=mkm_parameters["beta0"],
        model_name=model_name,
        core_radius_type=core_type,
        use_stochastic_model=False
    )
    smk_params = MKTableParameters(
        domain_radius=smk_parameters["domain_radius"],
        nucleus_radius=smk_parameters["nucleus_radius"],
        z0=smk_parameters["z0"],
        beta0=smk_parameters["beta0"],
        model_name=model_name,
        core_radius_type=core_type,
        use_stochastic_model=True
    )
    osmk_params = MKTableParameters(
        domain_radius=osmk_parameters["domain_radius"],
        nucleus_radius=osmk_parameters["nucleus_radius"],
        z0=osmk_parameters["z0"],
        beta0=osmk_parameters["beta0"],
        model_name=model_name,
        core_radius_type=core_type,
        use_stochastic_model=True
    )


    ## Generate specific energy tables
    print(f"\nGenerating tables for ion Z = {atomic_number} (using source '{source}')...")
    mk_table = MKTable(parameters=mk_params, sp_table_set=sp_table_set)
    smk_table = MKTable(parameters=smk_params, sp_table_set=sp_table_set)
    osmk_table = MKTable(parameters=osmk_params, sp_table_set=sp_table_set)

    ## Compute survival curves for hypoxic and normoxic conditions adopting
    ## the different models. Curves are stored in a dictionary, where for each 
    ## pO2 level key all the curves are stored
    sf_tables = {}

    # Shared input parameters for SF comptuation among OSMK models
    common_OSMK_params = {
        "mktable": osmk_table,
        "alphaS": osmk_parameters["alphaS"],
        "alphaL": osmk_parameters["alphaL"],
        "beta0": osmk_parameters["beta0"],
        "K": osmk_parameters["K"],
        "pO2": pO2
    }

    # Prepare model-specific parameters
    model_configs = [
        {
            "model": "MKM",
            "params": {"mktable": mk_table, "alpha0": mkm_parameters["alpha0"], "beta0": mkm_parameters["beta0"]}
        },
        {
            "model": "SMK",
            "params": {"mktable": smk_table, "alpha0": smk_parameters["alpha0"], "beta0": smk_parameters["beta0"]}
        },
        {
            "model": "OSMK2021",
            "params": {**common_OSMK_params, "zR": osmk_parameters["zR"], "gamma": osmk_parameters["gamma"], "Rm": osmk_parameters["Rm"]}
        },
        {
            "model": "OSMK2023",
            "params": {**common_OSMK_params, "f_rd_max": osmk_parameters["f_rd_max"], "f_z0_max": osmk_parameters["f_z0_max"], "Rmax": osmk_parameters["Rmax"]}
        }
    ]

    #  Store parameters for survival fraction calculation for all models
    for config in model_configs:
        sf_params = SFTableParameters(**config["params"])
        sf_table = SFTable(parameters=sf_params)
        sf_table.compute(ion=atomic_number, let=LET, force_recompute=True, apply_oxygen_effect=True if "OSMK" in config["model"] else False)
        sf_tables[config["model"]] = sf_table.table[0]

    ## Plot settings
    _, ax = plt.subplots(figsize=(8, 6))
    color = sp_table_set.get(atomic_number).color
    linestyle_map = {
        "MKM": "-",
        "SMK": "",
        "OSMK2021": ":",
        "OSMK2023": ""
    }
    alpha_map = {
        "MKM": 0.4,
        "SMK": 1,
        "OSMK2021": 0.4,
        "OSMK2023": 1
    }
    marker_map = {
        "MKM": None,
        "SMK": "s",
        "OSMK2021": None,
        "OSMK2023": "o"
    }

    ## Plot specific energies result using built-in method
    for model, sf_table in sf_tables.items():
        linestyle = linestyle_map.get(model, "-")
        marker = marker_map.get(model, None)
        alpha = alpha_map.get(model, 1.0)
        # Assuming sf_table has attributes 'dose' and 'survival' for plotting
        dose = sf_table["data"]["dose"]
        survival = sf_table["data"]["survival_fraction"]
        ax.plot(dose, survival, 
                label=f"{model}", 
                color=color, 
                linestyle=linestyle, 
                lw=5,
                alpha=alpha,
                marker=marker,
                markersize=8 if marker else 0)
            
    ax.set_xlim(0, 8)
    ax.set_ylim(1e-4, 1)
    ax.set_yscale("log")
    ax.set_xlabel("Dose [Gy]")
    ax.set_ylabel("Survival fraction")
    ax.set_title(f"LET = {LET/10:.1f} keV/μm", fontsize=14)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(loc='upper right', fontsize=12)
            
    plt.show()

if __name__ == "__main__":
    main()

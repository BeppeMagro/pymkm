import numpy as np
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from pymkm.mktable.core import MKTableParameters, MKTable
from pymkm.sftable.core import SFTableParameters, SFTable
from pymkm.io.table_set import StoppingPowerTableSet
from pymkm.utils.parallel import optimal_worker_count
from validation_utils.inverse_dose import inverse_dose_from_survival

"""
Example usage of MKTableParameters to compute specific energies (z_d*, z_d and z_n*) tables
for the oxygen-effect-incorporated stochastic MK (OSMK) model [Inaniwa et al. 2021] (be aware
that the introduction of hypoxia effects does not affect the table generation, but can only affect
survival curves and, clearly, OER calculations).

This script demonstrates how to:
  - Load stopping power tables from the default MSTAR source ("mstar_3_12").
  - Store input parameters (no specific energies computation required).
  - Compute the Oxygen Enhancement Ratio (OER) for a given cell line as a function of LET. This calculation
    requires the redundant calculation of survival curves for both normoxic (here, corresponding to
    pO2 = 160mmhg, as in Inaniwa et al. 2021) and hypoxic conditions (here, pO2 = 0 mmHg) for many LET levels.
  - Plot OER trends as a function of LET.
"""

warnings.filterwarnings("ignore", category=UserWarning)

def compute_d10_pair(sf_table: SFTable, Z: int, energy: np.ndarray, let: np.ndarray):
    """
    Compute survival fractions for a given ion and energy/LET.
    Returns input energy and LET, and corresponding survival curve (i.e. dose and survival points as numpy arrays).
    """
    sf_table.compute(ion=Z, 
                     energy=energy, 
                     let=let, 
                     force_recompute=True, 
                     apply_oxygen_effect=True)
    
    entry = sf_table.table[0] # first element corresponds to the first (and only) energy-LET pair
    dose = entry["data"]["dose"]
    survival = entry["data"]["survival_fraction"]

    return energy, let, dose, survival

def inverse_dose_from_survival(doses: np.ndarray, survivals: np.ndarray, S_target: float) -> float:
    """
    Estimate the radiation dose corresponding to a given survival fraction using
    numpy linear interpolation (assumes that the survival curve di monotonic).
    """
    doses = np.asarray(doses)
    survivals = np.asarray(survivals)
    if not (0 < S_target <= 1):
        raise ValueError("S_target must be in the interval (0, 1].")
    if not np.all(np.diff(doses) > 0):
        raise ValueError("Dose values must be strictly increasing.")
    
    # Determine monotonicity
    if np.all(np.diff(survivals) < 0):  # survival is decreasing
        inv_dose = np.interp(S_target, survivals[::-1], doses[::-1])
    elif np.all(np.diff(survivals) > 0):  # survival is increasing
        inv_dose = np.interp(S_target, survivals, doses)
    else:
        raise ValueError("Survival curve must be monotonic for interpolation.")
    
    return inv_dose

def main():

    ## Select input parameters for specific energy tables generation
    cell_type = "HSG"
    atomic_number = 6 # C
    source = "mstar_3_12" # Source code used to generate stopping power tables (available with pymkm: fluka_2020_0, geant4_11_3_0 or mstar_3_12)
    model_name = "Kiefer-Chatterjee" # Amorphous track structure model (Kiefer-Chatterjee or Scholz-Kraft)
    core_type = "energy-dependent" # Core radius model ('constant' or 'energy-dependent')
    domain_radius = 0.23 # μm
    nucleus_radius = 8.1 # μm
    alphaL = 0.0 # 1/Gy
    alphaS = 0.21 # 1/Gy
    beta0 = 0.043 # 1/Gy^2
    z0 = 88.0 # Gy
    K = 3 # mmHg
    zR = 28.0 # Gy
    gamma = 1.30 
    Rm = 2.9
    pO2 = {
        "hypoxia": 0.0, # mmHg
        "normoxia": 160.0 # mmHg
    }

    ## Load stopping power tables
    print(f"\nGenerating stopping power tables for ion Z = {atomic_number} (using source '{source}')...")
    sp_table_set = StoppingPowerTableSet.from_default_source(source).filter_by_ions([atomic_number])
    sp_table = sp_table_set.get(atomic_number)  # Only one ion is used
    # Quantities useful for later OER calculation
    energy = sp_table.energy
    LET = sp_table.let
    energyLETpairs = list(zip(energy, LET))
    worker_count = optimal_worker_count(energyLETpairs)

    ## Store input parameters
    params = MKTableParameters(
        domain_radius=domain_radius,
        nucleus_radius=nucleus_radius,
        z0=z0,
        beta0=beta0,
        model_name=model_name,
        core_radius_type=core_type,
        use_stochastic_model=True
    )

    ## Generate specific energies table
    print(f"\nGenerating OSMK tables for ion Z = {atomic_number} (using source '{source}')...")
    osmk_table = MKTable(parameters=params, sp_table_set=sp_table_set)

    ## OER is a ratio of survival fractions in hypoxic and normoxic conditions. Its calculation requires
    ## the redundante calculation of survival fractions for both conditions.
    doseFor10Survival = {}
    for status, po2 in pO2.items():

        # Store parameters for survival fraction calculation
        sf_params = SFTableParameters(
            mktable=osmk_table,
            beta0=beta0,
            alphaS=alphaS,
            alphaL=alphaL,
            pO2=po2,
            K=K,
            zR=zR,
            gamma=gamma,
            Rm=Rm
        )
        sf_table = SFTable(parameters=sf_params)

        # Compute survival fractions for all energy-LET pairs
        print(f"\nCalculating survival fractions for ion Z = {atomic_number}, assuming pO₂ = {po2:.1f}...")
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            results = list(
                        tqdm(
                            executor.map(partial(compute_d10_pair, sf_table, atomic_number), *zip(*energyLETpairs)),
                            total=len(energyLETpairs),
                            unit="energy"
                        ))

        d10 = np.array([inverse_dose_from_survival(dose, survival, S_target=0.1) for _, _, dose, survival in results])
        doseFor10Survival[status] = d10

    OER = doseFor10Survival["hypoxia"] / doseFor10Survival["normoxia"]

    ## Plot OER as a function of LET
    _, ax = plt.subplots()
    ax.plot(LET, OER, label=f"OER (pO₂ = {pO2['hypoxia']:.1f} mmHg)", color=sp_table.color, linewidth=5, alpha=0.5)
    ax.set_xscale("log")
    ax.set_yscale("linear")
    ax.set_xlim(1E2, 1E4)
    ax.set_ylim(1.0, max(OER) * 1.1)
    ax.set_xlabel("dose-averaged LET [MeV/cm]")
    ax.set_ylabel("OER (10% Survival)")
    ax.set_grid(alpha=0.4)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

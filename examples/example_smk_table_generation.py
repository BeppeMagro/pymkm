import matplotlib.pyplot as plt

from pymkm.mktable.core import MKTableParameters, MKTable
from pymkm.io.table_set import StoppingPowerTableSet

"""
Example usage of MKTableParameters to compute and visualize specific energies (z_d*, z_d and z_n*) tables
for the stochastic MK (SMK) model [Inaniwa et al. 2018].

This script demonstrates how to:
  - Load stopping power tables from the default MSTAR source ("mstar_3_12").
  - Store input parameters for specific energies computation.
  - Compute specific energies z_d*, z_d and z_n.
  - Plot the specific energy curve for different ions.
  - Write the table to a .txt file.
"""

def main():

    ## Select input parameters for specific energy tables generation
    cell_type = "HSG"
    atomic_numbers = [2, 6, 8] # He, C, O
    source = "mstar_3_12" # Source code used to generate stopping power tables (available with pymkm: fluka_2020_0, geant4_11_3_0 or mstar_3_12)
    model_name = "Kiefer-Chatterjee" # Amorphous track structure model (Kiefer-Chatterjee or Scholz-Kraft)
    core_type = "energy-dependent" # Core radius model ('constant' or 'energy-dependent')
    domain_radius = 0.23 # μm
    nucleus_radius = 8.1 # μm
    alpha0 = 0.12 # 1/Gy
    beta0 = 0.043 # 1/Gy^2
    z0 = 88.0 # Gy
    alpha_ref = 0.12 # 1/Gy
    beta_ref = 0.0615 # 1/Gy^2
    clinical_scale_factor = 1.0

    ## Load stopping power tables
    print(f"\nGenerating stopping power tables for ion Z = {atomic_numbers} (using source '{source}')...")
    sp_table_set = StoppingPowerTableSet.from_default_source(source).filter_by_ions(atomic_numbers)

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
    print(f"\nGenerating SMK tables for ion Z = {atomic_numbers} (using source '{source}')...")
    smk_table = MKTable(parameters=params, sp_table_set=sp_table_set)
    smk_table.compute(ions=atomic_numbers, parallel=True)

    ## Plot specific energies result using built-in method
    _, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    smk_table.plot(ions=atomic_numbers, x="energy", y="z_bar_star_domain", verbose=True, ax=axes[0])
    smk_table.plot(ions=atomic_numbers, x="energy", y="z_bar_domain", verbose=True, ax=axes[1])
    smk_table.plot(ions=atomic_numbers, x="energy", y="z_bar_nucleus", verbose=True, ax=axes[2])
    plt.tight_layout()
    plt.show()

    ## Write the SMKTable to a .txt file
    path = "./SMK_table.txt"
    params = {
        "CellType": cell_type,
        "Alpha_ref": alpha_ref,
        "Beta_ref": beta_ref,
        "scale_factor": clinical_scale_factor,
        "Alpha0": alpha0,
        "Beta0": beta0
    }
    smk_table.write_txt(model="stochastic", params=params, filename=path, max_atomic_number=max(atomic_numbers))


if __name__ == "__main__":
    main()

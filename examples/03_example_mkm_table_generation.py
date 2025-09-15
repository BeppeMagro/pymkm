from pymkm.mktable.core import MKTableParameters, MKTable
from pymkm.io.table_set import StoppingPowerTableSet

"""
Example usage of MKTable to compute and visualize specific energy (z_d*) tables
for the modified-MK model [Inaniwa et al. 2010].

This script demonstrates how to:
  - Load stopping power tables from the default MSTAR source ("mstar_3_12").
  - Store input parameters for specific energies computation.
  - Compute specific energies z_d*.
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
    domain_radius = 0.32 # μm
    nucleus_radius = 3.9 # μm
    alpha0 = 0.172 # 1/Gy
    beta0 = 0.0615 # 1/Gy^2

    ## Load stopping power tables
    print(f"\nGenerating stopping power tables for ion Z = {atomic_numbers} (using source '{source}')...")
    sp_table_set = StoppingPowerTableSet.from_default_source(source).filter_by_ions(atomic_numbers)

    ## Store input parameters
    params = MKTableParameters(
        domain_radius=domain_radius,
        nucleus_radius=nucleus_radius,
        beta0=beta0,
        model_name=model_name,
        core_radius_type=core_type,
    )

    ## Generate specific energy table
    print(f"\nGenerating MKM tables for ion Z = {atomic_numbers} (using source '{source}')...")
    mk_table = MKTable(parameters=params, sp_table_set=sp_table_set)
    mk_table.compute(ions=atomic_numbers, parallel=True)

    ## Plot specific energies result using built-in method
    mk_table.plot(ions=atomic_numbers, x="energy", y="z_bar_star_domain", verbose=True)

    ## Write the MKTable to a .mkm file
    path = "./MKM_table.mkm"
    params = {
        "CellType": cell_type,
        "Alpha_0": alpha0,
        "Beta": beta0
    }
    mk_table.write_txt(params=params, filename=path, max_atomic_number=max(atomic_numbers))


if __name__ == "__main__":
    main()
